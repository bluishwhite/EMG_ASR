# -*- coding: utf-8 -*-
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.configs import default_configs, pretty_configs
from src.utils import auto_mkdir
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.utils.common_utils import *
from src.audio_helper import *
from src.metric.bleu_scorer import SacreBLEUScorer
import argparse
import yaml
import os
import random
import ntpath
import numpy as np


def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def train(nmt, configs, args):
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    global_step = 0

    print('Building Optimizer...')
    lrate = optimizer_configs['learning_rate']
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=nmt_model,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    if optimizer_configs['schedule_method'] is not None:

        if optimizer_configs['schedule_method'] == "loss":

            scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                                 **optimizer_configs["scheduler_configs"]
                                                 )

        elif optimizer_configs['schedule_method'] == "noam":
            scheduler = NoamScheduler(optimizer=optim, **optimizer_configs['scheduler_configs'])
        else:
            print("Unknown scheduler name {0}. Do not use lr_scheduling.".format(optimizer_configs['schedule_method']))
            scheduler = None
    else:
        scheduler = None

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.saveto, args.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    # best_model_prefix = os.path.join(args.saveto, args.model_name + ".best")
    # best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    print('start training...')
    best_cer = 100

    for iter in range(3000):
        total_stats = Statistics()
        training_iter = nmt.training_iterator.build_generator()
        batch_iter, total_iters = 0, len(nmt.training_iterator)
        for batch in training_iter:
            global_step = global_step + 1
            if optimizer_configs["schedule_method"] is not None \
                    and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=global_step)

            seqs_x, seqs_y = batch

            n_samples_t = len(seqs_x)
            #print(n_samples_t)
            batch_iter += n_samples_t
            n_words_t = sum(len(s) for s in seqs_y)

            lrate = list(optim.get_lrate())[0]
            optim.zero_grad()
            try:
                # Prepare data
                for seqs_x_t, seqs_y_t in split_shard(seqs_x, seqs_y, split_size=training_configs['update_cycle']):
                    
                    stat = nmt.train_batch(seqs_x_t, seqs_y_t, n_samples_t)
                    
                    total_stats.update(stat)
                total_stats.print_out(global_step - 1, iter, batch_iter, total_iters, lrate, n_words_t)
                optim.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    optim.zero_grad()
                else:
                    raise e
            
            if global_step % training_configs['bleu_valid_freq'] == 0:
                dev_start_time = time.time()
                dev_cer = evaluate(nmt, training_configs, data_configs)
                during_time = float(time.time() - dev_start_time)
                print("step %d, epoch %d: dev cer: %.2f, time %.2f" \
                      % (global_step, iter, dev_cer, during_time))

                if dev_cer < best_cer:
                    print("Exceed best cer: history = %.2f, current = %.2f, lr_ratio = %.6f" % \
                          (best_cer, dev_cer, lrate))
                    best_cer = dev_cer
                    checkpoint_saver.save(global_step=global_step, model=nmt.model,
                                          optim=optim, lr_scheduler=scheduler)
            


def evaluate(nmt, training_configs, data_configs):
    batch_size = training_configs['valid_batch_size']

    #valid_dataset = TextLineDataset(data_path=data_configs['valid_data'][0])

    valid_bitext_dataset = ZipDataset(
        AudioDataset(data_configs['valid_data'][0]
                     ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        max_len=data_configs['max_len'][1],
                        ),
        shuffle=training_configs['shuffle']
    )
    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                          batch_size=training_configs['valid_batch_size'],
                                          use_bucket=training_configs['use_bucket'],
                                          buffer_size=100000,
                                          numbering=True)
    max_steps = training_configs["bleu_valid_configs"]["max_steps"]
    beam_size = training_configs["bleu_valid_configs"]["beam_size"]
    alpha = training_configs["bleu_valid_configs"]["alpha"]

    nmt.model.eval()

    numbers = []
    trans = []
    truth = []
    total_cer = 0.0
    #lens_y = 0
    valid_iter = valid_iterator.build_generator(batch_size=batch_size)

    for xs in valid_iter:
        
        seq_nums = xs[0]
        numbers += seq_nums
        seqs_x = xs[1]
        seqs_y = xs[2]
        result = []
        for y_t in seqs_y:
            #lens_y +=len(y_t)
            result.append(' '.join(y_t))
        sub_trans = nmt.translate_batch(seqs_x, beam_size, max_steps, alpha)
        trans += sub_trans
        truth += result

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]
    truth = [truth[ii] for ii in origin_order]

    for i in range(0,len(trans)):
        
        pre = trans[i].split()
        tru = truth[i].split()
        total_cer += cer(tru,pre)

    total_cer =  float(total_cer/len(trans))  
    return total_cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="/home/xps/audio/Speech/Test/configs/debug.yaml",
                        help="The path to config file.")
    parser.add_argument('--saveto', type=str, default="./save",
                        help="The path for saving models. Default is ./save.")
    parser.add_argument('--model_name', type=str, default="debug",
                        help="The name of the model for saving archives.")
    parser.add_argument('--thread', default=1
                        , type=int, help='thread num')
    parser.add_argument('--gpu', default=0, type=int, help='Use id of gpu, -1 if cpu.')

    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path.strip()) as f:
        configs = yaml.load(f)

    torch.set_num_threads(args.thread)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    use_cuda = False
    if gpu and args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        print("GPU ID: ", args.gpu)
    else:
        args.gpu = -1

    print("\nGPU using status: " + str(args.gpu))

    configs = default_configs(configs)
    print(pretty_configs(configs))

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    #src_vocab = Vocabulary(**data_configs["vocabularies"][0])
    tgt_vocab = Vocabulary(**data_configs["vocabularies"][0])

    set_seed(training_configs['seed'])

    nmt_model = build_model(tgt_vocab=tgt_vocab, **model_configs)

    auto_mkdir(args.saveto)

    # GlobalNames.SEED = training_configs['seed']

    nmt = EMG(nmt_model, tgt_vocab, configs, use_cuda)

    train(nmt, configs, args)

