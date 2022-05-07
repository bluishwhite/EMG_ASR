# -*- coding: utf-8 -*-
# from src.emg_helper import EMGHelper
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.configs import default_configs, pretty_configs
from src.utils import auto_mkdir
from src.data.vocabulary import Vocabulary
from src.data.vocabulary_transducer import Vocabulary_T
from src.models import build_model
from src.utils.common_utils import *
from src.emg_ctc_voice_npy_mix_helper import *
from pyctcdecode import build_ctcdecoder

import argparse
import yaml
import os
import random
import ntpath
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def train(nmt, configs, args,decoder):
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
    checkpoint_saver.load_latest(model=nmt.model, optim=optim, lr_scheduler=scheduler)
    best_cer = 100

    global_step = 0

    for iter in range(502):
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
                    
                    stat = nmt.train_batch(seqs_x_t, seqs_y_t, n_samples_t,global_step)
                    
                    total_stats.update(stat)
                total_stats.print_out(global_step - 1, iter, batch_iter, total_iters, lrate, n_words_t)
                optim.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    optim.zero_grad()
                else:
                    raise e
            
            if global_step % training_configs['loss_valid_freq'] == 0:
                dev_start_time = time.time()
                dev_cer = evaluate(nmt, training_configs, data_configs,decoder)
                during_time = float(time.time() - dev_start_time)
                print("step %d, epoch %d: dev cer: %.5f, time %.2f" \
                      % (global_step, iter, dev_cer, during_time))

                if dev_cer < best_cer:
                    print("Exceed best cer: history = %.5f, current = %.5f, lr_ratio = %.6f" % \
                          (best_cer, dev_cer, lrate))
                    best_cer = dev_cer
                    checkpoint_saver.save(global_step=global_step, model=nmt.model,
                                          optim=optim, lr_scheduler=scheduler)
        if iter % training_configs['epcoh_valid_freq'] == 0 and iter!= 0: 
            dev_start_time = time.time()
            dev_cer = evaluate(nmt, training_configs, data_configs,decoder)
            during_time = float(time.time() - dev_start_time)
            print("step %d, epoch %d: dev cer: %.5f, time %.2f" \
                    % (global_step, iter, dev_cer, during_time))
        
            if dev_cer < best_cer:
                print("Exceed best cer: history = %.5f, current = %.5f, lr_ratio = %.6f" % \
                        (best_cer, dev_cer, lrate))
                best_cer = dev_cer
                checkpoint_saver.save(global_step=global_step, model=nmt.model,
                                        optim=optim, lr_scheduler=scheduler)
        

def evaluate(nmt, training_configs, data_configs,beam_decoder):
    batch_size = training_configs['valid_batch_size']

    #valid_dataset = TextLineDataset(data_path=data_configs['valid_data'][0])

    valid_bitext_dataset = ZipDataset(
        EMGDataset(data_configs['valid_data'][0],nfft=data_configs['nfft'],
                        hop_length=data_configs['hop_length'],norm=data_configs['norm'],
                        ss=data_configs['ssfilter']
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

        sub_trans = nmt.ctc_beamsearch_batch(seqs_x,beam_decoder, beam_size, max_steps, alpha)

        trans += sub_trans
        truth += result

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]
    truth = [truth[ii] for ii in origin_order]

    for i in range(0,len(trans)):
        if i% 50 == 0:
            print("predict: "+ trans[i])
            print("truth: "+ truth[i])
        pre = trans[i].split()
        tru = truth[i].split()
        total_cer += cer(tru,pre)

    total_cer = float(total_cer/len(trans))
    
    return total_cer



def test(nmt, configs, args, beam_decoder):

    data_configs = configs['data_configs']
    training_configs = configs['training_configs']

    global_step = 0

    print('Building Optimizer...')

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.saveto, args.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    # best_model_prefix = os.path.join(args.saveto, args.model_name + ".best")
    # best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    print('start testing...')
    checkpoint_saver.load_latest(model=nmt.model)

    batch_size = training_configs['valid_batch_size']


    valid_bitext_dataset = ZipDataset(
        EMGDataset(data_configs['test_data'][0],nfft=data_configs['nfft'],
                        hop_length=data_configs['hop_length'],norm=data_configs['norm'],
                        ss=data_configs['ssfilter']
                     ),
        TextLineDataset(data_path=data_configs['test_data'][1],
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
        sub_trans = nmt.ctc_beamsearch_batch(seqs_x,beam_decoder, beam_size, max_steps, alpha)

        trans += sub_trans
        truth += result

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]
    truth = [truth[ii] for ii in origin_order]

    for i in range(0,len(trans)):
        if i%500:
            print("predict: "+ trans[i])
            print("truth: "+ truth[i])
        pre = trans[i].split()
        tru = truth[i].split()
        total_cer += cer(tru,pre)

    total_cer = float(total_cer/len(trans))
    
    return total_cer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="",
                        help="The path to config file.")
    parser.add_argument('--saveto', type=str, default="./save",
                        help="The path for saving models. Default is ./save.")
    parser.add_argument('--model_name', type=str, default="",
                        help="The name of the model for saving archives.")
    parser.add_argument('--thread', default=1
                        , type=int, help='thread num')
    parser.add_argument('--gpu', default=2, type=int, help='Use id of gpu, -1 if cpu.')

    parser.add_argument('--evaluate',default=False, type=bool, help='evaluate the trained model')

    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    torch.set_num_threads(args.thread)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    use_cuda = False
    if gpu and args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        print(torch.cuda.current_device())
        print("GPU ID: ", args.gpu)
    else:
        args.gpu = -1

    print("\nGPU using status: " + str(args.gpu))

    configs = default_configs(configs)
    print(pretty_configs(configs))

    
    configs.update(vars(args))
    with open(os.path.join(args.saveto, "config.yml"), "w") as f:
        yaml.dump(configs, f, Dumper=yaml.Dumper)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    #src_vocab = Vocabulary(**data_configs["vocabularies"][0])

    tgt_vocab = Vocabulary_T(**data_configs["vocabularies"][0])


    vocab_list = []
    for i in range(len(tgt_vocab._id2token)):
        vocab_list.append(tgt_vocab._id2token[i])
    vocab_list[4] = '<BLANK>'
    
    kenlm_model = None

    ctc_beam_decoder = build_ctcdecoder(
                vocab_list,
                kenlm_model,
                alpha=0.5,  # tuned on a val set
                beta=1.0,  # tuned on a val set
            )    


    set_seed(training_configs['seed'])

    nmt_model = build_model(tgt_vocab=tgt_vocab, **model_configs)

    #auto_mkdir(args.saveto)

    # GlobalNames.SEED = training_configs['seed']

    nmt = EMGVoiceNPYMIXCTCHelper(nmt_model, tgt_vocab, configs, use_cuda)

    if args.evaluate:

        print('start evaluate...')
        cer = test(nmt, configs, args, ctc_beam_decoder)
        print("the cer", cer)
        out_info = ("cer:  %.5f") % (cer)
        print(out_info)
    else:
        train(nmt, configs, args,ctc_beam_decoder)

