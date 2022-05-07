# -*- coding: utf-8 -*-
# from src.emg_helper import EMGHelper
from ast import arg
from src.optim import Optimizer
from src.optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler
from src.utils.configs import default_configs, pretty_configs
from src.utils import auto_mkdir
from src.data.vocabulary import Vocabulary
from src.data.vocabulary_transducer import Vocabulary_T
from src.models import build_model
from src.utils.common_utils import *
from src.emg_asr_helper import *
# from src.emg_rnnt_helper import *
# import kenlm
import argparse
import yaml
import os
import random
import ntpath
import numpy as np
from pyctcdecode import build_ctcdecoder

def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True




def test(nmt, configs, data_configs,beamdecoder):

    data_configs = configs['data_configs']
    training_configs = configs['training_configs']


    print('Building Optimizer...')

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(configs['saveto'], configs['model_name'])),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    # best_model_prefix = os.path.join(args.saveto, args.model_name + ".best")
    # best_model_saver = Saver(save_prefix=best_model_prefix, num_max_keeping=training_configs['num_kept_best_model'])

    print('start test...')
    checkpoint_saver.load_latest(model=nmt.model)

    batch_size = training_configs['valid_batch_size']

    #valid_dataset = TextLineDataset(data_path=data_configs['valid_data'][0])
    point_detection_state=data_configs['point_detection'] if data_configs['ssfilter'] else False

    valid_bitext_dataset = ZipDataset(
        EMGDataset(data_configs['test_data'][0],nfft=data_configs['nfft'],
                        hop_length=data_configs['hop_length'],norm=data_configs['norm'],ss=data_configs['ssfilter'],
                        point_detection=point_detection_state
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
            
        sub_trans = nmt.ctc_beamsearch_batch(seqs_x,beamdecoder, beam_size, max_steps, alpha)
        # sub_trans = nmt.ctc_decode_batch(seqs_x, beam_size, max_steps, alpha)

        # sub_trans = nmt.translate_batch(seqs_x, beam_size, max_steps, alpha)

        trans += sub_trans
        truth += result

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]
    truth = [truth[ii] for ii in origin_order]
    import re
    for i in range(0,len(trans)):
        # line = re.sub(" ","",trans[i])
       
        # line = re.sub(r'([\u4e00-\u9fa5])\1+', r'\1',line)
        print(trans[i])

        # print("predi: "+ trans[i])
        # print("truth: "+ truth[i])
        # pre = list(line)
        pre = trans[i].split()
        tru = truth[i].split()
        total_cer += cer(tru,pre)

    total_cer = float(total_cer/len(trans))
    
    return total_cer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="save/expand_sentences/transformer_joint_ctc/att_decay_smooth0_dropout0_greedy_815data_kl300/config-dev.yml",
                        help="The path to config file.")
    parser.add_argument('--thread', default=1
                        , type=int, help='thread num')
    parser.add_argument('--gpu', default=4, type=int, help='Use id of gpu, -1 if cpu.')

    parser.add_argument('--evaluate',default=True, type=bool, help='evaluate the trained model')

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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(torch.cuda.current_device())
        print("GPU ID: ", args.gpu)

    else:
        args.gpu = -1

    print("\nGPU using status: " + str(args.gpu))
    
    configs.update(vars(args))

    configs = default_configs(configs)
    print(pretty_configs(configs))
    
    
    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    #src_vocab = Vocabulary(**data_configs["vocabularies"][0])
    if model_configs['joint_ctc']:
        tgt_vocab = Vocabulary_T(**data_configs["vocabularies"][0])
    else:
        tgt_vocab = Vocabulary(**data_configs["vocabularies"][0])

    vocab_list = []
    for i in range(len(tgt_vocab._id2token)):
        vocab_list.append(tgt_vocab._id2token[i])
    vocab_list[4] = '<BLANK>'
    

    kenlm_model = None

    

    decoder = build_ctcdecoder(
                vocab_list,
                kenlm_model,
                alpha=0.5,  # tuned on a val set
                beta=1.0,  # tuned on a val set
            )
    set_seed(training_configs['seed'])

    nmt_model = build_model(tgt_vocab=tgt_vocab, **model_configs)

    #auto_mkdir(args.saveto)

    # GlobalNames.SEED = training_configs['seed']

    nmt = EMGASRHelper(nmt_model, tgt_vocab, configs, use_cuda)
    # nmt = EMGRNNTHelper(nmt_model, tgt_vocab, configs, use_cuda)


    print('start evaluate...')
    cer = test(nmt, configs, args,decoder)
    print("the cer", cer)
    out_info = ("cer:  %.2f") % (cer)
    print(out_info)
    

