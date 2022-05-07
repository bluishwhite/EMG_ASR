from src.data.data_iterator import DataIterator
from src.data.dataset import EMGDataset, TextLineDataset, ZipDataset,EMGNPYDataset
from src.decoding import beam_search,beam_search_ctc,beam_for_ctc
import torch
import numpy as np
from src.modules.criterions import JointCtcDekayAttention,NMTCriterion
import torchaudio as ta
import time
import math
import sys
# import torch.nn as nn

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def kl_anneal_function(step, k=0.005, x0=1000): #!x0 =650 =950 =1250  =300
    return float(1 / (1 + np.exp(-k * (step - x0))))


def compute_torch_fbank(file):
    wavform, _ = ta.load_wav(file)  # 加载wav文件
    feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=80)  # 计算fbank特征
    # 特征归一化
    mean = torch.mean(feature)
    std = torch.std(feature)
    feature = (feature - mean) / std
    return feature


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, vae_loss=0, c_loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.vae_loss = vae_loss
        self.c_loss = c_loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.vae_loss += stat.vae_loss
        self.c_loss += stat.c_loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return safe_exp(self.loss / self.n_words)

    def vae_ppl(self):
        return safe_exp(self.vae_loss / self.n_words)

    def without_vae_ppl(self):
        return safe_exp(self.c_loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, step, epoch, batch, n_batches, lr, batch_size):
        t = self.elapsed_time()
        out_info = ("Step %d, Epoch %d, %d/%d| lr: %.6f| words: %d| "
                    "acc: %.2f| ppl: %.2f| loss %.2f| %.1f tgt tok/s| %.2f s elapsed") \
                   % (step, epoch, batch, n_batches, lr, int(batch_size), self.accuracy(), self.ppl(), \
                    self.loss, self.n_words / (t + 1e-5), time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()

    def print_valid(self, step):
        t = self.elapsed_time()
        out_info = ("Valid at step %d: acc %.2f, ppl: %.2f,loss %.2f, %.1f tgt tok/s, %.2f s elapsed") % \
              (step, self.accuracy(), self.ppl(), self.loss, self.n_words / (t + 1e-5),
               time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()


class EMGASRDecayHelper(object):
    def __init__(self, model, tgt_vocab, configs, use_cuda):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.tgt_pad = tgt_vocab.pad()
        self.tgt_eos = tgt_vocab.eos()
        self.tgt_bos = tgt_vocab.bos()
        data_configs = configs['data_configs']
        model_configs = configs['model_configs']
        training_configs = configs['training_configs']
        self.joint_ctc = model_configs['joint_ctc']
        if self.joint_ctc:
            self.critic = JointCtcDekayAttention(blank_id=tgt_vocab.blank_id(),padding_idx=self.tgt_pad)
        else:
            self.critic = NMTCriterion(padding_idx=self.tgt_pad,
                                    label_smoothing=model_configs['label_smoothing'])

        # self.critic = nn.CrossEntropyLoss(ignore_index=self.tgt_vocab.blank_id)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            self.critic = self.critic.cuda()
            
        p = next(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.device = p.get_device() if self.use_cuda else None
        point_detection_state=data_configs['point_detection'] if data_configs['ssfilter'] else False


        train_bitext_dataset = ZipDataset(
            EMGNPYDataset(data_path=data_configs['train_data'][0],nfft=data_configs['nfft'],
                        hop_length=data_configs['hop_length'],norm=data_configs['norm'],
                        ss=data_configs['ssfilter'],
                        point_detection=point_detection_state
                            ),
            TextLineDataset(data_path=data_configs['train_data'][1],
                            max_len=data_configs['max_len'][1],
                            ),
            shuffle=training_configs['shuffle']
        )

        train_batch_size = training_configs["batch_size"] * max(1, training_configs["update_cycle"])
        train_buffer_size = training_configs["buffer_size"] * max(1, training_configs["update_cycle"])

        self.training_iterator = DataIterator(dataset=train_bitext_dataset,
                                              batch_size=train_batch_size,
                                              use_bucket=training_configs['use_bucket'],
                                              buffer_size=train_buffer_size,
                                              batching_func=training_configs['batching_key'])

    def tgt_data_id(self, tgt_input):
        result = [self.tgt_vocab.token2id(cur_word) for cur_word in tgt_input]
        return [self.tgt_bos] + result + [self.tgt_eos]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            eval_data.append((self.src_data_id(src_input), src_input))

        return eval_data

    def pair_data_variable(self, features, seqs_y_t):
        batch_size = len(seqs_y_t)

        features_length = [feature.shape[0] for feature in features]
        ctc_targets = []
        for seq_y_t in seqs_y_t:
            ctc_targets.extend(seq_y_t[1:-1])
        padded_features = []
        #src_lengths = [len(seqs_x_t[i]) for i in range(batch_size)]
        max_feature_length = int(np.max(features_length))

        tgt_lengths = [len(seqs_y_t[i]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        for feature in features:
            feat_len = feature.shape[0]
            padded_features.append(np.pad(feature, ((0, max_feature_length - feat_len), (0, 0)), mode='constant',
                                          constant_values=0.0))
        tgt_words = torch.zeros([batch_size, max_tgt_length], dtype=torch.int64, requires_grad=False)
        tgt_words = tgt_words.fill_(self.tgt_pad)

        for b in range(batch_size):
            for index, word in enumerate(seqs_y_t[b]):
                tgt_words[b, index] = word
        ctc_tgt_lengths = [len(seqs_y_t[i])-2 for i in range(batch_size)]
        padded_features = torch.FloatTensor(padded_features)
        ctc_tgt_lengths = torch.IntTensor(ctc_tgt_lengths)
        ctc_targets = torch.IntTensor(ctc_targets)
        
        if self.use_cuda:
            padded_features = padded_features.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)
            ctc_tgt_lengths = ctc_tgt_lengths.cuda(self.device)
            ctc_targets = ctc_targets.cuda(self.device)


        return padded_features, tgt_words, ctc_tgt_lengths,ctc_targets

    def source_data_variable(self, features):
        batch_size = len(features)

        features_length = [feature.shape[0] for feature in features]
        padded_features = []
        # src_lengths = [len(seqs_x_t[i]) for i in range(batch_size)]
        max_feature_length = int(np.max(features_length))

        for feature in features:
            feat_len = feature.shape[0]
            padded_features.append(np.pad(feature, ((0, max_feature_length - feat_len), (0, 0)), mode='constant',
                                          constant_values=0.0))

        padded_features = torch.FloatTensor(padded_features)

        if self.use_cuda:
            padded_features = padded_features.cuda(self.device)

        return padded_features

    def compute_forward(self, seqs_x, seqs_y,ctc_y, ctc_lengths, norm,step):
        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()
        

        self.model.train()
        self.critic.train()
        # For training
        att_weight = kl_anneal_function(step)
        with torch.enable_grad():
            
            log_probs, encoder_log_pro, encoder_output_length = self.model(seqs_x, y_inp)
            # log_probs, mu, log_var = self.model(seqs_x, y_inp)
            # KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            # KLD = torch.sum(KLD_element).mul_(-0.5)

            # c_loss = self.critic(inputs=log_probs, labels=y_label, reduce=False, normalization=norm)

            # loss = c_loss + kl_anneal_function(step) * KLD
            # loss = loss.sum()
            # KLD_loss = KLD.sum()
            # c_loss = c_loss.sum()
            if self.joint_ctc:
                loss = self.critic(inputs=log_probs, labels=y_label, encoder_output=encoder_log_pro.transpose(1,0), ctc_targets=ctc_y, input_lengths=encoder_output_length, target_lengths=ctc_lengths, weight=att_weight, reduce=False, normalization=norm)
            else:
                loss = self.critic(inputs=log_probs, labels=y_label, reduce=False, normalization=norm)
            loss = loss.sum()
        torch.autograd.backward(loss)

        mask = y_label.detach().ne(self.tgt_pad)
        pred = log_probs.detach().max(2)[1]  # [batch_size, seq_len]

        num_correct = y_label.detach().eq(pred).float().masked_select(mask).sum()
        num_total = mask.sum().float()

        lossvalue = norm * loss.item()
        
        stats = Statistics(lossvalue, 0, 0, num_total, num_correct)

        return stats

    def train_batch(self, seqs_x_t, seqs_y_t, norm,step):
        self.model.train()

        feature = [x_t for x_t in seqs_x_t ]
        seqs_y = [self.tgt_data_id(y_t) for y_t in seqs_y_t]
        src_words, tgt_words,tgt_lengths,ctc_targets = self.pair_data_variable(feature, seqs_y)
        
        stat = self.compute_forward(src_words, tgt_words,ctc_targets,tgt_lengths, norm,step)

        return stat

    def translate_batch(self, seqs_x_t, beam_size, max_steps, len_pen):
        x_features = [x_t for x_t in seqs_x_t]
        pad_features = self.source_data_variable(x_features)
        with torch.no_grad():
            word_ids = beam_search(nmt_model=self.model,
                                   beam_size=beam_size,
                                   max_steps=max_steps,
                                   src_seqs=pad_features,
                                   alpha=len_pen)

        word_ids = word_ids.cpu().numpy().tolist()

        # Append result
        trans = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != self.tgt_pad] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == self.tgt_eos:
                    break
                x_tokens.append(self.tgt_vocab.id2token(wid))
            if len(x_tokens) > 0:
                trans.append(self.tgt_vocab.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % self.tgt_vocab.id2token(self.tgt_eos))

        return trans
    
    def ctc_decode_batch(self, seqs_x_t, beam_size, max_steps, len_pen):
        x_features = [x_t for x_t in seqs_x_t]
        pad_features = self.source_data_variable(x_features)
        with torch.no_grad():
            # word_ids = beam_search_ctc(nmt_model=self.model,
            #                        beam_size=beam_size,
            #                        src_seqs=pad_features,
            #                        alpha=len_pen)
            y_hats =beam_search_ctc(nmt_model=self.model,
                                   beam_size=beam_size,
                                   src_seqs=pad_features,
                                   alpha=len_pen)

        word_ids = y_hats.cpu().numpy().tolist()

        # Append result
        trans = []
        blank_id = self.tgt_vocab.blank_id()
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != self.tgt_pad and wid != blank_id] for line in sent_t]
            x_tokens = []

            for wid in sent_t[0]:
                if wid == self.tgt_eos:
                    break
                x_tokens.append(self.tgt_vocab.id2token(wid))
            if len(x_tokens) > 0:
                trans.append(self.tgt_vocab.tokenizer.detokenize(x_tokens))
            else:
                trans.append('%s' % self.tgt_vocab.id2token(self.tgt_eos))

        return trans
    
    def ctc_beamsearch_batch(self, seqs_x_t,beamdecoder, beam_size, max_steps, len_pen):
        x_features = [x_t for x_t in seqs_x_t]
        pad_features = self.source_data_variable(x_features)
        with torch.no_grad():
            # word_ids = beam_search_ctc(nmt_model=self.model,
            #                        beam_size=beam_size,
            #                        src_seqs=pad_features,
            #                        alpha=len_pen)
            word_ids =beam_for_ctc(nmt_model=self.model,decoder=beamdecoder, 
                                   beam_size=beam_size,
                                   src_seqs=pad_features,
                                   alpha=len_pen)
    
        trans = []

        for sent_t in word_ids:
            sent_t = [wd for wd in sent_t]
            text = ' '.join(sent_t)
            
            trans.append(text)
            # for wid in sent_t[0]:
            #     if wid == self.tgt_eos:
            #         break
            #     x_tokens.append(self.tgt_vocab.id2token(wid))
            # if len(x_tokens) > 0:
            #     trans.append(self.tgt_vocab.tokenizer.detokenize(x_tokens))
            # else:
            #     trans.append('%s' % self.tgt_vocab.id2token(self.tgt_eos))

        return trans