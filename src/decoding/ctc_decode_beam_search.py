from multiprocessing import Pool
import torch
def beam_for_ctc(nmt_model,decoder,beam_size, src_seqs, alpha=-1.0):
    nmt_model.eval()
    enc_outputs = nmt_model.encode(src_seqs)
    probs = enc_outputs['encoder_log_pro']
    # decoded = probs.transpose(0,1)
    
    decoded = probs.cpu().numpy()
    text_list = []
    for i in range(decoded.shape[0]):
        text = decoder.decode(decoded[i],beam_size)
        text_list.append(text)
    return text_list

        