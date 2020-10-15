"""
This was heavily borrowed from 
https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from beam import Beam

def beam_decode(model, src, hand_regions, rel_mask, src_mask, max_len=30, start_symbol=1, n_beam=3):

    if(type(rel_mask) != type(None)):
        src_mask = rel_mask

    memory = model.src_emb(src)
    memory = model.position(memory)
    memory = model.encode(memory, None, src_mask)

    ys = []
    scores = []

    n_hyp = 1

    for b in range(src.shape[0]):
        src_seq = src[b].unsqueeze(0)
        src_enc = memory[b].unsqueeze(0)
        if(src_mask is None):
            mask = None
        else:
            mask = src_mask[b].unsqueeze(0)

        #NOTE: may produce eos tokens
        batch_hyp, batch_scores = translate_batch(model, src_seq, src_enc, mask, n_beam, n_hyp, max_len)

        pred = []
        #Remove EOS token
        for elem in batch_hyp[0][0]:
            pred.append(elem)
            if(elem == 2):
                break

        
        #NOTE:this is because we have just one hypotheses
        ys.append(pred)
        scores.append(batch_scores)

    ys = [torch.IntTensor(y) for y in ys]
    
    return ys

def translate_batch(model, src_seq, src_enc, src_mask, beam_size = 2, n_best=1, max_len=20, device='cuda'):
    ''' Translation work in one batch '''

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)

        beamed_tensor = beamed_tensor.view(*new_shape)
        return beamed_tensor

    def collate_active_info(src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        #active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        #return active_src_seq, active_src_enc, active_inst_idx_to_position_map
        return active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(
        inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_mask(len_dec_seq, n_active_inst, n_bm):
            #dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
            #dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            attn_shape = (n_active_inst*n_bm, len_dec_seq, len_dec_seq)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            dec_partial_pos = torch.from_numpy(subsequent_mask) == 0
            dec_partial_pos = dec_partial_pos.type(torch.uint8).to('cuda')

            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_enc, n_active_inst, n_bm):

            #dec_mask = dec_mask.unsqueeze(1)
            dec_output = model.decode(src_enc, dec_seq, src_mask, dec_pos)

            #(beam, vocab)
            dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
            #print(dec_output.shape)
            word_prob = model.output_layer(dec_output)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)
            #print(word_prob.shape)
            #sd
            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []

            for inst_idx, inst_position in inst_idx_to_position_map.items():

                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_mask(len_dec_seq, n_active_inst, n_bm)

        word_prob = predict_word(dec_seq, dec_pos, src_enc, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):

            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        #-- Encode
        #src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
        #memory = model.module.encode(src_seq, regions, src_mask)

        #-- Repeat data for beam search
        n_bm = beam_size
        n_inst, len_s, d_h = src_enc.size()

        #src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s, -1)
        src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
        #print(src_enc.shape)
        if(src_mask is not None):
            src_mask = src_mask.repeat(1, n_bm, 1).view(n_inst * n_bm, 1, len_s)

        #-- Prepare beams
        inst_dec_beams = [Beam(n_bm, device=device) for _ in range(n_inst)]

        #-- Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        #-- Decode
        for len_dec_seq in range(1, max_len + 1):

            #print(len_dec_seq)

            active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

            if not active_inst_idx_list:
                #print("BREAK")
                break  # all instances have finished their path to <EOS>

            src_enc, inst_idx_to_position_map = collate_active_info(
                    src_enc, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

    return batch_hyp, batch_scores
