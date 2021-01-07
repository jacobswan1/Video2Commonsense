''' This module will handle the text generation with beam search. '''
import numpy as np
from utils.utils import *
import torch.nn.functional as F
from model.transformer.Beam import Beam

__author__ = 'Jacob Zhiyuan Fang'


def pos_emb_generation(word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return tgt_pos*binary_mask


def translate_batch(model, src_emb, cap_label, opt):
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

    def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).cuda()

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm, mode):
        ''' Decode and update beam status, and then return active beam idx '''

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).cuda()
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long).cuda()
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
            if mode == 'cap':
                dec_output, *_ = model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cap_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

            elif mode == 'int':
                dec_output, *_ = model.cms_decoder(dec_seq, dec_pos, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(model.cms_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
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
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

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
        # <--------------------------------------------- Decode Video ------------------------------------------------->
        src_seq = src_emb.cuda()
        src_enc, *_ = model.encoder(src_seq)
        video_encoding = src_enc

        # <--------------------------------------------- Decode CAP --------------------------------------------------->
        cap_pos = pos_emb_generation(cap_label)
        cap_label, cap_pos = cap_label[:, :-1], cap_pos[:, :-1]

        cap_dec_output, *_ = model.decoder(cap_label, cap_pos, video_encoding, video_encoding)

        # Concatenate visual and caption encoding
        cat_encoding = torch.cat((video_encoding, cap_dec_output), 1)
        # cat_encoding = cap_dec_output

        # <--------------------------------------------- Decode CMS --------------------------------------------------->
        # Repeat data for beam search for CMS
        n_bm = 2
        n_inst, len_s, d_h = cat_encoding.size()
        src_enc = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
        src_seq = cat_encoding.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, -1)

        # Prepare beams
        inst_dec_beams = [Beam(n_bm, device='cuda') for _ in range(n_inst)]

        # Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, opt['eff_max_len'] - 1):

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm, mode='int')

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        cms_batch_hyp, cms_batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

    return cms_batch_hyp



