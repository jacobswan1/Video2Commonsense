''' Define the Transformer model '''
import numpy as np
from utils.utils import *
from model.Decoder import Decoder
from model.transformer.Layers import EncoderLayer

__author__ = 'Yu-Hsiang Huang'
__AugmentedBy__ = 'Jacob Zhiyuan Fang'


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1).cuda()


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask.cuda()

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_word_vec, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        _ = torch.rand(src_emb.shape[0], src_emb.shape[1])
        slf_attn_mask = get_attn_key_pad_mask(seq_k=_, seq_q=_)
        non_pad_mask = get_non_pad_mask(_)

        # -- Forward
        enc_output = src_emb + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask,
                                                 slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_cap_vocab, n_cms_vocab, cap_max_seq, cms_max_seq, vis_emb=2048,
            d_word_vec=512, d_model=512, d_inner=2048, n_layers=6, rnn_layers=1,
            n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True):

        super().__init__()

        # set RNN layers at 1 or 2 yield better performance.
        self.vis_emb = nn.Linear(vis_emb, d_model)
        self.encoder = Encoder(40, d_model, rnn_layers, n_head, d_k, d_v,
                               d_model, d_inner, dropout=0.1)

        self.decoder = Decoder(
            n_tgt_vocab=n_cap_vocab, len_max_seq=cap_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cms_decoder = Decoder(
            n_tgt_vocab=n_cms_vocab, len_max_seq=cms_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.cap_word_prj = nn.Linear(d_model, n_cap_vocab, bias=False)
        self.cms_word_prj = nn.Linear(d_model, n_cms_vocab, bias=False)

        nn.init.xavier_normal_(self.cap_word_prj.weight)
        nn.init.xavier_normal_(self.cms_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, ' \
            'the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.cap_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.cms_word_prj.weight = self.cms_decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, vis_feat, tgt_seq, tgt_pos, cms_seq, cms_pos):
        vis_feat = self.vis_emb(vis_feat)
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        cms_seq, cms_pos = cms_seq[:, :-1], cms_pos[:, :-1]

        vis_pos = torch.tensor(list(range(0, 40))).cuda().unsqueeze(0).repeat(vis_feat.shape[0], 1)
        enc_output, *_ = self.encoder(vis_feat, vis_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, vis_feat, enc_output)
        seq_logit = self.cap_word_prj(dec_output) * self.x_logit_scale

        # Concatenate visual and caption encoding
        cat_output = torch.cat((enc_output, dec_output), 1)

        cms_dec_output, *_ = self.cms_decoder(cms_seq, cms_pos, cat_output, cat_output)
        cms_logit = self.cms_word_prj(cms_dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2)), cms_logit.view(-1, cms_logit.size(2))

