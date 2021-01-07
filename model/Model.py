''' Define the Transformer model '''
from utils.utils import *
from model.Decoder import Decoder
from model.EncoderRNN import EncoderRNN

__author__ = 'Jacob Zhiyuan Fang'


class Model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_cap_vocab, n_cms_vocab, cap_max_seq, cms_max_seq, vis_emb=2048,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, rnn_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True):

        super().__init__()

        # set RNN layers at 1 or 2 yield better performance.
        self.encoder = EncoderRNN(vis_emb, d_model, n_layers=rnn_layers, bidirectional=0)

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

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        cms_seq, cms_pos = cms_seq[:, :-1], cms_pos[:, :-1]

        enc_output, *_ = self.encoder(vis_feat)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, vis_feat, enc_output)
        seq_logit = self.cap_word_prj(dec_output) * self.x_logit_scale

        # Concatenate visual and caption encoding
        cat_output = torch.cat((enc_output, dec_output), 1)

        cms_dec_output, *_ = self.cms_decoder(cms_seq, cms_pos, cat_output, cat_output)
        cms_logit = self.cms_word_prj(cms_dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2)), cms_logit.view(-1, cms_logit.size(2))

