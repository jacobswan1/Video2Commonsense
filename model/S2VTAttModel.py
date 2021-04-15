import torch.nn as nn
import torch

class S2VTAttModel(nn.Module):
    def __init__(self, encoder, decoder, cms_decoder):
        """

        Args:
            encoder (nn.Module): Encoder rnn
            decoder (nn.Module): Decoder rnn
        """
        super(S2VTAttModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cms_decoder = cms_decoder

    def forward(self, vid_feats, cap_labels=None, cms_labels=None, mode='train', opt={}):
        """

        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): ground truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        encoder_outputs, encoder_hidden = self.encoder(vid_feats)
        # seq_prob, _, cap_encoding, cap_hidden = self.decoder(encoder_outputs,encoder_hidden, cap_labels, 'train', opt)
        _, seq_prob, cap_encoding, cap_hidden = self.decoder(encoder_outputs, encoder_hidden,
                                                          None, 'inference', opt)

        cat_encoding = torch.cat((encoder_outputs, cap_encoding), 1)
        if mode == 'test':
            _, cms_seq_prob, _, _ = self.cms_decoder(cat_encoding, cap_hidden, targets=None, mode='inference', opt=opt)
        else:
            cms_seq_prob, _, _, _ = self.cms_decoder(cat_encoding, cap_hidden, cms_labels, mode='train', opt=opt)
        return seq_prob, cms_seq_prob
