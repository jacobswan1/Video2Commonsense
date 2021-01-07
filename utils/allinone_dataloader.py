import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_cms_vocab_size(self):
        return len(self.get_cms_vocab())

    def get_cap_vocab_size(self):
        return len(self.get_cap_vocab())

    def get_cms_vocab(self):
        return self.cms_ix_to_word

    def get_cap_vocab(self):
        return self.cap_ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode='train'):
        super(VideoDataset, self).__init__()
        self.mode = mode

        self.captions = json.load(open(opt['caption_json']))
        cms_info = json.load(open(opt['info_json']))
        self.cms_ix_to_word = cms_info['ix_to_word']
        self.cms_word_to_ix = cms_info['word_to_ix']
        self.splits = cms_info['videos']

        # Load caption dictionary
        cap_info = json.load(open(opt['cap_info_json']))
        self.cap_ix_to_word = cap_info['ix_to_word']
        self.cap_word_to_ix = cap_info['word_to_ix']

        print('Caption vocab size is ', len(self.cap_ix_to_word))
        print('CMS vocab size is ', len(self.cms_ix_to_word))
        print('number of train videos: ', len(self.splits['train']))
        print('number of test videos: ', len(self.splits['test']))
        print('number of val videos: ', len(self.splits['val']))

        self.feats_dir = opt['feats_dir']
        print('load feats from %s' % self.feats_dir)

        self.cap_max_len = opt['cap_max_len']
        self.int_max_len = opt['int_max_len']
        self.eff_max_len = opt['eff_max_len']
        self.att_max_len = opt['att_max_len']
        print('max sequence length of caption is', self.cap_max_len)
        print('max sequence length of intention is', self.int_max_len)
        print('max sequence length of effect is', self.eff_max_len)
        print('max sequence length of attribute is', self.att_max_len)

    def __getitem__(self, ix=False):
        if not ix:
            if self.mode == 'train':
                ix = random.choice(self.splits['train'])
            elif self.mode == 'test':
                ix = self.splits['test'][ix]
        
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % ix)))
        fc_feat = np.concatenate(fc_feat, axis=1)

        total_length = self.int_max_len + self.cap_max_len + self.eff_max_len + self.att_max_len
        cap_mask = np.zeros(total_length)
        cap_gts = np.zeros(total_length)

        idx = 'video%i' % ix
        if idx not in self.captions.keys():
            raw_data = self.captions[random.choice(list(self.captions.keys()))]
        else:
            raw_data = self.captions[idx]

        cap_ix = random.randint(0, len(raw_data) - 1)   # Random pick out one caption

        caption = raw_data[cap_ix]['final_caption']
        intentions = raw_data[cap_ix]['intention']
        intention = intentions[random.randint(0, len(intentions)-1)][1]

        effects = raw_data[cap_ix]['effect']
        effect = effects[random.randint(0, len(effects)-1)][1]

        attributes = raw_data[cap_ix]['attribute']
        attribute = attributes[random.randint(0, len(attributes)-1)][1]

        allinone_caption = intention[:-1] + ['<eos>'] + caption[1:-1] + ['<eos>'] + \
                           effect[1:-1] + ['<eos>'] + attribute[1:]

        if len(allinone_caption) > total_length:
            allinone_caption = allinone_caption[:total_length]
            allinone_caption[-1] = '<eos>'

        for j, w in enumerate(allinone_caption):
            cap_gts[j] = self.cap_word_to_ix.get(w, '1')

        non_zero = (cap_gts == 0).nonzero()
        if len(non_zero[0]) != 0: cap_mask[:int(non_zero[0][0])] = 1
        else: cap_mask += 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['cap_labels'] = torch.from_numpy(cap_gts).type(torch.LongTensor)
        data['cap_masks'] = torch.from_numpy(cap_mask).type(torch.FloatTensor)

        data['video_ids'] = 'video%i' % ix
        return data

    def __len__(self):
        return len(self.splits[self.mode])
