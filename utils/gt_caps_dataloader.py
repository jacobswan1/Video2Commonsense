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
        self.cap_max_len = opt['cap_max_len']

        print('load feats from %s' % self.feats_dir)
        print('max sequence length of caption is', self.cap_max_len)

    def __getitem__(self, ix):

        if self.mode == 'train':
            ix = random.choice(self.splits['train'])
        elif self.mode == 'test':
            ix = self.splits['test'][ix]
        
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % ix)))
        fc_feat = np.concatenate(fc_feat, axis=1)

        raw_data = self.captions['video%i' % ix]
        num_cap = len(raw_data)
        cap_mask = np.zeros((num_cap, self.cap_max_len))
        cap_gts = np.zeros((num_cap, self.cap_max_len))
        int_list, eff_list, att_list = [], [], []

        # Load all num_cap gt captions
        for cap_ix in range(num_cap):
            caption = raw_data[cap_ix % len(raw_data)]['final_caption']

            if len(caption) > self.cap_max_len:
                caption = caption[:self.cap_max_len]
                caption[-1] = '<eos>'

            for j, w in enumerate(caption[0: self.cap_max_len]):
                cap_gts[cap_ix, j] = self.cap_word_to_ix.get(w, '1')

            intentions, effects, attributes =  raw_data[cap_ix]['intention'], raw_data[cap_ix]['effect'],\
                                               raw_data[cap_ix]['attribute']

            # Concatenate all CMS
            int_str, att_str, eff_str = '', '', ''
            for int, eff, att in zip(intentions, effects, attributes):
                int_str += ';' + int[0]
                eff_str += ';' + eff[0]
                att_str += ';' + att[0]

            int_list.append(int_str)
            eff_list.append(eff_str)
            att_list.append(att_str)

        # Insert mask
        cap_mask[(cap_gts != 0)] = 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['cap_labels'] = torch.from_numpy(cap_gts).type(torch.LongTensor)
        data['cap_masks'] = torch.from_numpy(cap_mask).type(torch.FloatTensor)
        data['video_ids'] = 'video%i' % ix

        return data, int_list, eff_list, att_list

    def __len__(self):
        return len(self.splits[self.mode])

