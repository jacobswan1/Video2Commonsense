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

    def __getitem__(self, ix):

        if self.mode == 'train':
            ix = random.choice(self.splits['train'])
        elif self.mode == 'test':
            ix = self.splits['test'][ix]
        
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % ix)))
        fc_feat = np.concatenate(fc_feat, axis=1)

        cap_mask = np.zeros(self.cap_max_len)
        int_mask = np.zeros(self.int_max_len)
        eff_mask = np.zeros(self.eff_max_len)
        att_mask = np.zeros(self.att_max_len)

        # cap_gts = np.zeros((10, self.cap_max_len))
        cap_gts = np.zeros((1, self.cap_max_len))
        int_gts = np.zeros(self.int_max_len)
        eff_gts = np.zeros(self.eff_max_len)
        att_gts = np.zeros(self.att_max_len)

        raw_data = self.captions['video%i' % ix]

        cap_ix = random.randint(0, len(raw_data) - 1)   # Random pick out one caption

        caption = raw_data[cap_ix]['final_caption']

        intentions = raw_data[cap_ix]['intention']
        intention = intentions[random.randint(0, len(intentions)-1)][1]

        effects = raw_data[cap_ix]['effect']
        effect = effects[random.randint(0, len(effects)-1)][1]

        attributes = raw_data[cap_ix]['attribute']
        attribute = attributes[random.randint(0, len(attributes)-1)][1]

        # Load all intentions again for eval
        # intentions = [item['intention'][0] for item in raw_data]
        # effects = [item['effect'][0] for item in raw_data]
        # attributes = [item['attribute'][0] for item in raw_data]

        if len(caption) > self.cap_max_len:
            caption = caption[:self.cap_max_len]
            caption[-1] = '<eos>'
        if len(effect) > self.eff_max_len:
            effect = effect[:self.eff_max_len]
            effect[-1] = '<eos>'
        if len(attribute) > self.att_max_len:
            attribute = attribute[:self.att_max_len]
            attribute[-1] = '<eos>'

        # Load all 10 gt captions
        # for i in range(10):
        #     _ = len(raw_data)
        #     caption = raw_data[i%_]['final_caption']
        #     for j, w in enumerate(caption[0:28]):
        #         cap_gts[i, j] = self.cap_word_to_ix.get(w, '1')

        # Load one random gt captions
        for j, w in enumerate(caption[0:28]):
            cap_gts[0, j] = self.cap_word_to_ix.get(w, '1')

        for j, w in enumerate(intention):
            int_gts[j] = self.cms_word_to_ix.get(w, '1')
        for j, w in enumerate(effect):
            eff_gts[j] = self.cms_word_to_ix.get(w, '1')
        for j, w in enumerate(attribute):
            att_gts[j] = self.cms_word_to_ix.get(w, '1')

        non_zero = (cap_gts == 0).nonzero()
        if len(non_zero[0]) != 0: cap_mask[:int(non_zero[0][0])] = 1
        else: cap_mask += 1

        non_zero = (int_gts == 0).nonzero()
        if len(non_zero[0]) != 0: int_mask[:int(non_zero[0][0])] = 1
        else: int_mask += 1

        non_zero = (eff_gts == 0).nonzero()
        if len(non_zero[0]) != 0: eff_mask[:int(non_zero[0][0])] = 1
        else: eff_mask += 1

        non_zero = (att_gts == 0).nonzero()
        if len(non_zero[0]) != 0: att_mask[:int(non_zero[0][0])] = 1
        else: att_mask += 1

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['cap_labels'] = torch.from_numpy(cap_gts).type(torch.LongTensor)
        data['cap_masks'] = torch.from_numpy(cap_mask).type(torch.FloatTensor)
        data['int_labels'] = torch.from_numpy(int_gts).type(torch.LongTensor)
        data['int_masks'] = torch.from_numpy(int_mask).type(torch.FloatTensor)
        data['eff_labels'] = torch.from_numpy(eff_gts).type(torch.LongTensor)
        data['eff_masks'] = torch.from_numpy(eff_mask).type(torch.FloatTensor)
        data['att_labels'] = torch.from_numpy(att_gts).type(torch.LongTensor)
        data['att_masks'] = torch.from_numpy(att_mask).type(torch.FloatTensor)

        data['video_ids'] = 'video%i' % ix

        # Concatenate all CMS
        int_str = ''
        for _ in intentions:
            int_str += ';' + _[0]
        att_str = ''
        for _ in attributes:
            att_str += ';' + _[0]
        eff_str = ''
        for _ in effects:
            eff_str += ';' + _[0]
        return data, int_str, att_str, eff_str

    def __len__(self):
        return len(self.splits[self.mode])
