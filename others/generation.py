import json
import torch
import random
import numpy as np
from opts import *
from model.Model import Model
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Constants import *
from nltk.translate.bleu_score import corpus_bleu
from model.transformer.Translator import translate_batch

import sys
sys.path.append("utils/pycocoevalcap/")


def pos_emb_generation(visual_feats):
    '''
        Generate the position embedding input for Transformers.
    '''
    seq = list(range(1, visual_feats.shape[1] + 1))
    src_pos = torch.tensor([seq] * visual_feats.shape[0]).cuda()
    return src_pos


def list_to_sentence(list):
    sentence = ''
    for element in list:
        sentence += ' ' + element
    return sentence


def test(loader, model, opt, cap_vocab, cms_vocab):
    bleu_scores = []
    write_to_txt = []

    gts = []
    res = []
    for batch_id, data in enumerate(loader):

        fc_feats = data['fc_feats'].cuda()
        cap_labels = data['cap_labels'].cuda()
        video_ids = data['video_ids']

        with torch.no_grad():
            # Beam Search Starts From Here
            try:
                batch_hyp, cms_batch_hyp = translate_batch(model, fc_feats, opt)
            except:
                continue

        # Stack all GTs captions
        references = []
        for video in video_ids:
            video_caps = []
            for cap in opt['captions'][video]:
                for _ in cap['attribute']:
                    video_caps.append(cap['final_caption'][1:-1] + _[1][1:-1])
            references.append(video_caps)

        # Stack all Predicted Captions
        hypotheses = []
        for cms_predict, predict in zip(cms_batch_hyp, batch_hyp):
            _ = []
            if CAP_EOS in predict[0]:
                sep_id = predict[0].index(CAP_EOS)
            else:
                sep_id = -1
            for word in predict[0][1: sep_id]:
                _.append(cap_vocab[str(word)])

            if CAP_EOS in cms_predict[0]:
                sep_id = cms_predict[0].index(CAP_EOS)
            else:
                sep_id = -1
            for word in cms_predict[0][0: sep_id]:
                _.append(cms_vocab[str(word)])
            hypotheses.append(_)

        # Print out the predicted sentences and GT
        for random_id in range(5):
            if 0 in batch_hyp[random_id][0]:
                stop_idx = batch_hyp[random_id][0].index(EOS)
            else:
                stop_idx = -1

            video_id = video_ids[random_id]
            cap = list_to_sentence([cap_vocab[str(widx)] for widx in batch_hyp[random_id][0][1: stop_idx] if widx != 0])
            cms = list_to_sentence([cms_vocab[str(widx)] for widx in cms_batch_hyp[random_id][0][: -1] if widx != 0])
            cap_gt = list_to_sentence([cap_vocab[str(word.cpu().numpy())] for word in cap_labels[random_id, 1:] if word != 0][0:-1])
            _ = str(video_id + ',' + cap + ',' + cms + ',' + cap_gt)
            write_to_txt.append(_)
            print('Generated Caption:', cap, ' ', 'Generated CMS:', cms)
            print('GT Caption:', cap_gt)
            print('\n')
            print(batch_id, ' ', batch_id * opt['batch_size'], ' out of ', '3010')

        # Compute the BLEU-4 score
        bleu_1 = corpus_bleu(references, hypotheses, weights=[1, 0, 0, 0])
        bleu_2 = corpus_bleu(references, hypotheses, weights=[0.5, 0.5, 0, 0])
        bleu_3 = corpus_bleu(references, hypotheses, weights=[0.333, 0.333, 0.333, 0])
        bleu_4 = corpus_bleu(references, hypotheses, weights=[0.25, 0.25, 0.25, 0.25])
        bleu_scores.append([bleu_1, bleu_2, bleu_3, bleu_4])

    print("Bleu scores 1-4:", np.mean(np.asarray(bleu_scores), 0))


def main(opt):
    dataset = VideoDataset(opt, 'test')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False)
    opt['cms_vocab_size'] = dataset.get_cms_vocab_size()
    opt['cap_vocab_size'] = dataset.get_cap_vocab_size()

    if opt['cms'] == 'int':
        cms_text_length = opt['int_max_len']
    elif opt['cms'] == 'eff':
        cms_text_length = opt['eff_max_len']
    else:
        cms_text_length = opt['att_max_len']

    model = Model(
        dataset.get_cap_vocab_size(),
        dataset.get_cms_vocab_size(),
        cap_max_seq=opt['cap_max_len'],
        cms_max_seq=cms_text_length,
        tgt_emb_prj_weight_sharing=True,
        vis_emb=opt['dim_vis_feat'],
        rnn_layers=opt['rnn_layer'],
        d_k=opt['dim_head'],
        d_v=opt['dim_head'],
        d_model=opt['dim_model'],
        d_word_vec=opt['dim_word'],
        d_inner=opt['dim_inner'],
        n_layers=opt['num_layer'],
        n_head=opt['num_head'],
        dropout=opt['dropout'])

    if len(opt['load_checkpoint']) != 0:
        state_dict = torch.load(opt['load_checkpoint'])
        model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()
    test(dataloader, model, opt, dataset.get_cap_vocab(), dataset.get_cms_vocab())


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    opt['captions'] = json.load(open(opt['caption_json']))
    opt['batch_size'] = 30
    main(opt)