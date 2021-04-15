import sys
import json
import torch
from opts import *
import numpy as np
import nltk
from utils.utils import *
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torch.utils.data import DataLoader
from model.transformer.Constants import *
from utils.gt_caps_dataloader import VideoDataset

# sys.path.append("./pycocoevalcap/")


def test(loader, model, opt, cap_vocab, cms_vocab):
    res = {}
    gts = {}
    eval_id = 0

    total_cms = set()
    ppl_scores = []

    for batch_id, raw_data in enumerate(loader):
        if opt['cuda']: torch.cuda.synchronize()

        # iterate each video within the batch
        for iterate_id in range(len(raw_data)):
            fc_feats = raw_data[iterate_id][0]['fc_feats'].unsqueeze(0)
            video_ids = raw_data[iterate_id][0]['video_ids']
            cap_labels = raw_data[iterate_id][0]['cap_labels']

            if opt['cms'] == 'int':
                cms_list = raw_data[iterate_id][1]
            elif opt['cms'] == 'eff':
                cms_list = raw_data[iterate_id][2]
            else:
                cms_list = raw_data[iterate_id][3]

            if opt['cuda']:
                # cms_list = cms_list.cuda()
                cap_labels = cap_labels.cuda()
                fc_feats = fc_feats.cuda()

            # repeat the fc features for num_cap times
            fc_feats = fc_feats.repeat(len(cap_labels), 1, 1)

            # iterate through all captions per video
            with torch.no_grad():

                # Note, currently we used BEAM search to decode the captions, while greedy strategy
                #  should yield close or even better results.

                # Beam Search Starts From Here
                _, cms_batch_hyp = model(fc_feats, cap_labels=cap_labels, cms_labels=None, mode='test')

            for random_id in range(cap_labels.shape[0]):
                # Print out the predicted sentences and GT
                if EOS in cms_batch_hyp[random_id]:
                    stop_id = list(cms_batch_hyp[random_id]).index(EOS)
                else:
                    stop_id = -1

                cms = list_to_sentence([cms_vocab[str(widx.detach().cpu().numpy())] for widx in
                                        cms_batch_hyp[random_id][: stop_id] if widx != 0])
                cap_gt = list_to_sentence([cap_vocab[str(word.cpu().numpy())] for word in
                                           cap_labels[random_id, 1:] if word != 0][0:-1])

                print(video_ids, '\n', 'Predicted CMS: ', cms)
                print('GT Caption: ', cap_gt)
                print('GT CMS Knowledge: ', cms_list[random_id].split(';')[1:])
                print('\n')
                print(batch_id * opt['batch_size'], ' out of ', '3010')

                # Save for evaluation
                cmses = cms_list[random_id].split(';')[1:]
                res[eval_id] = [cms]
                gts[eval_id] = cmses

                eval_id += 1

                ppl_corpus = ''
                for c in cmses:
                    total_cms.add(c.lower())
                    ppl_corpus += ' ' + c.lower()
                tokens = nltk.word_tokenize(ppl_corpus)
                unigram_model = unigram(tokens)
                ppl_scores.append(perplexity(cms.lower(), unigram_model))

    # Compute PPL score
    print('Perplexity score: ', sum(ppl_scores)/len(ppl_scores))

    avg_bleu_score, bleu_scores = Bleu(4).compute_score(gts, res)
    avg_cider_score, cider_scores = Cider().compute_score(gts, res)
    avg_meteor_score, meteor_scores = Meteor().compute_score(gts, res)
    avg_rouge_score, rouge_scores = Rouge().compute_score(gts, res)
    print('C, M, R, B:', avg_cider_score, avg_meteor_score, avg_rouge_score, avg_bleu_score)


def main(opt):
    dataset = VideoDataset(opt, 'test')
    dataloader = DataLoader(dataset, collate_fn=test_collate_fn, batch_size=opt['batch_size'],
                            shuffle=False)
    opt['cms_vocab_size'] = dataset.get_cms_vocab_size()
    opt['cap_vocab_size'] = dataset.get_cap_vocab_size()

    if opt['cms'] == 'int':
        cms_text_length = opt['int_max_len']
    elif opt['cms'] == 'eff':
        cms_text_length = opt['eff_max_len']
    else:
        cms_text_length = opt['att_max_len']

    from model.DecoderRNN import DecoderRNN
    from model.S2VT_EncoderRNN import EncoderRNN
    from model.S2VTAttModel import S2VTAttModel

    encoder = EncoderRNN(2048, 512, 0, 0.2, rnn_cell='gru')

    decoder = DecoderRNN(dataset.get_cap_vocab_size(), opt['cap_max_len'], 512, 512)

    cms_decoder = DecoderRNN(dataset.get_cms_vocab_size(), cms_text_length, 512, 512)

    model = S2VTAttModel(encoder, decoder, cms_decoder)


    model.eval()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    model.load_state_dict(torch.load(opt['load_checkpoint']))

    if opt['cuda']:
        model = model.cuda()

    test(dataloader, model, opt, dataset.get_cap_vocab(), dataset.get_cms_vocab())


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    opt['captions'] = json.load(open(opt['caption_json']))
    opt['batch_size'] = 30

    main(opt)
