import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import model.transformer.Constants as Constants

# Construct the uni-gram language model
def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model [f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] /= N
    return model


# Computes perplexity of the UniGram model on a test-set
def perplexity(testset, model):
    testset = testset.split()
    ppl = 1
    N = 0
    for word in testset:
        N += 1
        ppl *= 1/model[word]
    ppl = pow(ppl, 1/float(N))
    return ppl


# Mean Pool Out the word2vec features of sentences.
def mean_pool_vec(sentence, wordmodel):
    vector = np.zeros(50)
    vector += np.mean([wordmodel[ele] for ele in sentence.split(' ') if ele in wordmodel.keys()], 0)
    return vector


def test_collate_fn(batch):
    '''
    :param batch: input batch data
    :return: aligned features
    '''

    return batch


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


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = target.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def pos_emb_generation(word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return tgt_pos*binary_mask


def show_prediction(seq_probs, labels, vocab, caption=True):
    '''
        :return: predicted words and GT words.
    '''
    # Print out the predicted sentences and GT
    _ = seq_probs.view(labels.shape[0], labels.shape[1], -1)[0]
    pred_idx = torch.argmax(_, 1)
    # print(' \n')
    if caption:
        print('Caption: ')
    else:
        print('CMS: ')

    pr = 'Generation: ', list_to_sentence([vocab[str(widx.cpu().numpy())] for widx in pred_idx if widx != 0])
    gt = 'GT: ', list_to_sentence([vocab[str(word.cpu().numpy())] for word in labels[0] if word != 0])
    print(pr)
    print(gt)
    return pr, gt

