#!/usr/bin/env python

import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def accuracy(logits, targets, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    _, preds = logits.max(dim=-1)
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    acc = acc.mean()
    return acc


def perplexity(logits, targets, weight=None, padding_idx=None, device=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1), device=device)
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):

    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def knowledge_hit(preds, text_dict):
    hit_count = 0
    unknown = 0
    assert len(preds)==len(text_dict)
    for pred, text_entity in zip(preds, text_dict):
        if len(text_entity)==0:
            unknown += 1
            continue
        for entity in text_entity:
            if entity in pred:
                hit_count += 1
    hit_ratio = hit_count / (len(preds)-unknown)
    # print(unknown) # 1305
    return hit_ratio

def goal_hit(response_goals, preds):
    hit_count = 0
    unknown = 0
    assert len(response_goals) == len(preds)
    for pred, goals in zip(preds, response_goals):
        if len(goals) == 0:
            unknown += 1
            continue
        for goal in goals:
            if goal in pred:
            # if fuzz.partial_ratio(goal, pred) >= 80:
                hit_count += 1
    hit_ratio = hit_count / (len(preds)-unknown)
    return hit_ratio

def goal_hit_fuzz(response_goals, preds):
    hit_count = 0
    unknown = 0
    assert len(response_goals) == len(preds)
    for pred, goals in zip(preds, response_goals):
        if len(goals) == 0:
            unknown += 1
            continue
        for goal in goals:
            # if goal in pred:
            if fuzz.partial_ratio(goal, pred) >= 60:
                hit_count += 1
    hit_ratio = hit_count / (len(preds)-unknown)
    return hit_ratio

def metrics_cal_gen(preds,responses):
    def bleu_cal(sen1, tar1):
        bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
        bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
        bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
        return bleu1, bleu2, bleu3, bleu4

    def distinct_metrics(outs):
        # outputs is a list which contains several sentences, each sentence contains several words
        unigram_count = 0
        bigram_count = 0
        trigram_count=0
        quagram_count=0
        unigram_set = set()
        bigram_set = set()
        trigram_set=set()
        quagram_set=set()
        for sen in outs:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen)-2):
                trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_count+=1
                trigram_set.add(trg)
            for start in range(len(sen)-3):
                quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_count+=1
                quagram_set.add(quag)
        dis1 = len(unigram_set) / len(outs)#unigram_count
        dis2 = len(bigram_set) / len(outs)#bigram_count
        dis3 = len(trigram_set)/len(outs)#trigram_count
        dis4 = len(quagram_set)/len(outs)#quagram_count
        return dis1, dis2, dis3, dis4

    predict_s=preds
    golden_s=responses
    #print(rec_loss[0])
    #self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
    generated=[]
    metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

    for out, tar in zip(predict_s, golden_s):
        bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
        generated.append(out)
        metrics_gen['bleu1']+=bleu1
        metrics_gen['bleu2']+=bleu2
        metrics_gen['bleu3']+=bleu3
        metrics_gen['bleu4']+=bleu4
        metrics_gen['count']+=1

    dis1, dis2, dis3, dis4=distinct_metrics(generated)
    metrics_gen['dist1']=dis1
    metrics_gen['dist2']=dis2
    metrics_gen['dist3']=dis3
    metrics_gen['dist4']=dis4
    
    output_dict_gen={}
    for key in metrics_gen:
        if 'bleu' in key:
            output_dict_gen[key]=metrics_gen[key]/metrics_gen['count']
        else:
            output_dict_gen[key]=metrics_gen[key]
    return output_dict_gen

