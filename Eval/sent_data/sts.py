# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr, pearsonr

from sent_data.utils import cosine
from sent_data.sick import SICKEval


class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1], dtype=object)[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2], dtype=object)[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []

        def align_loss(x, y, alpha=2):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x*y).sum(dim=-1).mean()

        def uniform_loss(x, t=2):
            x = F.normalize(x, dim=-1, p=2)
            return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

        def dist_loss(x1, x2):
            pdist = nn.PairwiseDistance(p=2, keepdim=True)
            # x1 = F.normalize(x1, dim=-1, p=2)
            # x2 = F.normalize(x2, dim=-1, p=2)
            p = pdist(x1, x2)
            n = torch.cdist(x1, x2, p=2.0)
            mask = 1 - torch.eye(x1.shape[0]).to(x1.device)
            d = (n/p)[mask.bool()].mean()
            # d = ((n/p) * mask).sum(dim=-1) / mask.sum(dim=-1)
            return d

        def ang_dist_loss(x1, x2):
            cos = nn.CosineSimilarity(dim=-1)
            sim = cos(x1.unsqueeze(1), x2.unsqueeze(0))
            mask = 1 - torch.eye(x1.shape[0]).to(x1.device)
            d = ((sim[~mask.bool()] - sim) * mask).sum(dim=-1) / mask.sum(dim=-1)
            return d.mean()
        
        all_loss_align = []
        all_loss_uniform = []
        all_loss_dist = []
        all_loss_a_dist = []
        all_enc1 = []
        all_enc2 = []

        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                batch_gs_scores = gs_scores[ii:ii + params.batch_size] 

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    pos_indices = [i for i in range(len(batch_gs_scores)) if batch_gs_scores[i] >= 4.0]
                    enc1_norm = enc1/torch.norm(enc1, p=2, dim=-1, keepdim=True)
                    enc2_norm = enc2/torch.norm(enc2, p=2, dim=-1, keepdim=True)
                    enc1_pos = enc1_norm[pos_indices]
                    enc2_pos = enc2_norm[pos_indices]
                    loss_align = align_loss(enc1_pos, enc2_pos)
                    loss_uniform = uniform_loss(torch.cat((enc1_norm, enc2_norm), dim=0))
                    loss_dist = dist_loss(enc1_pos, enc2_pos)
                    loss_a_dist = ang_dist_loss(enc1_pos, enc2_pos)
                    all_loss_align.append(loss_align)
                    all_loss_uniform.append(loss_uniform)
                    all_loss_dist.append(loss_dist)
                    all_loss_a_dist.append(loss_a_dist)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)     
                    
                    all_enc1.append(enc1)
                    all_enc2.append(enc2)
            #####################align & uniform############################
            ##########################start#################################                              
 
            def collapse(x1, x2, x3=None, mode='unsupervised'):

                x = torch.cat([x1, x2], dim=0)
                if mode == 'supervised':
                    x = torch.cat([x, x3], dim=0)  

                mo, mr = std_(x)

                return mo, mr

            def std_(x):
                x = F.normalize(x, dim=-1, p=2)
                o_ = x.mean(dim=0)
                r_ = x - o_
                
                x_ = x.norm(dim=-1)
                o_ = o_.norm(dim=0)
                r_ = r_.norm(dim=-1)
                
                mo = (o_ / x_).mean()
                mr = (r_ / x_).mean() 

                return mo, mr
            
            # def dist(x, y):
            #     return (((x - y))**2).mean()


            # align = align_loss(
            #     torch.cat(all_enc1 + all_enc2), 
            #     torch.cat(all_enc3 + all_enc4), 
            #     ).item()

            # unif = (uniform_loss(torch.cat(all_enc1)).item() + uniform_loss(torch.cat(all_enc2)).item()) / 2
            mo, mr = collapse(torch.cat(all_enc1), torch.cat(all_enc2))
            # dist = dist(torch.cat(all_enc1), torch.cat(all_enc2))
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {
                                'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores),
                                'align': float(np.mean(all_loss_align)),
                                'uniform': float(np.mean(all_loss_uniform)),
                                'dist':float(np.mean(all_loss_dist)),
                                'a_dist':float(np.mean(all_loss_a_dist)),
                                'mo':mo.item(),
                                'mr':mr.item(),
                                # 'dist':dist.item(),
                                }
                        
            logging.debug('%s : pearson = %.4f, spearman = %.4f'  %
                          (dataset, 
                           results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0],
                        #    results[dataset]['uniform'],
                        #    results[dataset]['mo'],
                        #    results[dataset]['mr'],
                        #    results[dataset]['dist'],
                           ))
            ##########################end#################################
        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
