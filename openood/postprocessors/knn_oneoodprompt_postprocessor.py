from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.special import logsumexp
from copy import deepcopy
from .base_postprocessor import BasePostprocessor
import pdb

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


def knn_score(bankfeas, queryfeas, k=100, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))
    bankfeas = bankfeas.astype(np.float32)
    queryfeas = queryfeas.astype(np.float32)
    # pdb.set_trace()
    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    # pdb.set_trace()

    index.add(bankfeas)
    D, _ = index.search(queryfeas, k)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))
    return scores


class KnnOneOodPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KnnOneOodPromptPostprocessor, self).__init__(config)
        # self.args = self.config.postprocessor.postprocessor_args
        # self.K = self.args.K
        self.K = 10

        # self.alpha = self.args.alpha
        # self.activation_log = None
        # self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            bank_feas = []
            bank_logits = []
            bank_ood = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    # logit, feature = net(data, return_feature=True)
                    image_features, text_features, logit_scale = net(data, return_feat=True)
                    logit = logit_scale * image_features @ text_features.t() # batch * class.
                    logit = logit[:, :1000]

                    score = torch.softmax(logit, dim=1)
                    conf_out = 1 / (torch.sum(score[:, 1000:], dim=1) + 1e-3)

                    bank_feas.append(normalizer(image_features.data.cpu().numpy()))
                    bank_ood.append(conf_out.data.cpu().numpy())

                    # bank_logits.append(logit.data.cpu().numpy())
                    ############ keep all few shot samples.
                    # if len(bank_feas
                    #        ) * id_loader_dict['train'].batch_size > int(
                    #            len(id_loader_dict['train'].dataset) *
                    #            self.alpha):
                    #     break

            bank_feas = np.concatenate(bank_feas, axis=0)
            bank_ood = np.concatenate(bank_ood, axis=0)
            self.bank_guide = bank_feas
            self.bank_guide = bank_feas * bank_ood[:, np.newaxis] 

            # bank_confs = logsumexp(np.concatenate(bank_logits, axis=0), axis=-1)
            # self.bank_guide = bank_feas * bank_confs[:, None]  ## 这里的bank_conf 有没有影响不大

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # logit, feature = net(data, return_feature=True)
        image_features, text_features, logit_scale = net(data, return_feat=True)
        logit = logit_scale * image_features @ text_features.t() # batch * class.

        score = torch.softmax(logit, dim=1)
        conf_out = - torch.sum(score[:, 1000:], dim=1)

        logit = logit[:,:1000]

        feas_norm = normalizer(image_features.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)
        # score = torch.from_numpy(conf * energy)  ## 86/67
        # pdb.set_trace()
        score = torch.from_numpy(conf) * conf_out.cpu() ## 78/49.89  
        # score = torch.from_numpy(conf)

        _, pred = torch.max(torch.softmax(logit, dim=1), dim=1)
        return pred, score

    # def set_hyperparam(self, hyperparam: list):
    #     self.K = hyperparam[0]
    #     self.alpha = hyperparam[1]

    # def get_hyperparam(self):
    #     return [self.K, self.alpha]
