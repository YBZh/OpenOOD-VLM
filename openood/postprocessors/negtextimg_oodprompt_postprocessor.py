from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
import pdb

# the last output dim is ood dim.
class OneOodPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = self.args.beta
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # id_loader_dict['train']
        pass
        # pdb.set_trace()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention,
        if len(text_features.shape) == 3: ## 11K*7*512
            sim = text_features @ image_features.t()  ## 11k*7*256
             #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
            sim = torch.exp(-self.beta * (-sim + 1))
            temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
            sa_text_features = temp.sum(2) ## 256*11k*512
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
            output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        else:
            output = logit_scale * image_features @ text_features.t() # batch * class.
        
        _, pred_in = torch.max(output[:, :class_num], dim=1)
        ############################### only score in.
        output_only_in = output[:, :class_num]
        output_only_out = output[:, class_num:]
        score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        cosin_only_in, _ = torch.max(output_only_in, dim=1)
        ############################## including score out. 
        score = torch.softmax(output / self.tau, dim=1)
        conf_in = torch.sum(score[:, :class_num], dim=1)
        conf_out = torch.sum(score[:, class_num:], dim=1)
        # max in prob - max out prob
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            conf = conf_in - conf_out 
        elif self.in_score == 'oodscore_cosin':
            # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
            conf = conf_out * cosin_only_in  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        elif self.in_score == 'oodscore_cosout':
            # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
            # pdb.set_trace()
            conf = - conf_out * output_only_out[:,0]  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        elif self.in_score == 'maxidcosdis':
            conf = cosin_only_in
        elif self.in_score == 'maxoodcosdis':
            conf = - output_only_out[:,0]
        elif self.in_score == 'maxidscore':
            conf = conf_only_in
        elif self.in_score == 'energy': 
            # bad results. 
            conf = self.tau * torch.log(torch.exp(output_only_in / self.tau).sum(1)) - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        elif self.in_score == 'ood_energy': 
            # bad results. 
            conf = - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()
        # pdb.set_trace()
        # conf, pred = torch.max(score, dim=1)
        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau



# the last output dim is ood dim.
class OneOodPromptDevelopPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        image_features, text_features, logit_scale = net(data, return_feat=True)
        output = logit_scale * image_features @ text_features.t() # batch * class.
        ############################### only score in.
        output_only_in = output[:, :-1]
        score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        cosin_only_in, _ = torch.max(output_only_in, dim=1)
        ############################## including score out. 
        score = torch.softmax(output / self.tau, dim=1)
        conf_in, pred_in = torch.max(score[:, :-1], dim=1)
        conf_in = torch.sum(score[:, :-1], dim=1)
        # if self.in_score == 'sum':
        #     conf_in = torch.sum(score[:, :-1], dim=1)
        # elif self.in_score == 'max':
        #     raise NotImplementedError # use the sum in_score, deactivate the max score.
        # else:
        #     raise NotImplementedError
        conf_out = score[:, -1]
        # max in prob - max out prob
        conf = conf_in - conf_out  
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            pass
        elif self.in_score == 'oodscore_cosdis':
            # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
            conf = conf * cosin_only_in  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        else:
            raise NotImplementedError

        # pdb.set_trace()
        # conf, pred = torch.max(score, dim=1)
        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
