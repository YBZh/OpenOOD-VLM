from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
        self.in_score = self.args.in_score  # sum | max
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # id_loader_dict['train']
        pass
        # pdb.set_trace()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        # class_num = net.n_cls
        class_num = 1000
        image_features, text_features, logit_scale = net(data, return_feat=True)
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention,
        if len(text_features.shape) == 3: ## 11K*7*512, weighting the text features instead of simple average, Do not work, not used. 
            sim = text_features @ image_features.t()  ## 11k*7*256
             #### may combine with temperature and softmax !!, here use cose sim directly. here with negative values.
            sim = torch.exp(-self.beta * (-sim + 1))
            temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
            sa_text_features = temp.sum(2) ## 256*11k*512
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
            output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        else:
            output = logit_scale * image_features @ text_features.t() # batch * class.
        
        _, pred_in = torch.max(output[:, :class_num], dim=1)

        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in_vanilla = scores.mean(dim=-1) ### the mean ID score of multiple groups. 
        conf = conf_in_vanilla
        
        # ############################### only score in.
        # output_only_in = output[:, :class_num]
        # output_only_out = output[:, class_num:]
        # score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        # conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        # cosin_only_in, _ = torch.max(output_only_in, dim=1)
        # ############################## including score out. 
        # score = torch.softmax(output / self.tau, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)
        # if self.in_score == 'oodscore' or self.in_score == 'sum':
        #     conf = conf_in 
        ################# tested variants, not effective.
        # elif self.in_score == 'oodscore_wiwopt_sum':
        #     vanilla_text_classifier = net.text_features.mean(1)
        #     vanilla_text_classifier /= vanilla_text_classifier.norm(dim=-1, keepdim=True)  ## renorm.
        #     output_vanilla = logit_scale * image_features @ vanilla_text_classifier.t() # batch * class.
        #     score_vanilla = torch.softmax(output_vanilla, dim=1)
        #     conf_in_pt = torch.sum(score_vanilla[:, :class_num], dim=1)
        #     # pdb.set_trace()
        #     # # self.text_features
        #     conf = conf_in + conf_in_pt  
        # elif self.in_score == 'oodscore_wiwopt_mul':
        #     vanilla_text_classifier = net.text_features.mean(1)
        #     vanilla_text_classifier /= vanilla_text_classifier.norm(dim=-1, keepdim=True)  ## renorm.
        #     output_vanilla = logit_scale * image_features @ vanilla_text_classifier.t() # batch * class.
        #     score_vanilla = torch.softmax(output_vanilla, dim=1)
        #     conf_in_pt = torch.sum(score_vanilla[:, :class_num], dim=1)
        #     # pdb.set_trace()
        #     # # self.text_features
        #     conf = conf_in * conf_in_pt 
        # elif self.in_score == 'oodscore_cosin':
        #     # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
        #     conf = conf_out * cosin_only_in  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        # elif self.in_score == 'oodscore_cosout':
        #     # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
        #     # pdb.set_trace()
        #     conf = - conf_out * output_only_out[:,0]  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        # elif self.in_score == 'maxidcosdis':
        #     conf = cosin_only_in
        # elif self.in_score == 'maxoodcosdis':
        #     conf = - output_only_out[:,0]
        # elif self.in_score == 'maxidscore':
        #     conf = conf_only_in
        # elif self.in_score == 'energy': 
        #     # bad results. 
        #     conf = self.tau * torch.log(torch.exp(output_only_in / self.tau).sum(1)) - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        # elif self.in_score == 'ood_energy': 
        #     # bad results. 
        #     conf = - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        # else:
        #     raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()
        # pdb.set_trace()
        # conf, pred = torch.max(score, dim=1)
        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau


def pca(X, k=300):
    # 中心化数据
    X_mean = X.mean(0)
    X = X - X_mean.expand_as(X)
    # SVD
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]
    # return torch.mm(X, U[:, :k])

# LDA
def lda(X, y, k=256):
    # 计算类内散度矩阵
    Sw = torch.zeros(X.shape[1], X.shape[1])
    # 计算类间散度矩阵
    Sb = torch.zeros(X.shape[1], X.shape[1])
    # 类别的均值
    mean_overall = torch.mean(X, dim=0)
    classes = torch.unique(y)
    for i in classes:
        Xi = X[y == i]
        mean_c = torch.mean(Xi, dim=0)
        # 类内散度矩阵
        Sw += torch.matmul((Xi - mean_c).t(), (Xi - mean_c))
        # 类间散度矩阵
        Ni = Xi.shape[0]
        mean_diff = (mean_c - mean_overall).view(-1, 1)
        Sb += Ni * torch.matmul(mean_diff, mean_diff.t())
        
    # 求解Sw^-1 * Sb的特征值和特征向量
    eigvals, eigvecs = torch.eig(torch.mm(torch.pinverse(Sw), Sb), eigenvectors=True)
    
    # 提取前k个特征向量
    eigvecs = eigvecs[:, :k]
    
    return torch.mm(X, eigvecs)

############################ following nnguide!!  besides single points, using its neighbor images. 
class OneOodPromptDevelopPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptDevelopPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ### get the image feature of each classes, construct the image feature classifier/memory.
        net.eval()
        # net.text_features 11k*512, if empty, fill with net.
        out_dim = net.n_output
        if self.setup_flag:
            # estimate class mean from training set
            # net.text_features.t() ## N*512
            # net.logit_scale
            # with torch.no_grad():
            #     output_text = net.logit_scale * net.text_features.t() @ net.text_features # class_num * class_num
            #     output_text = torch.softmax(output_text, dim=1) 
            # all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
            # self.text_idscore_cache = all_weights_text ## 11k

            with torch.no_grad():
                output_text = net.logit_scale * net.text_features_unselected.t() @ net.text_features # class_num * class_num
                output_text = torch.softmax(output_text, dim=1) 
            all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
            self.text_idscore_unselected_cache = all_weights_text ## 11k

            print('\n geting image features from (generated) training set...')
            all_feats = []
            all_weights = []
            for i in range(out_dim):
                all_feats.append([])  ## category-wise feature list
                all_weights.append([])
            
            ############################## init with text feature.
            # # pdb.set_trace()
            # for i in range(out_dim):
            #     all_feats[i].append(net.text_features.t()[i].unsqueeze(0))  ## category-wise feature list

            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    image_features, text_features, logit_scale = net(data, return_feat=True)
                    ####### weighting image features according to the classification probability.
                    output = logit_scale * image_features @ text_features.t() # batch * class, using the classification score as weights. 
                    output_prob = torch.softmax(output, dim=1) ## use category prob. as weights or ID prob. as weights?
                    indice = torch.arange(output.size(0))
                    # pdb.set_trace()
                    ########################## category prob level weights.
                    # weights = output_prob[indice, labels]
                    ########################## id/ood level sample weights
                    weights = output_prob[:, :1000].sum(1)
                    # for i in range(image_features.size(0)):
                    #     if labels[i].item() < 1000: 
                    #         ## ID class
                    #         weights[i] = output_prob[i, :1000].sum()
                    #     else:
                    #         weights[i] = output_prob[i, 1000:].sum()

                    for i in range(image_features.size(0)):
                        all_feats[labels[i]].append(image_features[i].unsqueeze(0))
                        all_weights[labels[i]].append(weights[i].unsqueeze(0))

            for i in range(len(all_feats)):
                if len(all_feats[i]) != 0:
                    all_feats[i] = torch.cat(all_feats[i], dim=0)
                    all_weights[i] = torch.cat(all_weights[i], dim=0)
                
                    # pdb.set_trace()
                    # cate_mean = (torch.cat(all_feats[i],dim=0) * torch.cat(all_weights[i], dim=0).unsqueeze(1)).mean(0,keepdim=True)
                    # cate_mean = torch.cat(all_feats[i],dim=0).mean(0,keepdim=True)
                    # cate_mean /= cate_mean.norm(dim=-1, keepdim=True)  ## renorm.  ## how to weight different image features?
                    # all_feats[i] = cate_mean
            all_feats = [x for x in all_feats if isinstance(x, torch.Tensor)]
            all_weights = [x for x in all_weights if isinstance(x, torch.Tensor)]

            all_feats_id = torch.cat(all_feats[:1000], dim=0) ## 11k * 512
            all_weights_id = torch.cat(all_weights[:1000], dim=0) ## 11k * 512
            # pdb.set_trace()
            all_feats_ood = torch.cat(all_feats[1000:], dim=0) ## 11k * 512
            all_weights_ood = torch.cat(all_weights[1000:], dim=0) ## 11k * 512
            self.image_classifier = all_feats_id

            self.image_feat_cache_id = all_feats_id
            self.image_idscore_cache_id = all_weights_id

            self.image_feat_cache_ood = all_feats_ood
            self.image_idscore_cache_ood = all_weights_ood

            self.image_feat_cache = torch.cat((self.image_feat_cache_id, self.image_feat_cache_ood), dim=0)
            self.image_idscore_cache = torch.cat(( self.image_idscore_cache_id,  self.image_idscore_cache_ood), dim=0)
        else:
            pass



    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        # pdb.set_trace()
        image_features, text_features, logit_scale = net(data, return_feat=True)
        # image_classifier = self.image_classifier

        ## image_features: 256*512, 
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention,
        # if len(text_features.shape) == 3: ## 11K*7*512
        #     sim = text_features @ image_features.t()  ## 11k*7*256
        #      #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        #     sim = torch.exp(-self.beta * (-sim + 1))
        #     temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
        #     sa_text_features = temp.sum(2) ## 256*11k*512
        #     sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        #     output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        # else:
        #     output = logit_scale * image_features @ text_features.t() # batch * class.
            # output_image = logit_scale * image_features @ image_classifier.t() # batch * class.
        output = logit_scale * image_features @ text_features.t() # batch * class.
        # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(output[:, :class_num], dim=1)

        # _, pred_in_img = torch.max(output_image[:, :class_num], dim=1)
        # ############################### only score in. here is a 
        # output_only_in = output[:, :class_num]
        # output_only_out = output[:, class_num:]
        # score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        # conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        # cosin_only_in, _ = torch.max(output_only_in, dim=1)
        # ###########
        # print('pay attention, no scale is applied here.')
        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            # pdb.set_trace()
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # # pdb.set_trace()
        # # self.image_feat_cache N*512
        # # self.image_idscore_cache  N
        # # image_features ## 256*512
        # # cossim = torch.matmul(image_features, self.image_feat_cache.t()) ## 256*N
        # # guided_sim = cossim * self.image_idscore_cache ## 256*N
        # # value, indice = torch.sort(guided_sim, 1) # 256 * N, small 2 large
        # # guided_score = value[:, -self.beta:].mean(1)

        # ############### text NNguided with positive and negtive KNN, with unselected text 
        # ############### 或许可以只根据cos sim 对text id score 进行加权， 总之这里需要再仔细斟酌一下。总的思路是ensemble NN 得到当前test point 的 ood score. 
        # cossim = torch.matmul(image_features, net.text_features_unselected) ## 256*8W
        # value_cos, indice_cos = torch.sort(cossim, 1) # 256 * N, small 2 large

        # cossim_selected = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) ## 256*beta
        # # prob = torch.softmax(cossim_selected / self.tau, dim=1)
        # prob = torch.exp(-self.tau * (1 - cossim_selected)) ## not sentative to the value of beta.
        # prob = prob / prob.sum(1, keepdim=True)
        # guided_score_neartext_modulated = (prob * self.text_idscore_unselected_cache[indice_cos[:, -self.beta:]]).sum(1)


        # guided_score_neardata = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) * self.text_idscore_unselected_cache[indice_cos[:, -self.beta:]] ## 256*N
        # guided_score_neartext = guided_score_neardata.mean(1)

        # guided_sim_id = cossim * self.text_idscore_unselected_cache ## 256*N  #### 这里是否需要加入scaling & softmax??
        # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id

        # guided_sim_ood = cossim * (1 - self.text_idscore_unselected_cache) ## 256*N
        # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood； 这里多个直接求平均值或许可以改成加权和？

        # guided_score_text_mul = guided_score_iddata * (-guided_score_ooddata)  ## score 没有归一化，会有比较大的影响
        # guided_score_text_norm = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)  ### 这个结果差不多，有少量提升;  这里的乘法关系可能也不合适，改成加法。
        # guided_score_text_add = guided_score_iddata - guided_score_ooddata  ### 上面乘法结果更好好一些？ 神奇。。
        

        # # ############### text NNguided with positive and negtive KNN, with selected text 
        # # cossim_id = torch.matmul(image_features, text_features.t()[:, :1000]) ## 256*N
        # # guided_sim_id = cossim_id * self.text_idscore_cache[:1000] ## 256*N
        # # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id.

        # # cossim_ood = torch.matmul(image_features, text_features.t()[:, 1000:]) ## 256*N
        # # guided_sim_ood = cossim_ood * (1 - self.text_idscore_cache[1000:]) ## 256*N
        # # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood. 
        # # guided_score_text = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)  ###

        # ############### image NNguided with positive and negtive KNN; here is the focus!!.
        # cossim = torch.matmul(image_features, self.image_feat_cache.t()) ## 256*8W
        # value_cos, indice_cos = torch.sort(cossim, 1) # 256 * N, small 2 large

        # cossim_selected = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) ## 256*beta
        # # prob = torch.softmax(cossim_selected / self.tau, dim=1)
        # prob = torch.exp(-self.tau * (1 - cossim_selected)) ## not sentative to the value of beta.
        # prob = prob / prob.sum(1, keepdim=True)
        # guided_score_nearimg_modulated = (prob * self.image_idscore_cache[indice_cos[:, -self.beta:]]).sum(1)

        # guided_score_neardata = torch.gather(cossim, 1, indice_cos[:, -self.beta:]) * self.image_idscore_cache[indice_cos[:, -self.beta:]] ## 256*N
        # guided_score_nearimg = guided_score_neardata.mean(1)

        # cossim_id = torch.matmul(image_features, self.image_feat_cache_id.t()) ## 256*N
        # guided_sim_id = cossim_id * self.image_idscore_cache_id ## 256*N
        # value, indice = torch.sort(guided_sim_id, 1) # 256 * N, small 2 large
        # guided_score_iddata = value[:, -self.beta:].mean(1) ## larger means id.

        # cossim_ood = torch.matmul(image_features, self.image_feat_cache_ood.t()) ## 256*N
        # guided_sim_ood = cossim_ood * (1 - self.image_idscore_cache_ood) ## 256*N
        # value, indice = torch.sort(guided_sim_ood, 1) # 256 * N, small 2 large
        # guided_score_ooddata = value[:, -self.beta:].mean(1) ## larger means ood. 

        # guided_score_img_norm = guided_score_iddata / (guided_score_iddata + guided_score_ooddata)
        # guided_score_img_mul = guided_score_iddata * (-guided_score_ooddata) 
        # guided_score_img_add = guided_score_iddata - guided_score_ooddata  


        # guided_score = value[:, -self.beta:].mean(1)


        # # print('pay attention, an additional 3.0 scale is applied here.')
        # score_image = torch.softmax(output_image / self.tau, dim=1)
        # conf_in_image = torch.sum(score_image[:, :class_num], dim=1)
        # conf_out_image = torch.sum(score_image[:, class_num:], dim=1)

        # score_merge = score * score_image  
        # conf_in_merge = torch.sum(score_merge[:, :class_num], dim=1)
        # # 

        # max in prob - max out prob
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            conf = conf_in  ## = 1-conf_out
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

 
# # the last output dim is ood dim. vanilla develop, before NNguide.
# class OneOodPromptDevelopPostprocessor(BasePostprocessor):
#     def __init__(self, config):
#         super(OneOodPromptDevelopPostprocessor, self).__init__(config)
#         self.args = self.config.postprocessor.postprocessor_args
#         self.tau = self.args.tau
#         self.beta = self.args.beta
#         self.args_dict = self.config.postprocessor.postprocessor_sweep
#         self.in_score = self.args.in_score # sum | max
#         self.setup_flag = False
#         self.proj_flag = False
    
#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         ### get the image feature of each classes, construct the image feature classifier. 
#         net.eval()
#         # net.text_features 11k*512, if empty, fill with net.
#         out_dim = net.n_output
#         if not self.setup_flag:
#             # estimate class mean from training set
#             print('\n Estimating class mean from training set...')
#             all_feats = []
#             all_weights = []
#             for i in range(out_dim):
#                 all_feats.append([])  ## category-wise feature list
#                 all_weights.append([])
            
#             ############################## init with text feature.
#             # # pdb.set_trace()
#             # for i in range(out_dim):
#             #     all_feats[i].append(net.text_features.t()[i].unsqueeze(0))  ## category-wise feature list
#             with torch.no_grad():
#                 for batch in tqdm(id_loader_dict['train'],
#                                   desc='Setup: ',
#                                   position=0,
#                                   leave=True):
#                     data, labels = batch['data'].cuda(), batch['label']
#                     image_features, text_features, logit_scale = net(data, return_feat=True)
#                     ####### weighting image features according to the classification probability.
#                     output = logit_scale * image_features @ text_features.t() # batch * class, using the classification score as weights. 
#                     output_prob = torch.softmax(output, dim=1) ## use category prob. as weights or ID prob. as weights?
#                     indice = torch.arange(output.size(0))
#                     # pdb.set_trace()
#                     ########################## category prob level weights.
#                     weights = output_prob[indice, labels]
#                     # ########################## id/ood level sample weights
#                     # for i in range(image_features.size(0)):
#                     #     if labels[i].item() < 1000: 
#                     #         ## ID class
#                     #         weights[i] = output_prob[i, :1000].sum()
#                     #     else:
#                     #         weights[i] = output_prob[i, 1000:].sum()

#                     for i in range(image_features.size(0)):
#                         all_feats[labels[i]].append(image_features[i].unsqueeze(0))
#                         all_weights[labels[i]].append(weights[i].unsqueeze(0))

#             for i in range(len(all_feats)):
#                 # pdb.set_trace()
#                 # cate_mean = (torch.cat(all_feats[i],dim=0) * torch.cat(all_weights[i], dim=0).unsqueeze(1)).mean(0,keepdim=True)
#                 cate_mean = torch.cat(all_feats[i],dim=0).mean(0,keepdim=True)
#                 cate_mean /= cate_mean.norm(dim=-1, keepdim=True)  ## renorm.  ## how to weight different image features?
#                 all_feats[i] = cate_mean
#             # pdb.set_trace()
#             all_feats = torch.cat(all_feats, dim=0) ## 11k * 512
#             self.image_classifier = all_feats
#         else:
#             pass
#         if self.proj_flag:
#             # 计算两个类别的均值
#             X_a = text_features[:1000, :]
#             X_b = text_features[1000:, :]
#             mean_a = torch.mean(X_a, dim=0)
#             mean_b = torch.mean(X_b, dim=0)

#             # 计算类内散度矩阵
#             Sw_a = torch.matmul((X_a - mean_a).t(), (X_a - mean_a))
#             Sw_b = torch.matmul((X_b - mean_b).t(), (X_b - mean_b))
#             Sw = Sw_a + Sw_b

#             # 计算类间散度矩阵
#             mean_diff = (mean_a - mean_b).view(-1, 1)
#             Sb = mean_diff.matmul(mean_diff.t())

#             # 求解Sw^-1 * Sb的特征值和特征向量
#             e, v = torch.symeig(torch.pinverse(Sw.float()).matmul(Sb.float()), eigenvectors=True)

#             # 取出最大的特征值对应的特征向量
#             w = v[:, -1:]
#             self.proj_matrix = w

#             # self.proj_matrix = pca(text_features.float(), 400) 
#     # find the direction that maximize the seperation between id and ood samples. 
#     # find the direction the maximize the consistency of text/image ID features. 


#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         net.eval()
#         class_num = net.n_cls
#         # pdb.set_trace()
#         image_features, text_features, logit_scale = net(data, return_feat=True)
#         image_classifier = self.image_classifier
#         # if self.proj_flag:
#         #     # pdb.set_trace()
#         #     conf_in_fisher = torch.mm(image_features.float(), self.proj_matrix)[:,0]
#         #     # text_features = torch.mm(text_features.float(), self.proj_matrix)
#         #     # image_classifier = torch.mm(image_classifier.float(), self.proj_matrix)
#         ## image_features: 256*512, 
#         ## text_features: 11k*7*512
#         ## extract sample adaptative classifier via attention,
#         if len(text_features.shape) == 3: ## 11K*7*512
#             sim = text_features @ image_features.t()  ## 11k*7*256
#              #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
#             sim = torch.exp(-self.beta * (-sim + 1))
#             temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
#             sa_text_features = temp.sum(2) ## 256*11k*512
#             sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
#             output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
#         else:
#             output = logit_scale * image_features @ text_features.t() # batch * class.
#             output_image = logit_scale * image_features @ image_classifier.t() # batch * class.
#         # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
#         _, pred_in = torch.max(output[:, :class_num], dim=1)
#         _, pred_in_img = torch.max(output_image[:, :class_num], dim=1)
#         ############################### only score in. here is a 
#         output_only_in = output[:, :class_num]
#         output_only_out = output[:, class_num:]
#         score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
#         conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
#         cosin_only_in, _ = torch.max(output_only_in, dim=1)
#         ###########
#         output_image_only_in = output_image[:, :class_num]
#         output_image_only_out = output_image[:, class_num:]
#         score_only_in_image = torch.softmax(output_image_only_in / self.tau, dim=1)
#         conf_only_in_image, pred_only_in_image = torch.max(score_only_in_image, dim=1)
#         cosin_only_in_image, _ = torch.max(output_image_only_in, dim=1)
#         cosin_only_in_image_sum = torch.mean(output_image_only_in, dim=1)
#         # pdb.set_trace()
#         ##############################
#         # print('pay attention, no scale is applied here.')
#         score = torch.softmax(output, dim=1)
#         conf_in = torch.sum(score[:, :class_num], dim=1)
#         conf_out = torch.sum(score[:, class_num:], dim=1)

#         # print('pay attention, an additional 3.0 scale is applied here.')
#         score_image = torch.softmax(output_image / self.tau, dim=1)
#         conf_in_image = torch.sum(score_image[:, :class_num], dim=1)
#         conf_out_image = torch.sum(score_image[:, class_num:], dim=1)

#         score_merge = score * score_image  
#         conf_in_merge = torch.sum(score_merge[:, :class_num], dim=1)
#         # pdb.set_trace()

#         # max in prob - max out prob
#         if self.in_score == 'oodscore' or self.in_score == 'sum':
#             conf = conf_in  ## = 1-conf_out
#         elif self.in_score == 'oodscore_img': ## multi-modal
#             conf = conf_in_image 
#         elif self.in_score == 'oodscore_merge': ## 在image 分支把所有的tau 试了一遍，结果也比不过只用 image ood score 的结果，放弃。
#             conf = conf_in_merge 
#         elif self.in_score == 'oodscore_fisher': ## multi-modal
#             conf = conf_in_fisher
#             # pred_in = pred_in_img
#         elif self.in_score == 'oodscore_mm': ## multi-modal
#             conf = conf_in * conf_in_image 
#         elif self.in_score == 'oodscore_mm_add': ## multi-modal
#             conf = conf_in + conf_in_image*0.1
#             prob_only_in =  score_only_in + score_only_in_image * 0.1
#             _, pred_in = torch.max(prob_only_in, dim=1)
#         elif self.in_score == 'oodscore_cosin':
#             # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
#             conf = conf_in * cosin_only_in  ## with softmax + without softmax. 
#             # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
#         elif self.in_score == 'oodscore_cosout':
#             # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
#             # pdb.set_trace()
#             conf = conf_in * output_only_out[:,0]  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
#         elif self.in_score == 'maxidcosdis':
#             conf = cosin_only_in
#         elif self.in_score == 'maxidcosdis_img':
#             conf = cosin_only_in_image
#         elif self.in_score == 'sumidcosdis_img':
#             conf = cosin_only_in_image_sum
#         elif self.in_score == 'maxidscore':
#             conf = conf_only_in
#         elif self.in_score == 'energy': 
#             # bad results. 
#             conf = self.tau * torch.log(torch.exp(output_only_in / self.tau).sum(1)) - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
#         elif self.in_score == 'ood_energy': 
#             # bad results. 
#             conf = - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
#         else:
#             raise NotImplementedError
#         if torch.isnan(conf).any():
#             pdb.set_trace()

#         return pred_in, conf

#     def set_hyperparam(self, hyperparam: list):
#         self.tau = hyperparam[0]

#     def get_hyperparam(self):
#         return self.tau

