from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
import pdb

## update memory of each group individually. bad results.  random_permute must FALSE
class GroupTTAPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(GroupTTAPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True ## reset after each dataset. 
        self.memory_size = 30
        self.thres = self.args.thres
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

                    for i in range(image_features.size(0)):
                        all_feats[labels[i]].append(image_features[i].unsqueeze(0))
                        all_weights[labels[i]].append(weights[i].unsqueeze(0))

            for i in range(len(all_feats)):
                if len(all_feats[i]) != 0:
                    all_feats[i] = torch.cat(all_feats[i], dim=0)
                    all_weights[i] = torch.cat(all_weights[i], dim=0)
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

    def reset_memory(self):
        self.reset = True

    # Store (high confident) test features into the feature memory, and get new classifier by merging image features in the same memory slot.
    ### to do: multiple augmentations with filtering! 
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            self.reset=False
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class, to decide assign image to which category and the corresponding confidence.
        # prob_vanilla = torch.softmax(output_vanilla, dim=1)
        # conf_in_vanilla = torch.sum(prob_vanilla[:, :class_num], dim=1)
        pos_logit = output_vanilla[:, :class_num] ## B*C
        neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            raise NotImplementedError
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        group_size = neg_logit.size(-1)  ## number of negative classes in each group.
        scores = []
        for g in range(self.group_num):
            sub_logit = torch.cat([pos_logit, neg_logit[:, g, :]], dim=-1) 
            sub_prob = sub_logit.softmax(dim=-1)
            pos_score = sub_prob[:, :pos_logit.shape[1]].sum(dim=-1)

            ##### only select the OOD image to the OOD memory.
            activate_indicator = pos_score < (1-self.thres) ## only store high confident samples into feature memory.
            _, pred_all = torch.max(sub_prob[:, class_num:], dim=1)
            prob_ood = torch.softmax(neg_logit[:, g, :], dim=1)
            for i in range(activate_indicator.size(0)):
                if activate_indicator[i].item():
                    predicted_cate = pred_all[i].item() + class_num + g * group_size
                    predicted_prob = prob_ood[i]
                    current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                    # pdb.set_trace()
                    # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    if self.indice_memory[predicted_cate] == self.memory_size:
                        if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                            pass  ## the entropy of current test image is very large.
                        else:
                            # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                            _, indice = torch.sort(self.entropy_memory[predicted_cate])
                            to_replace_indice = indice[-1]  ## with max entropy, ascending.
                            self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                            self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                    else:
                        self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                        self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                        self.indice_memory[predicted_cate] += 1
                else:
                    pass

            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in_vanilla = scores.mean(dim=-1) ### the mean ID score of multiple groups. 


        # # pdb.set_trace()
        # # threshold = self.thres
        # activate_indicator = conf_in_vanilla > self.thres ## only store high confident samples into feature memory.
        # _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        # prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        # for i in range(activate_indicator.size(0)):
        #     if activate_indicator[i].item():
        #         predicted_cate = pred_all[i].item()
        #         predicted_prob = prob_id[i]
        #         current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
        #         # pdb.set_trace()
        #         # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
        #         if self.indice_memory[predicted_cate] == self.memory_size:
        #             if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
        #                 pass  ## the entropy of current test image is very large.
        #             else:
        #                 # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
        #                 _, indice = torch.sort(self.entropy_memory[predicted_cate])
        #                 to_replace_indice = indice[-1]  ## with max entropy, ascending.
        #                 self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
        #                 self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
        #         else:
        #             self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
        #             self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
        #             self.indice_memory[predicted_cate] += 1
        #     else:
        #         pass
        
        # activate_indicator = conf_in_vanilla < (1-self.thres) ## only store high confident samples into feature memory.
        # _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        # prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
        # for i in range(activate_indicator.size(0)):
        #     if activate_indicator[i].item():
        #         predicted_cate = pred_all[i].item() + class_num
        #         predicted_prob = prob_ood[i]
        #         current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
        #         # pdb.set_trace()
        #         # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
        #         if self.indice_memory[predicted_cate] == self.memory_size:
        #             if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
        #                 pass  ## the entropy of current test image is very large.
        #             else:
        #                 # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
        #                 _, indice = torch.sort(self.entropy_memory[predicted_cate])
        #                 to_replace_indice = indice[-1]  ## with max entropy, ascending.
        #                 self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
        #                 self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
        #         else:
        #             self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
        #             self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
        #             self.indice_memory[predicted_cate] += 1
        #     else:
        #         pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        sim = self.feat_memory @ image_features.t()  ## 11k*30*256
        #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        sim = torch.exp(-self.beta * (-sim + 1))  ## 
        # print(self.beta)
        # sim = torch.exp(-5.5 * (-sim + 1))
        ############################################################################################## the following command explode the memory.
        sa_text_features_list = []
        split_num = int(self.feat_memory.size(0) / 1000)
        for i in range(split_num):
            temp = sim[1000*i:1000*(i+1)].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*i:1000*(i+1)].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
            sa_text_features_split = temp.sum(2) ## 256*1k*512
            sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
            sa_text_features_list.append(sa_text_features_split)
        sa_text_features = torch.cat(sa_text_features_list, dim=1)
        # temp = sim.unsqueeze(0).transpose(0,-1) * self.feat_memory.unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
        # sa_text_features = temp.sum(2) ## 256*11k*512
        # sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        # pdb.set_trace()
        output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k

        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
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
        conf_in = scores.mean(dim=-1)
        # # ###########
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'sum':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'combine':
            conf = conf_in + conf_in_vanilla
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau



############################ test time adaptation with memory networks, modified from OneOodPromptDevelopPostprocessor 
### AdaNeg with adaptive gap. 
class TTAPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(TTAPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True ## reset after each dataset. 
        self.memory_size = self.args.memleng
        self.lambda_val = self.args.lambdaval
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
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

                    for i in range(image_features.size(0)):
                        all_feats[labels[i]].append(image_features[i].unsqueeze(0))
                        all_weights[labels[i]].append(weights[i].unsqueeze(0))

            for i in range(len(all_feats)):
                if len(all_feats[i]) != 0:
                    all_feats[i] = torch.cat(all_feats[i], dim=0)
                    all_weights[i] = torch.cat(all_weights[i], dim=0)
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

    def reset_memory(self):
        self.reset = True

    # Store (high confident) test features into the feature memory, and get new classifier by merging image features in the same memory slot.
    ### to do: multiple augmentations with filtering! 
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            ## 设置一个memory, 来缓存 positive | negative 的比例
            self.window_size = 10000
            self.recent_estimates = [] ## 0，1  ## 先在里面预存200 随机o 1
            self.estimated_positive_count = 100 
            self.estimated_negative_count = 100
            for i in range(self.estimated_positive_count): 
                self.recent_estimates.append(1)
                self.recent_estimates.append(0)

            self.reset=False
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_vanilla = torch.softmax(output_vanilla[:, :class_num], dim=1)
        # conf_in_vanilla = torch.sum(prob_vanilla[:, :class_num], dim=1)
        pos_logit = output_vanilla[:, :class_num] ## B*C
        neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
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
        activate_indicator_positive_vanilla = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.
        activate_indicator_negative_vanilla = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        for i in range(activate_indicator_positive_vanilla.size(0)):
            if activate_indicator_positive_vanilla[i].item():
                if len(self.recent_estimates) >= self.window_size:
                    oldest_estimate = self.recent_estimates.pop(0)
                    if oldest_estimate == 1:
                        self.estimated_positive_count -= 1
                    else:
                        self.estimated_negative_count -= 1
                self.recent_estimates.append(1)
                self.estimated_positive_count += 1


        for i in range(activate_indicator_negative_vanilla.size(0)):
            if activate_indicator_negative_vanilla[i].item():
                if len(self.recent_estimates) >= self.window_size:
                    oldest_estimate = self.recent_estimates.pop(0)
                    if oldest_estimate == 1:
                        self.estimated_positive_count -= 1
                    else:
                        self.estimated_negative_count -= 1
                self.recent_estimates.append(0)
                self.estimated_negative_count += 1

        #### use the estimated ratio to adjust the gap 
        estimated_ratio = self.estimated_positive_count / (self.estimated_positive_count + self.estimated_negative_count)  ## [0,1]
        ### 如果 estimated_ratio 远大于1， 则positive sample 很多，这样conf_in_vanilla 必须足够小才能放进negative memory (极端情况，不往negative memroy 添加样本)
        if estimated_ratio == 0.5:
            activate_indicator_positive = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.
            activate_indicator_negative = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        elif estimated_ratio > 0.5:  ## more ID than  OOD
            activate_indicator_positive = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.
            activate_indicator_negative = conf_in_vanilla < (self.thres - estimated_ratio * self.thres) ## only store high confident samples into feature memory.
        elif estimated_ratio < 0.5:  ## More OOD than ID.
            activate_indicator_positive = conf_in_vanilla > (self.thres + (1 - estimated_ratio) * (1-self.thres)) ## only store high confident samples into feature memory.
            activate_indicator_negative = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        # print(estimated_ratio)
        ### 如果 estimated_ratio 远小于1， negative sample 很多，这样conf_in_vanilla 必须足够大才能放进 positive memory (极端情况，不往positive memroy 添加样本)

        # pdb.set_trace()
        # threshold = self.thres
        # activate_indicator_positive = torch.randint(0, 2, (activate_indicator_positive.shape[0],), dtype=torch.bool)  ## for rebuttal analyses with high error of pseudo labels.
        _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        for i in range(activate_indicator_positive.size(0)):
            if activate_indicator_positive[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
        
        # activate_indicator_negative = ~activate_indicator_positive
        
        ####################################### in the above formulation, the default middle point is 0.5; thres - 0.5 is the unsed gap. 
        _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
        for i in range(activate_indicator_negative.size(0)):
            if activate_indicator_negative[i].item():
                predicted_cate = pred_all[i].item() + class_num
                predicted_prob = prob_ood[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        sim = self.feat_memory @ image_features.t()  ## 11k*30*256
        #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        sim = torch.exp(-self.beta * (-sim + 1))  ## 
        # print(self.beta)
        # sim = torch.exp(-5.5 * (-sim + 1))
        ############################################################################################## the following command explode the memory.
        if self.samada:
            sa_text_features_list = []
            split_num = int(self.feat_memory.size(0) / 1000.0)
            for i in range(split_num):
                temp = sim[1000*i:1000*(i+1)].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*i:1000*(i+1)].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                sa_text_features_split = temp.sum(2) ## 256*1k*512
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                sa_text_features_list.append(sa_text_features_split)
            if self.feat_memory.size(0) % 1000.0 > 0:
                ### processing the remaining one.
                temp = sim[1000*split_num:].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*split_num:].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                sa_text_features_split = temp.sum(2) ## 256*1k*512
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                sa_text_features_list.append(sa_text_features_split)
            sa_text_features = torch.cat(sa_text_features_list, dim=1)
        else:
            sa_text_features = self.feat_memory.mean(1)
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.

        # temp = sim.unsqueeze(0).transpose(0,-1) * self.feat_memory.unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
        # sa_text_features = temp.sum(2) ## 256*11k*512
        # sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        # pdb.set_trace()
        output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        # pdb.set_trace()
        prob_tta = torch.softmax(output[:, :class_num], dim=1)
        # prob_all = prob_vanilla + prob_tta * self.lambda_val
        prob_all = prob_vanilla + prob_tta
        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(prob_all[:, :class_num], dim=1)

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
        conf_in = scores.mean(dim=-1)
        # # ###########
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'adaonly':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'vanillaonly':
            conf = conf_in_vanilla
        elif self.in_score == 'combine':
            # conf = conf_in + conf_in_vanilla * self.lambda_val
            conf = conf_in + conf_in_vanilla
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau




## this is the adopted one in NIPS submission.
class TTAPromptPostprocessor_noadagap(BasePostprocessor):
    def __init__(self, config):
        super(TTAPromptPostprocessor_noadagap, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True ## reset after each dataset. 
        self.memory_size = self.args.memleng
        self.lambda_val = self.args.lambdaval
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
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

                    for i in range(image_features.size(0)):
                        all_feats[labels[i]].append(image_features[i].unsqueeze(0))
                        all_weights[labels[i]].append(weights[i].unsqueeze(0))

            for i in range(len(all_feats)):
                if len(all_feats[i]) != 0:
                    all_feats[i] = torch.cat(all_feats[i], dim=0)
                    all_weights[i] = torch.cat(all_weights[i], dim=0)
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

    def reset_memory(self):
        self.reset = True

    # Store (high confident) test features into the feature memory, and get new classifier by merging image features in the same memory slot.
    ### to do: multiple augmentations with filtering! 
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            self.reset=False
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_vanilla = torch.softmax(output_vanilla, dim=1)
        # conf_in_vanilla = torch.sum(prob_vanilla[:, :class_num], dim=1)
        pos_logit = output_vanilla[:, :class_num] ## B*C
        neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
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

        # pdb.set_trace()
        # threshold = self.thres
        activate_indicator = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.

        # activate_indicator = torch.randint(0, 2, (activate_indicator.shape[0],), dtype=torch.bool)  ## for rebuttal analyses with high error of pseudo labels.
        _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
        
        # activate_indicator = ~activate_indicator
        activate_indicator = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        # pdb.set_trace()
        ####################################### in the above formulation, the default middle point is 0.5; thres - 0.5 is the unsed gap. 
        _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item() + class_num
                predicted_prob = prob_ood[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        sim = self.feat_memory @ image_features.t()  ## 11k*30*256
        #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        sim = torch.exp(-self.beta * (-sim + 1))  ## 
        # print(self.beta)
        # sim = torch.exp(-5.5 * (-sim + 1))
        ############################################################################################## the following command explode the memory.
        if self.samada:
            sa_text_features_list = []
            split_num = int(self.feat_memory.size(0) / 1000.0)
            for i in range(split_num):
                temp = sim[1000*i:1000*(i+1)].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*i:1000*(i+1)].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                sa_text_features_split = temp.sum(2) ## 256*1k*512
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                sa_text_features_list.append(sa_text_features_split)
            if self.feat_memory.size(0) % 1000.0 > 0:
                ### processing the remaining one.
                temp = sim[1000*split_num:].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*split_num:].unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
                sa_text_features_split = temp.sum(2) ## 256*1k*512
                sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)  ## renorm.
                sa_text_features_list.append(sa_text_features_split)
            sa_text_features = torch.cat(sa_text_features_list, dim=1)
        else:
            sa_text_features = self.feat_memory.mean(1)
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.

        # temp = sim.unsqueeze(0).transpose(0,-1) * self.feat_memory.unsqueeze(0) ## 256*11k*30*1 * 1*11K*30*512 -->256, 11k, 30, 512
        # sa_text_features = temp.sum(2) ## 256*11k*512
        # sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        # pdb.set_trace()
        output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        # pdb.set_trace()
        prob_tta = torch.softmax(output, dim=1)
        prob_all = prob_vanilla + prob_tta * self.lambda_val
        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(prob_all[:, :class_num], dim=1)

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
        conf_in = scores.mean(dim=-1)
        # # ###########
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'adaonly':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'vanillaonly':
            conf = conf_in_vanilla
        elif self.in_score == 'combine':
            conf = conf_in + conf_in_vanilla * self.lambda_val
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau



def extract_most_similar_features(image_features, image_features_local):
    """
    Extracts the most similar feature (based on cosine similarity) in image_features_local
    for each feature in image_features.

    Parameters:
    - image_features (torch.Tensor): a tensor of shape (B, D) where B is the batch size and D is the feature dimension.
    - image_features_local (torch.Tensor): a tensor of shape (B, L, D) where L is the local feature count.

    Returns:
    - most_similar_features (torch.Tensor): a tensor of shape (B, D) containing the most similar local feature
      for each feature in image_features.
    """
    # Normalize the features to compute cosine similarity
    image_features_norm = F.normalize(image_features, p=2, dim=1)
    image_features_local_norm = F.normalize(image_features_local, p=2, dim=2)

    # Compute the cosine similarity
    # similarity_scores: (B, L)
    similarity_scores = torch.matmul(image_features_norm.unsqueeze(1), image_features_local_norm.transpose(2, 1)).squeeze(1)

    # Find the index of the most similar local features
    # max_indices: (B,)
    _, max_indices = torch.max(similarity_scores, dim=1)

    # Gather the most similar features using the indices
    # We use arange and indexing to select the appropriate features
    batch_indices = torch.arange(image_features_local.size(0)).to(image_features_local.device)
    most_similar_features = image_features_local[batch_indices, max_indices]

    return most_similar_features

############################ give up!  I have tried to use local feature in many ways, single position, various averaging, most similar to global feature, but the results are bad. 
###########################
class TTAPromptLocalfeatPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(TTAPromptLocalfeatPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.reset = True ## reset after each dataset. 
        self.memory_size = 30
        self.thres = self.args.thres
        self.lindice = int(self.args.localindice)
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass
        ### get the image feature of each classes, construct the image feature classifier/memory.
        # net.eval()
        # # net.text_features 11k*512, if empty, fill with net.
        # out_dim = net.n_output
        # if self.setup_flag:
        #     # estimate class mean from training set
        #     # net.text_features.t() ## N*512
        #     # net.logit_scale
        #     # with torch.no_grad():
        #     #     output_text = net.logit_scale * net.text_features.t() @ net.text_features # class_num * class_num
        #     #     output_text = torch.softmax(output_text, dim=1) 
        #     # all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
        #     # self.text_idscore_cache = all_weights_text ## 11k

        #     with torch.no_grad():
        #         output_text = net.logit_scale * net.text_features_unselected.t() @ net.text_features # class_num * class_num
        #         output_text = torch.softmax(output_text, dim=1) 
        #     all_weights_text = output_text[:, :1000].sum(1) # prob of ID 
        #     self.text_idscore_unselected_cache = all_weights_text ## 11k

        #     print('\n geting image features from (generated) training set...')
        #     all_feats = []
        #     all_weights = []
        #     for i in range(out_dim):
        #         all_feats.append([])  ## category-wise feature list
        #         all_weights.append([])
            
        #     ############################## init with text feature.
        #     # # pdb.set_trace()
        #     # for i in range(out_dim):
        #     #     all_feats[i].append(net.text_features.t()[i].unsqueeze(0))  ## category-wise feature list

        #     with torch.no_grad():
        #         for batch in tqdm(id_loader_dict['train'],
        #                           desc='Setup: ',
        #                           position=0,
        #                           leave=True):
        #             data, labels = batch['data'].cuda(), batch['label']
        #             image_features, text_features, logit_scale = net(data, return_feat=True)
        #             ####### weighting image features according to the classification probability.
        #             output = logit_scale * image_features @ text_features.t() # batch * class, using the classification score as weights. 
        #             output_prob = torch.softmax(output, dim=1) ## use category prob. as weights or ID prob. as weights?
        #             indice = torch.arange(output.size(0))
        #             # pdb.set_trace()
        #             ########################## category prob level weights.
        #             # weights = output_prob[indice, labels]
        #             ########################## id/ood level sample weights
        #             weights = output_prob[:, :1000].sum(1)

        #             for i in range(image_features.size(0)):
        #                 all_feats[labels[i]].append(image_features[i].unsqueeze(0))
        #                 all_weights[labels[i]].append(weights[i].unsqueeze(0))

        #     for i in range(len(all_feats)):
        #         if len(all_feats[i]) != 0:
        #             all_feats[i] = torch.cat(all_feats[i], dim=0)
        #             all_weights[i] = torch.cat(all_weights[i], dim=0)
        #     all_feats = [x for x in all_feats if isinstance(x, torch.Tensor)]
        #     all_weights = [x for x in all_weights if isinstance(x, torch.Tensor)]

        #     all_feats_id = torch.cat(all_feats[:1000], dim=0) ## 11k * 512
        #     all_weights_id = torch.cat(all_weights[:1000], dim=0) ## 11k * 512
        #     # pdb.set_trace()
        #     all_feats_ood = torch.cat(all_feats[1000:], dim=0) ## 11k * 512
        #     all_weights_ood = torch.cat(all_weights[1000:], dim=0) ## 11k * 512
        #     self.image_classifier = all_feats_id

        #     self.image_feat_cache_id = all_feats_id
        #     self.image_idscore_cache_id = all_weights_id

        #     self.image_feat_cache_ood = all_feats_ood
        #     self.image_idscore_cache_ood = all_weights_ood

        #     self.image_feat_cache = torch.cat((self.image_feat_cache_id, self.image_feat_cache_ood), dim=0)
        #     self.image_idscore_cache = torch.cat(( self.image_idscore_cache_id,  self.image_idscore_cache_ood), dim=0)
        # else:
        #     pass

    def reset_memory(self):
        self.reset = True

    # Store (high confident) test features into the feature memory, and get new classifier by merging image features in the same memory slot.
    ### to do: multiple augmentations with filtering! 
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features_all, text_features, logit_scale = net(data, return_feat=True)  ## image_features_all: B*(1+wh)*D;   text_features: (C_all = C_in + C_out) * D
        image_features = image_features_all[:,0,:]
        # pdb.set_trace()
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            self.reset=False
        ## image_features: 256*512, 
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        # output_local = logit_scale * image_features_all[:,1:,:] @ text_features.t() # B*wh*C_all
        # prob_local = torch.softmax(output_local, dim=-1)
        # conf_in_local =  torch.sum(prob_local[:, :, :class_num], dim=-1) ## B*wh
        # value_conf_in_local, indice_conf_in_local = torch.sort(conf_in_local, dim=-1, descending=True) ## ascending order
        # # value_max, indice_max = value_conf_in_local[:, 0], indice_conf_in_local[:, 0]
        # ########################### here try to use different local regions; use the max response may introduce noises, maybe we should use top 1, 3, 5, 10, 30, 50, 80, 100, 196
        # value_max, indice_max = value_conf_in_local[:, :self.lindice].mean(dim=1), indice_conf_in_local[:, 0]
        most_similar_features = extract_most_similar_features(image_features_all[:,0,:], image_features_all[:,1:,:])
        output_local = logit_scale * most_similar_features @ text_features.t() # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_local = torch.softmax(output_local, dim=1)
        value_max = torch.sum(prob_local[:, :class_num], dim=1)
        ############# find the maximum response for ID classes.


        output_vanilla = logit_scale * image_features_all[:,0,:] @ text_features.t() # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_vanilla = torch.softmax(output_vanilla, dim=1)
        conf_in_vanilla = torch.sum(prob_vanilla[:, :class_num], dim=1)
        # pdb.set_trace()
        # threshold = self.thres
        activate_indicator = conf_in_vanilla > self.thres ## only store high confident samples into feature memory.
        _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
        
        activate_indicator = conf_in_vanilla < (1-self.thres) ## only store high confident samples into feature memory.
        _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item() + class_num
                predicted_prob = prob_ood[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        sim = self.feat_memory @ image_features.t()  ## 11k*7*256
        #### may combine with temp and softmax !!!!!!!!!!!!!!!!!!!!!!!!!!, here use cose sim directly. here with negative values.
        sim = torch.exp(-self.beta * (-sim + 1))
        # print(self.beta)
        # sim = torch.exp(-5.5 * (-sim + 1))
        temp = sim.unsqueeze(0).transpose(0,-1) * self.feat_memory.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
        sa_text_features = temp.sum(2) ## 256*11k*512
        sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
        output = logit_scale * (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k

        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(output[:, :class_num], dim=1)

        # ###########
        score = torch.softmax(output, dim=1)
        conf_in = torch.sum(score[:, :class_num], dim=1)
        conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'sum':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'combine':
            conf = conf_in + conf_in_vanilla
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        elif self.in_score == 'localonly':
            conf = value_max
        elif self.in_score == 'localglobal_add':
            conf = conf_in + value_max
        elif self.in_score == 'localglobal_multiply':
            conf = conf_in * value_max
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
