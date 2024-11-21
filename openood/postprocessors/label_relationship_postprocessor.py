from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import openood.utils.comm as comm

from .base_postprocessor import BasePostprocessor
import pdb


class LabelRelationPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(LabelRelationPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = float(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = True
        self.proj_flag = False
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
        self.reset = True
    
    ###### 统计每个类sample score (MCM and Neglabel) 的均值和方差。使用GT 的label 还是predict label 呢？
    ## 也需要统计OOD sample 的分类别的均值和方差 (这里只能用predict label了)。 
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ### 
        net.eval()

        output_text = net.logit_scale * net.text_features.t() @ net.text_features # batch * class.
        score_output_text = torch.softmax(output_text/self.beta , dim=1)
        self.score_prior = score_output_text[:1000, :1000].sum(1)


        class_num = net.n_cls
        id_loader = id_loader_dict['test']
        ood_loader = ood_loader_dict['farood']['places']  ##['inaturalist', 'sun', 'places', 'dtd']  ## nearood: ssb_hard; ninco
        # pdb.set_trace()
        id_mcm_score_gt_list = []
        for i in range(class_num):
            id_mcm_score_gt_list.append([])
        id_neglabel_score_gt_list = []
        for i in range(class_num):
            id_neglabel_score_gt_list.append([])
        id_mcm_score_list = []
        for i in range(class_num):
            id_mcm_score_list.append([])
        id_neglabel_score_list = []
        for i in range(class_num):
            id_neglabel_score_list.append([])
        ood_mcm_score_list = []
        for i in range(class_num):
            ood_mcm_score_list.append([])
        ood_neglabel_score_list = []
        for i in range(class_num):
            ood_neglabel_score_list.append([])

        for batch in tqdm(id_loader, disable=False):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            image_features, text_features, logit_scale = net(data, return_feat=True)
            # image_classifier = self.image_classifier
    

            output = logit_scale * image_features @ text_features.t() # batch * class.
            # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
            _, pred_in = torch.max(output[:, :class_num], dim=1)

            ############################### only score in. here is a 
            output_only_in = output[:, :class_num]
            score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
            mcm_score, pred_only_in = torch.max(score_only_in, dim=1)
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

            neglabel_score = conf_in  ## = 1-conf_out
            # # max in prob - max out prob
            # if self.in_score == 'oodscore' or self.in_score == 'sum':
            #     neglabel_score = conf_in  ## = 1-conf_out
            # else:
            #     raise NotImplementedError
            if torch.isnan(neglabel_score).any():
                pdb.set_trace()
            for i in range(len(label)):
                id_mcm_score_gt_list[label[i]].append(mcm_score[i].cpu().unsqueeze(0))
                id_neglabel_score_gt_list[label[i]].append(neglabel_score[i].cpu().unsqueeze(0))

                id_mcm_score_list[pred_in[i]].append(mcm_score[i].cpu().unsqueeze(0))
                id_neglabel_score_list[pred_in[i]].append(neglabel_score[i].cpu().unsqueeze(0))

        id_mcm_score_gt_std = []
        id_neglabel_score_gt_std = []
        id_mcm_score_std = []
        id_neglabel_score_std = []
        id_prection_count = []
        for i in range(class_num):
            id_mcm_score_gt_list[i] = torch.cat(id_mcm_score_gt_list[i], dim=0)
            id_mcm_score_gt_std.append(torch.std(id_mcm_score_gt_list[i]).item())
            id_mcm_score_gt_list[i] = torch.mean(id_mcm_score_gt_list[i]).item()

            id_neglabel_score_gt_list[i] = torch.cat(id_neglabel_score_gt_list[i], dim=0)
            id_neglabel_score_gt_std.append(torch.std(id_neglabel_score_gt_list[i]).item())
            id_neglabel_score_gt_list[i] = torch.mean(id_neglabel_score_gt_list[i]).item()

            if len(id_mcm_score_list[i]) != 0:
                id_prection_count.append(len(id_mcm_score_list[i]))  ## count distribution of the prediction.
                id_mcm_score_list[i] = torch.cat(id_mcm_score_list[i], dim=0)  ### 有可能是空的，并不一定要assign top1 label; 比如top1 那个类很多都是0.9+, 一个0.6的score 可能并不是这一类的，要看top2,3. 
                # 现在的问题是，他就是有空的，那么如果是空的，就保持不变 []; soft加权来指定他的类别。
                id_mcm_score_std.append(torch.std(id_mcm_score_list[i]).item())
                id_mcm_score_list[i] = torch.mean(id_mcm_score_list[i]).item()

            else:
                id_prection_count.append(0)
                id_mcm_score_list[i] = 0.5
                id_mcm_score_std.append(0)

            if len(id_neglabel_score_list[i]) != 0:
                id_neglabel_score_list[i] = torch.cat(id_neglabel_score_list[i], dim=0)
                id_neglabel_score_std.append(torch.std(id_neglabel_score_list[i]).item())
                id_neglabel_score_list[i] = torch.mean(id_neglabel_score_list[i]).item()

            else:
                id_neglabel_score_list[i] = 0.5
                id_neglabel_score_std.append(0)
        # pdb.set_trace()
        #################### for ood 
        for batch in tqdm(ood_loader,
                          disable=False):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            image_features, text_features, logit_scale = net(data, return_feat=True)
            # image_classifier = self.image_classifier

            output = logit_scale * image_features @ text_features.t() # batch * class.
            # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
            _, pred_in = torch.max(output[:, :class_num], dim=1)

            ############################### only score in. here is a 
            output_only_in = output[:, :class_num]
            score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
            mcm_score, pred_only_in = torch.max(score_only_in, dim=1)
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

            neglabel_score = conf_in
            # # max in prob - max out prob
            # if self.in_score == 'oodscore' or self.in_score == 'sum':
            #     neglabel_score = conf_in  ## = 1-conf_out
            # else:
            #     raise NotImplementedError
            if torch.isnan(neglabel_score).any():
                pdb.set_trace()
            for i in range(len(label)):
                ood_mcm_score_list[pred_in[i]].append(mcm_score[i].cpu().unsqueeze(0))
                ood_neglabel_score_list[pred_in[i]].append(neglabel_score[i].cpu().unsqueeze(0))    


        ood_mcm_score_std = []
        ood_neglabel_score_std = []
        ood_prection_count = []
        for i in range(class_num):
            if len(ood_mcm_score_list[i]) != 0:
                ood_prection_count.append(len(ood_mcm_score_list[i]))
                ood_mcm_score_list[i] = torch.cat(ood_mcm_score_list[i], dim=0)  ### 有可能是空的，并不一定要assign top1 label; 比如top1 那个类很多都是0.9+, 一个0.6的score 可能并不是这一类的，要看top2,3. 
                # 现在的问题是，他就是有空的，那么如果是空的，就保持不变 []; soft加权来指定他的类别。
                ood_mcm_score_std.append(torch.std(ood_mcm_score_list[i]).item())
                ood_mcm_score_list[i] = torch.mean(ood_mcm_score_list[i]).item()

            else:
                ood_prection_count.append(0)
                ood_mcm_score_list[i] = 0.5
                ood_mcm_score_std.append(0)
            if len(ood_neglabel_score_list[i]) != 0:
                ood_neglabel_score_list[i] = torch.cat(ood_neglabel_score_list[i], dim=0)
                ood_neglabel_score_std.append(torch.std(ood_neglabel_score_list[i]).item())
                ood_neglabel_score_list[i] = torch.mean(ood_neglabel_score_list[i]).item()

            else:
                ood_neglabel_score_list[i] = 0.5
                ood_neglabel_score_std.append(0)
        self.id_neglabel_score_gt_list = id_neglabel_score_gt_list  ### 先用GT式一下，如果有效，那么GT应该也是有效的。
        self.id_neglabel_score_gt_std = id_neglabel_score_gt_std

    #     # 把ID ood 的均值方差画到一张图上，来facilitate 设计category wise threshold. 

    def reset_memory(self):
        self.reset = True



    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        if self.reset:
            self.class_count = torch.ones(net.n_cls) * self.beta
            self.mean_value = torch.zeros(net.n_cls)
            self.value_count = torch.ones(net.n_cls) * 3
            self.reset = False
        image_features, text_features, logit_scale = net(data, return_feat=True)
        # image_classifier = self.image_classifier

        output = logit_scale * image_features @ text_features.t() # batch * class.
        # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(output[:, :class_num], dim=1) 

        normalized_count = self.class_count.float() / self.class_count.sum() ### 
        # pdb.set_trace()
        ############################### only score in. here is a 
        # output_only_in = output[:, :class_num]
        # score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        # mcm_score, pred_only_in = torch.max(score_only_in, dim=1)
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
        conf_in = scores.mean(dim=-1).float() 
        ### 先算score, 在更新count. 
        for i in range(len(pred_in)):
            self.class_count[pred_in[i].item()] += 1
        # pdb.set_trace()

 

        # pdb.set_trace()
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'sum':
            conf_in = conf_in  ## = 1-conf_out
        elif self.in_score == 'div_mean':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / self.id_neglabel_score_gt_list[pred_in[i]] ## score mean for each class, class wise rescaling. 统计均值和方差，然后建模成高斯分布。减均值除以标准差
        elif self.in_score == 'div_score_prior':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / self.score_prior[pred_in[i]] 
        elif self.in_score == 'jian_mean':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] - self.id_neglabel_score_gt_list[pred_in[i]] 
        elif self.in_score == 'jian_mean_div_std':
            for i in range(len(conf_in)):
                conf_in[i] = (conf_in[i] - self.id_neglabel_score_gt_list[pred_in[i]]) /  self.id_neglabel_score_gt_std[pred_in[i]]
        elif self.in_score == 'div_count':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / normalized_count[pred_in[i]] / net.n_cls ### 错了，normalized_count 要根据label 找。
        else:
            raise NotImplementedError
        if torch.isnan(conf_in).any():
            pdb.set_trace()

        return pred_in, conf_in

    # def inference(self,
    #               net: nn.Module,
    #               data_loader: DataLoader,
    #               progress: bool = True):
    #     pred_list, conf_list, label_list = [], [], []
    #     for batch in tqdm(data_loader,
    #                       disable=not progress or not comm.is_main_process()):
 
    #         data = batch['data'].cuda()
    #         label = batch['label'].cuda()
    #         pdb.set_trace()
    #         pred, conf = self.postprocess(net, data)

    #         pred_list.append(pred.cpu())
    #         conf_list.append(conf.cpu())
    #         label_list.append(label.cpu())

    #     # convert values into numpy array
    #     pred_list = torch.cat(pred_list).numpy().astype(int)
    #     conf_list = torch.cat(conf_list).numpy()
    #     label_list = torch.cat(label_list).numpy().astype(int)

    #     return pred_list, conf_list, label_list

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

# 我也尝试了，对于每个样本选出最近的1000 个negative labels, 这样效果也不好。
# vanilla label relation
# class LabelRelationPostprocessor(BasePostprocessor):
#     def __init__(self, config):
#         super(LabelRelationPostprocessor, self).__init__(config)
#         self.args = self.config.postprocessor.postprocessor_args
#         self.tau = self.args.tau
#         self.beta = float(self.args.beta)
#         self.args_dict = self.config.postprocessor.postprocessor_sweep
#         self.in_score = self.args.in_score # sum | max
#         self.setup_flag = True
#         self.proj_flag = False
#         self.group_num = self.args.group_num
#         self.random_permute = self.args.random_permute
    
#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         ### get the image feature of each classes, construct the image feature classifier/memory.
#         net.eval()
#         # net.text_features 11k*512, if empty, fill with net.
#         out_dim = net.n_output
#         class_num = net.n_cls
#         pos_neg_text_features = net.text_features.t() ## class * dim
#         # all_text_features = net.text_features_all.t() ## class * dim
#         all_text_features = net.text_features.t() ## class * dim

#         # I want to use the NegLabel score as the similarity to ID domain. But it fails. 可能因为中间的点的similarity 不可靠。
#         # if self.setup_flag:
#         #     similarity = all_text_features @ pos_neg_text_features.t() ## all * ID_Negative class
#         #     # pdb.set_trace()
#         #     self.relation = (similarity * net.logit_scale).softmax(dim=-1)[:,:class_num].sum(-1) 
#         #     ## adopt the Neglabel score as the relation; 最后的image score 其实是相似的text score 的 加权和。
#         #     self.setup_flag = False
#         # else:
#         #     pass

#         #################################### only use the ID class to calculate the relation. 
#         pos_text_features = pos_neg_text_features[:class_num, :] ## class * dim
#         # pdb.set_trace()
#         if self.setup_flag:
#             similarity = pos_neg_text_features @ pos_text_features.t() ## all * ID class
#             self.relation, _ = similarity.max(1) ## max similartiy to one point as the relation, similar results to Neglabel.
#             # self.relation = similarity.mean(1) ## mean similarity as the relation, very bad results.
#             # self.relation, _ = (similarity * net.logit_scale).softmax(dim=-1).max(1) ## adopt the MCM score as the relation, worse than Neglabel.
#             self.setup_flag = False
#         else:
#             pass

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         net.eval()
#         class_num = net.n_cls
#         # pdb.set_trace()
#         image_features, text_features, logit_scale = net(data, return_feat=True)
#         # image_classifier = self.image_classifier

#         ## image_features: 256*512, 
#         ## text_features: 11k*7*512
#         ## extract sample adaptative classifier via attention,
#         output = logit_scale * image_features @ text_features.t() # batch * class.
#         # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
#         _, pred_in = torch.max(output[:, :class_num], dim=1)

#         # output_all_text = logit_scale * image_features @ net.text_features_all # batch * class.
#         output_all_text = logit_scale * image_features @ net.text_features # batch * class.

#         # print('pay attention, no scale is applied here.')
#         pos_logit = output_all_text[:, :class_num] ## B*C
#         neg_logit = output_all_text[:, class_num:] ## B*total_neg_num

#         pos_relation = self.relation[:class_num] ## 1k
#         neg_relation = self.relation[class_num:] ## 70k

#         drop = neg_logit.size(1) % self.group_num
#         if drop > 0:
#             neg_logit = neg_logit[:, :-drop]
#             neg_relation = neg_relation[:-drop]

#         if self.random_permute:
#             # print('use random permute')
#             SEED=0
#             torch.manual_seed(SEED)
#             torch.cuda.manual_seed(SEED)
#             idx = torch.randperm(neg_logit.shape[1]).to(output.device)
#             neg_logit = neg_logit.T ## total_neg_num*B
#             # pdb.set_trace()
#             neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
#             neg_relation = neg_relation[idx].reshape(self.group_num, -1).contiguous()
#         else:
#             neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
#             neg_relation = neg_relation.reshape(self.group_num, -1).contiguous()
#         scores = []
#         # pdb.set_trace()
#         for i in range(self.group_num):
#             full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
#             # full_prob = full_sim ## 直接用full sim 来做加权，导致正负比较难拉开，结果不好。

#             # pdb.set_trace()
#             # ## exp 也不太好控制, 因为原本的不同text 间的差距比较小；可以考虑先缩放到[-1,1] ,然后再用exp, 就可以把差距拉开了
#             # full_sim_rescale = full_sim / logit_scale
#             # max_value, _ = full_sim_rescale.max(0) ## 11k
#             # min_value, _ = full_sim_rescale.min(0) ## 11k
#             # full_sim_rescale = (full_sim_rescale - min_value.view(1, -1)) / (max_value - min_value).view(1, -1)
#             # full_prob = torch.exp(-self.beta * (1-full_sim_rescale)) 

#             full_prob = full_sim.softmax(dim=-1) ## N * class_num ### 这里是一个soft weight, 是比较好的。

#             # # 改成用topk 的: 结果并不好，比不过softmax; 放弃
#             # K = self.beta
#             # topk_values, topk_indices = torch.topk(full_sim, K, dim=1)
#             # scores_can = torch.arange(K, 0, -1, dtype=full_sim.dtype) / K
#             # full_prob = torch.zeros_like(full_sim)
#             # scores_can = scores_can.to(full_sim.device)
#             # full_prob = full_prob.to(full_sim.device)
#             # full_prob.scatter_(1, topk_indices, scores_can.expand(full_sim.size(0), -1))

#             # pdb.set_trace()
#             # full_relation = torch.cat([pos_relation, neg_relation[i]]).view(1, -1)
#             # pos_score = (full_prob * full_relation).mean(-1)
#             # scores.append(pos_score.unsqueeze(-1))

#             # full_relation = torch.cat([pos_relation, neg_relation[i]]).view(1, -1)
#             # pos_score = (full_prob * full_relation)[:, :pos_logit.shape[1]].sum(dim=-1)

#             pos_score = full_prob[:, :pos_logit.shape[1]].sum(dim=-1)
#             scores.append(pos_score.unsqueeze(-1))

#             scores.append(pos_score.unsqueeze(-1))
#         scores = torch.cat(scores, dim=-1)
#         conf_in = scores.mean(dim=-1)

#         # max in prob - max out prob
#         if self.in_score == 'oodscore' or self.in_score == 'sum':
#             conf = conf_in  ## 
#         else:
#             raise NotImplementedError
#         if torch.isnan(conf).any():
#             pdb.set_trace()

#         return pred_in, conf

#     def set_hyperparam(self, hyperparam: list):
#         self.tau = hyperparam[0]

#     def get_hyperparam(self):
#         return self.tau

 
