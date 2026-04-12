## I put the raw code here for future reference. 
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import openood.utils.comm as comm
import math

from .base_postprocessor import BasePostprocessor
import pdb


def compute_os_variance_tensor(os_tensor, th):
    """
    Calculate weighted variance for PyTorch tensors

    Args:
        os_tensor: (torch.Tensor) OOD scores tensor
        th: (float) threshold value

    Returns:
        (torch.Tensor) weighted variance
    """
    device = os_tensor.device
    
    # 二值化mask
    mask = (os_tensor >= th).float()
    
    # 计算权重
    n_pixels = os_tensor.numel()
    n_pixels1 = torch.sum(mask)
    weight1 = n_pixels1 / n_pixels
    weight0 = 1 - weight1
    
    # 处理空类别
    if weight1 == 0 or weight0 == 0:
        return torch.tensor(float('inf'), device=device)
    
    # 获取两类数值
    class1 = os_tensor[mask.bool()]
    class0 = os_tensor[~mask.bool()]
    
    # 计算方差
    var0 = torch.var(class0, unbiased=False) if class0.numel() > 0 else torch.tensor(0.0, device=device)
    var1 = torch.var(class1, unbiased=False) if class1.numel() > 0 else torch.tensor(0.0, device=device)
    
    return weight0 * var0 + weight1 * var1

def find_best_threshold(os_training_queue):
    """
    主函数：在GPU/CPU上寻找最优阈值
    
    Args:
        os_training_queue: (torch.Tensor) 输入分数张量
    
    Returns:
        best_threshold: (float) 最优阈值
    """
    # 生成阈值范围（自动匹配设备）
    threshold_range = torch.arange(0, 1, 0.01, device=os_training_queue.device)
    
    # 计算各阈值对应的指标
    criterias = torch.stack([compute_os_variance_tensor(os_training_queue, th) for th in threshold_range])
    
    # 找到所有最小方差的位置
    min_val = torch.min(criterias)
    mask = (criterias == min_val)
    candidate_indices = torch.where(mask)[0]

    # 从候选中选择中间位置的阈值
    if len(candidate_indices) == 0:
        return threshold_range[torch.argmin(criterias)].item()  # 回退机制

    # 直接取中间索引
    mid_index = len(candidate_indices) // 2
    best_threshold = threshold_range[candidate_indices[mid_index]]

    return best_threshold.item()


def kmeans_l2_normalized(x, n_clusters, max_iter=100, tol=1e-4):
    """
    L2归一化数据的高效K-Means（基于余弦相似度）
    Args:
        x: 输入数据（已L2归一化），形状 [N, D]
        n_clusters: 聚类簇数
        max_iter: 最大迭代次数
        tol: 中心点变化容忍度
        device: 计算设备（'cuda' 或 'cpu'）
    Returns:
        centroids: 聚类中心 [K, D]
        labels: 样本标签 [N]
    """
    N, D = x.shape
    # 初始化中心点：随机选择样本
    indices = torch.randperm(N)[:n_clusters].to(x.device)
    centroids = x[indices]

    for _ in range(max_iter):
        # 计算余弦相似度（等价于点积）[N, K]
        similarities = torch.mm(x, centroids.t())  # 关键优化：矩阵乘法代替距离计算

        # 分配标签：取最大相似度 [N]
        labels = torch.argmax(similarities, dim=1)

        # 更新中心点
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                # 对簇内样本求均值（保持L2归一化）
                new_centroids[k] = x[mask].mean(dim=0)
                new_centroids[k] /= torch.norm(new_centroids[k])  # 重新归一化
            else:
                # 处理空簇：随机选择一个样本
                new_centroids[k] = x[torch.randint(0, N, (1,)).to(x.device)]

        # 检查收敛条件
        if torch.norm(centroids - new_centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels

def update_queue(to_add, queue, max_len=1000, init=False):
    if init:
        queue = queue
    else:
        queue = torch.cat((queue, to_add), dim=0)
    if queue.size(0) > max_len:
        queue = queue[-max_len:, :]
    return queue

def activation_aware_score(output, id_num, ood_num, step, weight_by_activation_score=False):
    if isinstance(weight_by_activation_score, torch.Tensor) and weight_by_activation_score.numel() > 1:
        use_weight = True
    elif bool(weight_by_activation_score):
        use_weight = True
    else:
        use_weight = False
    if use_weight:
        if step == 0:
            full_sim = output.softmax(dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
        elif step == 1:
            full_sim = (output * weight_by_activation_score.unsqueeze(0)).softmax(dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
        elif step == 2:
            # S_{aa}^{ew2}(\vv): sum over i=1..C, denominator contains exp over ID + weighted exp over negative, as in formula
            weighted_logits = output.clone()
            weighted_logits = weighted_logits + weight_by_activation_score.log().unsqueeze(0)
            full_sim = torch.softmax(weighted_logits, dim=-1)
            conf_in = full_sim[:, :id_num].sum(dim=-1)
        ########### 这个setting 下，step 是新的含义：0代表不加权，1 代表内部，2代表外部
    elif step != 0:   ####### gamma =5 确实结果变好了，比0好，说明按照activation score 来加权确实结果更好！结果稳定了很多。
        softmax_sums = []
        step = int(step)
        for i in range(id_num, id_num + ood_num, step):  # 列索引从 1000 到 1999
            ## 下面方法每一段包含了前一段， 相当于前面的算了比较多次，权重大一些; 这样缓解了一些情况：step 1,2,10 比step 0效果好，但是还是会随着neg number 数量增加而变差; 一定要把activation score 的权重算进来。
            softmax_output = output[:,:i+step].softmax(dim=-1) 
            sum_score = softmax_output[:, :id_num].sum(dim=-1)  # 即使没做加权，结果也好了一些; 就用这个了。

            ##### 还是要做加权，试试乘以新加部分的权重!! 加权一直做不出来。
            ### 还是不行，乘上之后加过下降了！ 而且并没有想象中的对 large negative number 的稳定性， number 多的时候结果还是下降了不少； 下面的实验不再继续尝试。
            # score_weight = selected_combined_score[i-class_num:i-class_num+step].sum()
            # sum_score = sum_score * score_weight  
    
            #  如果每一段不包含前面一段呢？手动对每个进行加权？用 activation score 进行加权？
            # softmax_output = torch.cat((output[:,:class_num], output[:,i:i+step]),dim=-1).softmax(dim=-1)  # 对step 异常敏感; 即使某个step 有效也不行因为
            ############# 一定要改成，对后面的negative label apply small weights; 这样negative label 再多，其对结果的影响也会小一些。
            # sum_score = softmax_output[:, :class_num].sum(dim=-1)  # 分group 算
            # sum_score = sum_score * selected_combined_score[i-class_num:i-class_num+step].sum()  ### 结果对step 非常敏感，不能用。
            # print(selected_combined_score[class_num:i-class_num+step].sum())
            # 根绝neg score 做加权。这样 重要的negative label 可以得到更高的权重。引入更多的negative label 的影响就小了，就对neglabel number 不敏感了。否则还是敏感。
            softmax_sums.append(sum_score)
        # 将 softmax_sums 堆叠为一个张量：batch_size x 1000
        softmax_sums = torch.stack(softmax_sums, dim=-1)
        # 对列（1000 次 softmax 的结果）求均值作为最终结果
        conf_in = softmax_sums.mean(dim=-1)  # 最终为 batch_size 的向量
    else:
        full_sim = output.softmax(dim=-1)
        conf_in = full_sim[:, :id_num].sum(dim=-1)
    return conf_in

class ActivatedNegPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ActivatedNegPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = True
        self.proj_flag = False
        self.alpha = self.args.alpha
        self.gamma = self.args.gamma
        self.group_num = self.args.group_num
        self.group_size = self.args.group_size
        self.random_permute = self.args.random_permute
        self.reset = True
        self.thres = self.args.thres
        self.samada = self.args.samada
        self.gap = self.args.gap
        self.cluster_num = self.args.cluster_num
        self.cossim = self.args.cossim
        self.queue_len = self.args.memleng
        self.score_queue_len = 20000
        self.mute_mutual_enhancement = False
    
    ###### 统计每个类sample score (MCM and Neglabel) 的均值和方差。使用GT 的label 还是predict label 呢？
    ## 也需要统计OOD sample 的分类别的均值和方差 (这里只能用predict label了)。 
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        ### 
        return
        net.eval()

        output_text = net.logit_scale * net.text_features.t() @ net.text_features # batch * class.
        score_output_text = torch.softmax(output_text , dim=1)
        self.score_prior = score_output_text[:1000, :1000].sum(1)   ## ID 在 ID上的得分。


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

        # 我下面要再做一个分析，来说明对于OOD 数据来说，很多negative labels 是没有激活的。很简单，取出1W个neg 类别的得分，分析一下。
        ## for ID
        id_activation_list = []
        for batch in tqdm(id_loader,
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
            if self.group_num == -1:
                drop = neg_logit.size(1) % self.group_size
                if drop > 0:
                    neg_logit = neg_logit[:, :-drop]
                group_num = int(neg_logit.size(1) / self.group_size) ## recalculating group num with group size.
                group_size = self.group_size
            else:
                drop = neg_logit.size(1) % self.group_num
                if drop > 0:
                    neg_logit = neg_logit[:, :-drop]
                group_size = int(neg_logit.size(1) / self.group_num)
                group_num = self.group_num

            if self.random_permute:
                # print('use random permute')
                SEED=0
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                idx = torch.randperm(neg_logit.shape[1]).to(output.device)
                neg_logit = neg_logit.T ## total_neg_num*B
                # pdb.set_trace()
                neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, group_size).contiguous()
            else:
                neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, group_size).contiguous()
            scores = []
            id_activation = []
            for i in range(group_num):
                full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
                full_sim = full_sim.softmax(dim=-1)
                id_prob = full_sim[:,pos_logit.shape[1]:] ## B * C_O' 
                id_activation.append(id_prob)
                pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
                scores.append(pos_score.unsqueeze(-1))
            scores = torch.cat(scores, dim=-1)
            id_activation = torch.cat(id_activation, dim=-1) # B * C_O
            id_activation_list.append(id_activation) 


        # 我下面要再做一个分析，来说明对于OOD 数据来说，很多negative labels 是没有激活的。很简单，取出1W个neg 类别的得分，分析一下。
        ood_activation_list = []
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
            if self.group_num == -1:
                drop = neg_logit.size(1) % self.group_size
                if drop > 0:
                    neg_logit = neg_logit[:, :-drop]
                group_num = int(neg_logit.size(1) / self.group_size) ## recalculating group num with group size.
                group_size = self.group_size
            else:
                drop = neg_logit.size(1) % self.group_num
                if drop > 0:
                    neg_logit = neg_logit[:, :-drop]
                group_size = int(neg_logit.size(1) / self.group_num)
                group_num = self.group_num

            if self.random_permute:
                # print('use random permute')
                SEED=0
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                idx = torch.randperm(neg_logit.shape[1]).to(output.device)
                neg_logit = neg_logit.T ## total_neg_num*B
                # pdb.set_trace()
                neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, group_size).contiguous()
            else:
                neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, group_size).contiguous()
            scores = []
            ood_activation = []
            for i in range(group_num):
                full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
                full_sim = full_sim.softmax(dim=-1)
                ood_prob = full_sim[:,pos_logit.shape[1]:] ## B * C_O' 
                ood_activation.append(ood_prob)
                pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
                scores.append(pos_score.unsqueeze(-1))
            scores = torch.cat(scores, dim=-1)
            ood_activation = torch.cat(ood_activation, dim=-1) # B * C_O
            ood_activation_list.append(ood_activation) 
            conf_in = scores.mean(dim=-1)

            # neglabel_score = conf_in
            # # max in prob - max out prob
            # if self.in_score == 'oodscore' or self.in_score == 'sum':
            #     neglabel_score = conf_in  ## = 1-conf_out
            # else:
            #     raise NotImplementedError
            # if torch.isnan(neglabel_score).any():
            #     pdb.set_trace()
            # for i in range(len(label)):
            #     ood_mcm_score_list[pred_in[i]].append(mcm_score[i].cpu().unsqueeze(0))
            #     ood_neglabel_score_list[pred_in[i]].append(neglabel_score[i].cpu().unsqueeze(0))    


        # pdb.set_trace()
        ood_activation = torch.cat(ood_activation_list, dim=0) # B * C_O
        ood_activation = torch.mean(ood_activation,0)

        id_activation = torch.cat(id_activation_list, dim=0) # B * C_O
        id_activation = torch.mean(id_activation,0)
        print('max ood activation is:', ood_activation.sort()[0][-1])
        print('max id activation is:', id_activation.sort()[0][-1])
        threshold_index = self.beta
        print('threshold ood activation is:', ood_activation.sort()[0][-threshold_index])
        print('threshold id activation is:', id_activation.sort()[0][-threshold_index])
        print('use the full 10K negative classes.')
        # final_activation =  ood_activation - id_activation
        # print('selecting some important negative labels before inference.')
        ##### if the randomperturbation is used, there will be some problem. because net.text_features is in original order but final_activation is in random order. 
        # net.text_features = torch.cat([net.text_features[:,:1000], net.text_features[:,1000:].t()[final_activation.sort()[1][-threshold_index:]].t()], dim=-1)

        # net.text_features = net.text_features.t()[:,ood_activation.sort()[1][-2000:]].t()  # 512*11K --> 512*
        # torch.save(ood_activation, 'ood_activation_imagenet_places.pth')  ## 尝试把这些没激活的negative class 去掉，看一下对结果的影响。先卡到2000， 结果提升了很多。

    #     # 把ID ood 的均值方差画到一张图上，来facilitate 设计category wise threshold. 

    def reset_memory(self):
        self.reset = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        net.eval()
        class_num = net.n_cls
        if self.reset:  ## For each ID-OOD dataset pair.
            self.class_count = torch.ones(net.n_cls) * self.beta
            # self.mean_value = torch.zeros(net.n_cls)
            # self.value_count = torch.ones(net.n_cls) * 3

            self.online_text_features = net.text_features.clone() ## 512* [ID class + neg class]
            self.online_text_features_all = net.text_features_all.clone()
            ############## use the ID labels as positive samples, and the noise image features as negative samples； better results can be achieved with the following 
            self.score_from_pos = net.logit_scale * net.text_features_all[:, :class_num].t() @ net.text_features_all ## id num * all num
            if self.cossim:
                # self.score_from_pos = self.score_from_pos.mean(0)  ## use cosine similarity as the activation score directly.
                self.score_from_pos = self.score_from_pos[torch.randperm(self.score_from_pos.size(0))]
            else:
                # self.score_from_pos = torch.softmax(self.score_from_pos, dim=1).mean(0)
                self.score_from_pos = torch.softmax(self.score_from_pos, dim=1)
                self.score_from_pos = self.score_from_pos[torch.randperm(self.score_from_pos.size(0))]
            self.score_from_pos = self.score_from_pos[:, class_num:].float()
            self.score_from_pos = update_queue(self.score_from_pos, self.score_from_pos, self.queue_len, init=True)
            
            # self.pos_num = class_num

            self.score_from_neg = net.logit_scale * net.noise_image_features @ net.text_features_all
            if self.cossim:
                self.score_from_neg =  self.score_from_neg
            else:
                self.score_from_neg = torch.softmax(self.score_from_neg, dim=1)
            self.score_from_neg = self.score_from_neg[torch.randperm(self.score_from_neg.size(0))] ## 

            self.score_from_neg = self.score_from_neg[:, class_num:].float()
            self.score_from_neg = update_queue(self.score_from_neg, self.score_from_neg, self.queue_len, init=True)
            # self.neg_num = net.noise_image_features.size(0)  ## number of negative image samples.
            # self.neg_num_text_all = net.text_features_all.size(1) - class_num
            ## 设置一个自适应的threshold, 来处理不同ood number 以及 其他的一些情况。先设计一个tensor, 保留所有的ood score, including from positive labels and noise images.
            combined_score = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。
            # combined_score = self.score_from_neg  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。
            selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1)
            if self.mute_mutual_enhancement:
                output = net.logit_scale * net.text_features_all[:, :class_num].t() @ self.online_text_features ## use negative labels of NegLabel
            else:
                output = net.logit_scale * net.text_features_all[:, :class_num].t() @ selected_text_features
            ######################################################### 这里用 NegLabel score 来做threshold, 改成 activation-aware 的做法。
            conf_in_id = activation_aware_score(output, class_num, self.beta, step=self.gamma)

            # full_sim = output.softmax(dim=-1)
            # conf_in_id = full_sim[:, :class_num].sum(dim=-1)
            if self.mute_mutual_enhancement:
                output = net.logit_scale * net.noise_image_features @ self.online_text_features
            else:
                output = net.logit_scale * net.noise_image_features @ selected_text_features
            # full_sim = output.softmax(dim=-1)
            # conf_in_ood = full_sim[:, :class_num].sum(dim=-1)
            conf_in_ood = activation_aware_score(output, class_num, self.beta, step=self.gamma)

            self.scoretensor = torch.cat((conf_in_id, conf_in_ood), dim=0)
            # self.neg_image_feature_list = [] ## 
            ### advantage: adaptive, ID-OOD guided, active
            ### 现有基于negative proxy 的OOD detection has gathering increasing attention, 但是
            self.reset = False
        image_features, text_features, logit_scale = net(data, return_feat=True)
        # image_classifier = self.image_classifier


        output = logit_scale * image_features @ self.online_text_features # batch * class.
        output_all = logit_scale * image_features @ self.online_text_features_all  # batch * dim   @ dim * num        
        
        _, pred_in = torch.max(output[:, :class_num], dim=1) 

        # normalized_count = self.class_count.float() / self.class_count.sum() ### 
        # # pdb.set_trace()
        # # print('pay attention, no scale is applied here.')
        # pos_logit = output[:, :class_num] ## B*C
        # neg_logit = output[:, class_num:] ## B*total_neg_num
        # # drop = neg_logit.size(1) % self.group_num
        # if self.group_num == -1:
        #     drop = neg_logit.size(1) % self.group_size
        #     if drop > 0:
        #         neg_logit = neg_logit[:, :-drop]
        #     group_num = int(neg_logit.size(1) / self.group_size) ## recalculating group num with group size.
        #     group_size = self.group_size
        # else:
        #     drop = neg_logit.size(1) % self.group_num
        #     if drop > 0:
        #         neg_logit = neg_logit[:, :-drop]
        #     group_size = int(neg_logit.size(1) / self.group_num)
        #     group_num = self.group_num

        # if self.random_permute:
        #     # print('use random permute')
        #     SEED=0
        #     torch.manual_seed(SEED)
        #     torch.cuda.manual_seed(SEED)
        #     idx = torch.randperm(neg_logit.shape[1]).to(output.device)
        #     neg_logit = neg_logit.T ## total_neg_num*B
        #     # pdb.set_trace()
        #     neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        # else:
        #     neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        # scores = []
        # for i in range(group_num):
        #     full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
        #     full_sim = full_sim.softmax(dim=-1)
        #     pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
        #     scores.append(pos_score.unsqueeze(-1))
        # scores = torch.cat(scores, dim=-1)
        # conf_in = scores.mean(dim=-1).float() 
        # ### 先算score, 在更新count. 
        # for i in range(len(pred_in)):
        #     self.class_count[pred_in[i].item()] += 1
        if self.cossim:
            actscore = output_all
        else:
            actscore = torch.softmax(output_all.float(), dim=1)  # batch * [id+ood class num]

        ##################################### 每次都取1000， 但是每次取的可能不一样，根据当前的统计量，挑1000 个 [比如最近10000个样本的统计量，挑出OOD-ID 响应最高的1000个neg class.]
        combined_score = self.score_from_neg.mean(0) - self.score_from_pos.mean(0)  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。
        # combined_score =  self.score_from_neg  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。、
        # pdb.set_trace()
        selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1)
        #
        if self.mute_mutual_enhancement:
            output = logit_scale * image_features @ self.online_text_features
        else:
            output = logit_scale * image_features @ selected_text_features
        ######### 要先分析一下 clip feature space 中， t2t and i2i, and t2i 的cosine similarity 的关系，可能 i2i 的不同类 大于 t2i 的同类; 这样的话直接把image feature 作为
        # pdb.set_trace()
        # i2i = image_features @ image_features.t()  ## most 0.5+
        # t2t = self.online_text_features_all.t() @ self.online_text_features_all  ## most 0.7+
        # i2t = image_features @ self.online_text_features_all ## most 0.1-0.3+
        output = output.float()
        conf_in_domain = activation_aware_score(output, class_num, self.beta, step=self.gamma)

        # full_sim = output.softmax(dim=-1)
        # conf_in_domain = full_sim[:, :class_num].sum(dim=-1)  ######## 这里得 score conf 也考虑换成confidence aware 的。 

        self.scoretensor = torch.cat((self.scoretensor, conf_in_domain), dim=0) ## 这里也要做成queue 的形式。否则会太长了。
        if self.scoretensor.size(0) > self.score_queue_len:
            self.scoretensor = self.scoretensor[-self.score_queue_len:]
        
        # print('use manual threshold', self.thres)
        best_threshold = find_best_threshold(self.scoretensor) 
        self.thres = best_threshold

        # print(" best_threshold", best_threshold)  ### 为啥是 0.6 左右？ 很奇怪，是否可能是因为

        # pdb.set_trace()
        activate_indicator_id = conf_in_domain > (self.thres + self.gap * (1-self.thres)) ## select high confident samples； 这里用的还是original neglabel score. 
        activate_indicator_ood = conf_in_domain < (self.thres - self.gap * self.thres)  ## only select high confident samples
        used_cluster_num = 0
        # if self.cluster_num > 0:
        #     ######### 要先在所有上class 上做softmax, 然后想办法从 negative labels 中选出一部分？ 怎么选呢？主要是这个时候也不知道那些是ID, 哪些是OOD?  不能算ID_score OOD_score? 哪怎么做呢？ 
        #     ## 把 标准差最小的移除掉？ 因为ID-OOD 都有的话，希望保留分数有高有低的，如果分数都是高的？ 如果分数都低，说明不重要？
        #     ######### 现在这个做法没啥用呀。。。 为什么呢？
        #     for i in range(activate_indicator_ood.size(0)):
        #         if activate_indicator_ood[i].item():
        #             self.neg_image_feature_list.append(image_features[i].unsqueeze(0))
        #     # pdb.set_trace()
        #     if len(self.neg_image_feature_list) > self.cluster_num:
        #         used_cluster_num = self.cluster_num
        #     else:
        #         used_cluster_num = len(self.neg_image_feature_list)
        #     if used_cluster_num > 0:
        #         clusters, cluster_labels = kmeans_l2_normalized(torch.cat(self.neg_image_feature_list, dim=0), used_cluster_num) ## neg_image_feature_list 这个可能是空的。
        #         cluster_sizes = torch.bincount(cluster_labels)  # [K]
        #         # 获取按簇大小降序排列的索引
        #         sorted_indices = torch.argsort(cluster_sizes, descending=True)  # 排序后的簇索引 [K]
        #         # 按排序后的索引重新排列 clusters
        #         clusters = clusters[sorted_indices]

        #         clusters /= clusters.norm(dim=-1, keepdim=True)
        #         output_clusters = logit_scale * clusters @ self.online_text_features_all[:,class_num:] ## cluster num * negative text num
        #         neg_prob = torch.softmax(output_clusters, dim=-1) ## cluster num * negative text num
        #         reconst_clusters = neg_prob @ net.text_features_all[:, class_num:].T ## cluster num*512
        #         reconst_clusters /= reconst_clusters.norm(dim=-1, keepdim=True)


        # pdb.set_trace() 
        ## 统计当前batch 的 score from pos and score from neg； 
        if torch.any(activate_indicator_id):
            score_pos_in_batch = actscore[activate_indicator_id][:, class_num:]
            ins_adaptive_pos_score = self.alpha * self.score_from_pos.mean(0) + (1 - self.alpha) * score_pos_in_batch.mean(0)
        else:
            ins_adaptive_pos_score = self.score_from_pos.mean(0)

        if torch.any(activate_indicator_ood):
            score_neg_in_batch = actscore[activate_indicator_ood][:, class_num:]
            ins_adaptive_neg_score = self.alpha * self.score_from_neg.mean(0) + (1 - self.alpha) * score_neg_in_batch.mean(0)
        else:
            ins_adaptive_neg_score = self.score_from_neg.mean(0)

        # num_pos_in_batch = 0
        # score_pos_in_batch = torch.zeros_like(self.score_from_pos)
        # for i in range(activate_indicator_id.size(0)):
        #     if activate_indicator_id[i].item():
        #         score_pos_in_batch = score_pos_in_batch + actscore[i][class_num:]
        #         num_pos_in_batch = num_pos_in_batch + 1
        # if num_pos_in_batch > 0:
        #     score_pos_in_batch = score_pos_in_batch / num_pos_in_batch
        #     ins_adaptive_pos_score = self.alpha * self.score_from_pos + (1 - self.alpha) * score_pos_in_batch
        # else:
        #     ins_adaptive_pos_score = self.score_from_pos

        # num_neg_in_batch = 0
        # score_neg_in_batch = torch.zeros_like(self.score_from_neg)
        # for i in range(activate_indicator_ood.size(0)):
        #     if activate_indicator_ood[i].item():
        #         score_neg_in_batch = score_neg_in_batch + actscore[i][class_num:]
        #         num_neg_in_batch = num_neg_in_batch + 1
        # if num_neg_in_batch > 0:
        #     score_neg_in_batch = score_neg_in_batch / num_neg_in_batch ### 啊，这里之前写成了pos, 导致之前所有的实验结果都不对！！，重新做。
        #     ins_adaptive_neg_score = self.alpha * self.score_from_neg + (1-self.alpha) * score_neg_in_batch
        # else:
        #     ins_adaptive_neg_score = self.score_from_neg

        
        ############### 下面这个作用很大，其实就是用了自己的当前样本来重新计算了activated score; 
        ####### using the updated score to recalculating the conf_in ?? This is important, leading to large improvement! 这里就是 instance-adaptive activated labels. 
        combined_score = ins_adaptive_neg_score - ins_adaptive_pos_score  ## 69554.
        # combined_score =  ins_adaptive_neg_score  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。

        if used_cluster_num > 0:
            selected_text_features = torch.cat([net.text_features[:,:class_num], reconst_clusters.t(), self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta-used_cluster_num]].t()], dim=-1)
        else:
            selected_text_features = torch.cat([net.text_features[:,:class_num], self.online_text_features_all[:,class_num:].t()[combined_score.sort(descending=True)[1][:self.beta]].t()], dim=-1) ## 512*2000
        
        ######## 图像和图像之间的similarity 远大于同 图像和text 之间，导致不管ID & OOD 都和 negative image 相似度很高，所以，ID score 都是0; 所以clusters 要改成 negative text label 加权得到。
        # selected_combined_score = combined_score.sort(descending=True)[0][:self.beta]  ## 这里是从大到小排的！！### 里面有负数，这样不行，要全都是非负。
        # selected_combined_score = torch.clamp(selected_combined_score, min=0) ## make it non-negative
        # ################################## CVPR submission analyses, try explicit weighting according to activation score.
        # value, index = combined_score.sort(descending=True)
        # weight_by_activation_score = value[:self.beta]
        # weight_by_activation_score = weight_by_activation_score / weight_by_activation_score.mean() ## only negative labels
        # ones_tensor_id = torch.ones(class_num, device=weight_by_activation_score.device, dtype=weight_by_activation_score.dtype)
        # # Concatenate the tensors
        # weight_by_activation_score = torch.cat((ones_tensor_id, weight_by_activation_score), dim=0)
        import pdb; pdb.set_trace()

        output = logit_scale * image_features @ selected_text_features  ## batch * [id+ood class num]
        # output = (output * weight_by_activation_score.unsqueeze(0))[:,1000:]
        # full_sim = output.softmax(dim=-1)
        # # conf_in = - (full_sim[:, class_num:] * selected_combined_score.unsqueeze(0)).sum(dim=-1)  ### 加了这个，对near ood 有好处，对far ood 有坏处。
        # conf_in = full_sim[:, :class_num].sum(dim=-1)
        # print(conf_in)

        ######## 既然不能加权，那就试试根据activation score online 决定 ood number. 可能也不太行，因为最开始的时候，第一个epoch 的时候，用了多少就要定下来。
        conf_in = activation_aware_score(output, class_num, self.beta, step=self.gamma)
        # conf_in = activation_aware_score(output, class_num, self.beta, step=self.gamma, weight_by_activation_score=weight_by_activation_score)
        if torch.any(activate_indicator_id):
            self.score_from_pos = update_queue(actscore[activate_indicator_id][:, class_num:], self.score_from_pos, self.queue_len, init=False)
        if torch.any(activate_indicator_ood):
            self.score_from_neg = update_queue(actscore[activate_indicator_ood][:, class_num:], self.score_from_neg, self.queue_len, init=False)

        # ####################### 下面这个用起来结果反而更差了，不能这么做； 那怎么做？感觉这个分太细了；改成每100个分一个； 也不行，结果更差了; 那怎么办？ 动态决定ood number?? 但是ood number 数量不一致的情况下score 分数如何比较。
        # if self.gamma != 0:   ####### gamma =5 确实结果变好了，比0好，说明按照activation score 来加权确实结果更好！结果稳定了很多。
        #     softmax_sums = []
        #     # pdb.set_trace()
        #     # 在 1001 到 2000 列分别做 softmax
        #     step = int(self.gamma)
        #     for i in range(class_num, class_num+self.beta, step):  # 列索引从 1000 到 1999
        #         ## 下面方法每一段包含了前一段， 相当于前面的算了比较多次，权重大一些; 这样缓解了一些情况：step 1,2,10 比step 0效果好，但是还是会随着neg number 数量增加而变差; 一定要把activation score 的权重算进来。
        #         softmax_output = output[:,:i+step].softmax(dim=-1) 
        #         sum_score = softmax_output[:, :class_num].sum(dim=-1)  # 即使没做加权，结果也好了一些; 就用这个了。

        #         ##### 还是要做加权，试试乘以新加部分的权重!! 加权一直做不出来。
        #         ### 还是不行，乘上之后加过下降了！ 而且并没有想象中的对 large negative number 的稳定性， number 多的时候结果还是下降了不少； 下面的实验不再继续尝试。
        #         # score_weight = selected_combined_score[i-class_num:i-class_num+step].sum()
        #         # sum_score = sum_score * score_weight  
      
        #         #  如果每一段不包含前面一段呢？手动对每个进行加权？用 activation score 进行加权？
        #         # softmax_output = torch.cat((output[:,:class_num], output[:,i:i+step]),dim=-1).softmax(dim=-1)  # 对step 异常敏感; 即使某个step 有效也不行因为
        #         ############# 一定要改成，对后面的negative label apply small weights; 这样negative label 再多，其对结果的影响也会小一些。
        #         # sum_score = softmax_output[:, :class_num].sum(dim=-1)  # 分group 算
        #         # sum_score = sum_score * selected_combined_score[i-class_num:i-class_num+step].sum()  ### 结果对step 非常敏感，不能用。
        #         # print(selected_combined_score[class_num:i-class_num+step].sum())
        #         # 根绝neg score 做加权。这样 重要的negative label 可以得到更高的权重。引入更多的negative label 的影响就小了，就对neglabel number 不敏感了。否则还是敏感。
        #         softmax_sums.append(sum_score)
        #     # 将 softmax_sums 堆叠为一个张量：batch_size x 1000
        #     softmax_sums = torch.stack(softmax_sums, dim=-1)
        #     # 对列（1000 次 softmax 的结果）求均值作为最终结果
        #     conf_in = softmax_sums.mean(dim=-1)  # 最终为 batch_size 的向量

        ######### 做一个neglabel 生成的方法，用多个label 加权和作为neglabel; 生成的negative label 的activation score 也可以使用其 activation score加权和 !!!! 做成另外的，而且限制最大数量。

        # ## 这里用的手动fixed threshold, 不好，最好可以用自动划分的方式。
        # ## 统计符合条件的 sample, 统计他们在negative class 上的响应，把对positive sample 响应高的negative class 替换。
        # for i in range(activate_indicator_id.size(0)):
        #     if activate_indicator_id[i].item():
        #         self.score_from_pos = (self.score_from_pos * self.pos_num + actscore[i][class_num:]) / (self.pos_num + 1)
        #         self.pos_num = self.pos_num + 1


        # ## 统计符合条件的 sample, 统计他们在negative class 上的响应，把对negative sample 响应低的negative class 替换。
        # for i in range(activate_indicator_ood.size(0)):
        #     if activate_indicator_ood[i].item():
        #         self.score_from_neg = (self.score_from_neg * self.neg_num + actscore[i][class_num:]) / (self.neg_num + 1)
        #         self.neg_num = self.neg_num + 1

        # ############### 下面这个作用很大，其实就是用了自己的当前样本来重新计算了activated score; 
        # ####### using the updated score to recalculating the conf_in ?? This is important, leading to large improvement! 这里就是 instance-adaptive activated labels. 
        # combined_score = self.score_from_neg - self.score_from_pos  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。
        # selected_text_features = torch.cat([net.text_features[:,:1000], self.online_text_features_all[:,1000:].t()[combined_score.sort()[1][-self.beta:]].t()], dim=-1)
        # output = logit_scale * image_features @ selected_text_features
        # full_sim = output.softmax(dim=-1)
        # conf_in = full_sim[:, :1000].sum(dim=-1)

        ####### 这里基本完成了。再加一个对于选出来的negative samples, 把他加到negative pool 中，并assign a 均值score to it;  if we use features of negative image directly, maybe 
        ### there will be some modality gap; therefore, we can 使用最小二乘法来construct the close text feature to the negative image, with existing negative text features. 
        ####### 后面做一下这一步。
        
        # ########### 直接用图像特征不行，效果奇差; 因为 image text 存在modality gap, 会导致 all image feature 同 negative image feature 的相似度高于同text feature的。
        # ########3## 下面这个用最小二乘来扩充负类边界的速度太慢了，改成直接用negative softmax prob 对negative feature 做加权和； 问题是这和直接用多个negative labels 有区别吗？  
        # # 可能有？毕竟这样一个可能包含了多个的信息，这样对于negative num 的敏感度会下降很多。试试。
        # new_neg_image_feature_list = []
        # if len(neg_image_feature_list) > 0:
        #     length = len(neg_image_feature_list)
        #     # for j in range(length):
        #     #     ######### too slow!!
        #     #     # b = neg_image_feature_list[j] ## 1*512
        #     #     # AAT = torch.matmul(net.text_features_all.T, net.text_features_all) ## N*N
        #     #     # Ab = torch.matmul(net.text_features_all.T, b) ## N
        #     #     # # ww = torch.linalg.solve(AAT, Ab)
        #     #     # ww = torch.linalg.solve(AAT.float(), Ab.float())
        #     #     # neg_text_feat = torch.matmul(ww, net.text_features_all.T.float()).unsqueeze(0)
        #     #     # neg_text_feat /= neg_text_feat.norm(dim=-1, keepdim=True)
        #     #     # new_neg_image_feature_list.append(neg_text_feat.half())
        #     #     #########
        #     # b = neg_image_feature_list[j]
        #     # pdb.set_trace()
        #     #################### 引入下面weighted negative text 之后，结果变差了。我猜大概率是把orginal 很多都挤掉了，但是新增加的同质化又太严重了（都很像）
        #     # 改进： 1. 控制数量，避免把original 都挤掉, 而且同质化太严重。 2. 算分数的时候分开算 
        #     neg_prob = torch.softmax(output_all[:, class_num:][:, :self.neg_num_text_all][activate_indicator], dim=-1) ## B*num_neg
        #     weighted_neg_feat = neg_prob @ net.text_features_all[:, class_num:].T ## B*512
        #     weighted_neg_feat /= weighted_neg_feat.norm(dim=-1, keepdim=True)

        #     self.online_text_features_all = torch.cat([self.online_text_features_all, weighted_neg_feat.t()], dim=-1)
        #     mean_score_from_neg = self.score_from_neg.mean()
        #     mean_score_from_pos = self.score_from_pos.mean()
        #     self.score_from_neg = torch.cat([self.score_from_neg, mean_score_from_neg.repeat(length)])
        #     self.score_from_pos = torch.cat([self.score_from_pos, mean_score_from_pos.repeat(length)])



        # if (combined_score / combined_score.max().item() < self.alpha).any().item(): ## 如果有true,也就是有比较小的值。
        #     pdb.set_trace()
        #     remain_ind = combined_score / combined_score.max().item() > self.alpha
        #     if self.score_from_neg[remain_ind].size(0) > self.group_size: ## make sure the neglabel is larger than group size.
        #         self.score_from_neg = self.score_from_neg[remain_ind]
        #         self.score_from_pos = self.score_from_pos[remain_ind]
        #         remain_ood_feat = self.online_text_features[:,class_num:].t()[remain_ind].t()
        #         self.online_text_features = torch.cat([self.online_text_features[:,:class_num], remain_ood_feat], dim=-1)
        #         combined_score = self.score_from_neg
        #         print(combined_score.size(0))

        # ##################################### 用第一个batch 来做初筛。
        # combined_score = self.score_from_neg  ## 这里只用了neg score, positive score 最好也可以用上; 但是positive 用上会导致不能用比例来卡。
        # if (combined_score / combined_score.max().item() < self.alpha).any().item(): ## 如果有true,也就是有比较小的值。
        #     pdb.set_trace()
        #     remain_ind = combined_score / combined_score.max().item() > self.alpha
        #     if self.score_from_neg[remain_ind].size(0) > self.group_size: ## make sure the neglabel is larger than group size.
        #         self.score_from_neg = self.score_from_neg[remain_ind]
        #         self.score_from_pos = self.score_from_pos[remain_ind]
        #         remain_ood_feat = self.online_text_features[:,class_num:].t()[remain_ind].t()
        #         self.online_text_features = torch.cat([self.online_text_features[:,:class_num], remain_ood_feat], dim=-1)
        #         combined_score = self.score_from_neg
        #         print(combined_score.size(0))
        
  

            

        # ### 用updated negative proxies 来重新计算score.
        # output = logit_scale * image_features @ self.online_text_features # batch * class.
        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        # _, pred_in = torch.max(output[:, :class_num], dim=1) 

        # normalized_count = self.class_count.float() / self.class_count.sum() ### 
        # # pdb.set_trace()
        # # print('pay attention, no scale is applied here.')
        # pos_logit = output[:, :class_num] ## B*C
        # neg_logit = output[:, class_num:] ## B*total_neg_num
        # drop = neg_logit.size(1) % group_num
        # if drop > 0:
        #     neg_logit = neg_logit[:, :-drop]

        # if self.random_permute:
        #     # print('use random permute')
        #     SEED=0
        #     torch.manual_seed(SEED)
        #     torch.cuda.manual_seed(SEED)
        #     idx = torch.randperm(neg_logit.shape[1]).to(output.device)
        #     neg_logit = neg_logit.T ## total_neg_num*B
        #     # pdb.set_trace()
        #     neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        # else:
        #     neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        # scores = []
        # for i in range(group_num):
        #     full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
        #     full_sim = full_sim.softmax(dim=-1)
        #     pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
        #     scores.append(pos_score.unsqueeze(-1))
        # scores = torch.cat(scores, dim=-1)
        # conf_in = scores.mean(dim=-1).float() 

        # pdb.set_trace()

        # pdb.set_trace()
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'sum':
            conf_in = conf_in  ## = 1-conf_out
        elif self.in_score == 'div_count':
            for i in range(len(conf_in)):
                conf_in[i] = conf_in[i] / normalized_count[pred_in[i]] / net.n_cls ### 错了，normalized_count 要根据label 找。
        else:
            raise NotImplementedError
        if torch.isnan(conf_in).any():
            pdb.set_trace()

        return pred_in, conf_in


    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
