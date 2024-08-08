import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config
from openood.losses import soft_cross_entropy
import torch.distributions as dist
from .lr_scheduler import cosine_annealing
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

def softmax_oodmerge(logits, softlabels):
    # logits: B* (1000+N)
    # soft_label: B*1000+1
    epsilon = 1e-6
    class_num = softlabels.size(1) - 1
    batch_size = logits.size(0)
    full_prob = F.softmax(logits, dim=1)
    prob = torch.cat([full_prob[:, :class_num], full_prob[:, class_num:].sum(1, keepdim=True)], dim=1)
    clipped_prob = torch.clamp(prob, epsilon, 1 - epsilon)
    loss = -torch.sum(softlabels * torch.log(clipped_prob), dim=1)
    loss  = torch.mean(loss)
    # print('loss', loss)
    return loss

def energy(logits, weight = 1):
    batch_size, nClass = logits.size(0), logits.size(1)
    prob = F.softmax(logits, dim=1)

    if (prob.data.cpu() == 0).sum() != 0:
        weight_sum = torch.FloatTensor(batch_size, nClass).fill_(0)
        weight_sum[prob.data.cpu() == 0] = 1e-6
        weight_sum = Variable(weight_sum).cuda()
        loss_sum = -(prob + weight_sum).log().mul(prob).sum(1) * weight
    else:
        loss_sum = -prob.log().mul(prob).sum(1) * weight

    return loss_sum.mean()



class TensorFIFOQueue:
    def __init__(self, capacity, feature_dim):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.queue = torch.empty((0, feature_dim)).cuda()
    
    def push(self, tensor):
        # 允许批量添加元素
        if tensor.ndim != 2 or tensor.shape[1] != self.feature_dim:
            raise ValueError(f"Tensor must have a shape of [N, {self.feature_dim}]")
        # 计算可以添加的新元素数量
        num_new_elements = tensor.size(0)
        num_available_spots = self.capacity - self.queue.size(0)
        
        # 如果有足够的空间，直接添加
        if num_new_elements <= num_available_spots:
            self.queue = torch.cat((self.queue, tensor), dim=0)
        else:
            # 如果空间不足，移除旧元素以腾出空间
            num_elements_to_remove = num_new_elements - num_available_spots
            self.queue = torch.cat((self.queue[num_elements_to_remove:], tensor), dim=0)
    
    def pop(self, num_elements):
        # 弹出多个元素
        if self.queue.shape[0] < num_elements:
            raise ValueError(f"Not enough elements in the queue to pop {num_elements}")
        popped_elements = self.queue[:num_elements]
        self.queue = self.queue[num_elements:]
        return popped_elements

    def __len__(self):
        return self.queue.shape[0]

    def is_full(self):
        return self.queue.shape[0] == self.capacity

class ClassTensorQueues:
    def __init__(self, class_num, capacity, feature_dim):
        self.class_num = class_num
        self.queues = {class_id: TensorFIFOQueue(capacity, feature_dim) for class_id in range(class_num)}
        self.mean = {class_id: torch.empty((0, feature_dim)).cuda() for class_id in range(class_num)}
        self.covariance_matrix = {class_id: torch.empty((0, feature_dim)).cuda() for class_id in range(class_num)}

    def push(self, tensors, labels):
        # tensors: (N, feature_dim) | labels: (N,)
        if tensors.ndim != 2 or labels.ndim != 1 or tensors.shape[0] != labels.shape[0]:
            raise ValueError("The dimensions of tensors and labels do not match.")
        
        for tensor, label in zip(tensors, labels):
            if label < 0 or label >= self.class_num:
                raise ValueError(f"Label must be between 0 and {self.class_num - 1}")
            self.queues[label.item()].push(tensor.unsqueeze(0))

    def is_full(self):
        # 检查所有队列是否已满
        return all(queue.is_full() for queue in self.queues.values())
    
    def update_guassian(self):
        for class_id in range(self.class_num):
            self.mean[class_id] = torch.mean(self.queues[class_id].queue, 0)
            vectors_centered = self.queues[class_id].queue - self.mean[class_id]
            self.covariance_matrix[class_id] = vectors_centered.t().matmul(vectors_centered) / (vectors_centered.size(0) - 1)
            epsilon = 1e-8
            self.covariance_matrix[class_id] += epsilon * torch.eye(self.covariance_matrix[class_id].size(0)).cuda()
        self.multivariate_normal_dist = {class_id: dist.MultivariateNormal(self.mean[class_id], self.covariance_matrix[class_id]) for class_id in range(self.class_num)}

    def sampling_guassian(self, num_samples=1000, selected_number=10, soft_y=0, soft_split=False): 
        if soft_split:
            # sampled feature may present soft-labels based on the relative probability ranking
            syn_x_list = []
            syn_y_list = []
            id_label = []
            for class_id in range(self.class_num):
                multivariate_normal_dist = self.multivariate_normal_dist[class_id]
                new_samples = multivariate_normal_dist.sample((num_samples,))   
                # pdb.set_trace()
                log_probabilities = multivariate_normal_dist.log_prob(new_samples)
                # prob in high dimension is typically very small, we usually use log_probabilities.
                value, indice = torch.sort(log_probabilities) ## samll 2 large
                syn_x_list.append(new_samples[indice]) ## from low density to high density

                gradual_weights = torch.arange(num_samples).view(-1,1).cuda() / num_samples
                soft_ood_y = soft_y.new_zeros((num_samples, soft_y.size(1))) ## N * 1001
                soft_ood_y[:,-1] = 1
                soft_id_y = soft_y.new_zeros((num_samples, soft_y.size(1))) ## N * 1001
                soft_id_y[:, class_id] = 1
                syn_y_cate = gradual_weights * soft_ood_y + (1 - gradual_weights) * soft_id_y
                syn_y_list.append(syn_y_cate)
            syn_x = torch.cat(syn_x_list, dim=0)
            syn_y = torch.cat(syn_y_list, dim=0)
        else:
            # sampled feature may present soft-labels based on the relative probability ranking
            id_feat = []
            id_label = []
            ood_feat = []
            for class_id in range(self.class_num):
                multivariate_normal_dist = self.multivariate_normal_dist[class_id]
                new_samples = multivariate_normal_dist.sample((num_samples,))   
                # pdb.set_trace()
                log_probabilities = multivariate_normal_dist.log_prob(new_samples)
                # prob in high dimension is typically very small, we usually use log_probabilities.
                # probabilities = torch.exp(log_probabilities)
                value, indice = torch.sort(log_probabilities) ## samll 2 large
                id_syn = new_samples[indice[-selected_number:]]
                ood_syn = new_samples[indice[:selected_number]]
                id_feat.append(id_syn)
                id_label.append(torch.ones(selected_number).cuda() * class_id)
                ood_feat.append(ood_syn)
                # pdb.set_trace()
            id_feat = torch.cat(id_feat, dim=0)
            ood_feat = torch.cat(ood_feat, dim=0)
            id_label = torch.cat(id_label, dim=0)
            # pdb.set_trace()
            syn_x = torch.cat([id_feat, ood_feat], dim=0)
            soft_id_y = torch.zeros((id_feat.size(0), soft_y.size(1)), dtype=soft_y.dtype, device=soft_y.device)
            soft_id_y[torch.arange(id_feat.size(0)).cuda(), id_label.long().cuda()] = 1
            soft_ood_y = soft_y.new_zeros((ood_feat.size(0), soft_y.size(1))) ## 128 * 1001
            soft_ood_y[:,-1] = 1
            syn_y = torch.cat([soft_id_y, soft_ood_y], dim=0)
        return syn_x, syn_y




# copyed from: https://github.com/changdaeoh/multimodal-mixup
## can not self-mix, since torch.sin(theta)=0 will lead nan.
def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2


def create_mixed_indices(y):
    # 检索小于1000的样本的索引
    indices_less_than_1000 = torch.where(y < 1000)[0]
    # 检索大于等于1000的样本的索引
    indices_greater_than_999 = torch.where(y >= 1000)[0]

    # 如果任一组没有样本，则无法进行配对
    if len(indices_less_than_1000) == 0 or len(indices_greater_than_999) == 0:
        print("Not enough samples to pair up in one or both of the groups.")
        mixed_indices = mixed_indices = torch.arange(len(y))
    else:
        # 创建一个与输入y相同长度的索引数组
        mixed_indices = torch.arange(len(y))

        # 随机选择大于等于1000的样本的索引进行配对
        for idx in indices_less_than_1000:
            mixed_indices[idx] = indices_greater_than_999[torch.randint(len(indices_greater_than_999), (1,))]

        # 同样，随机选择小于1000的样本的索引进行配对
        for idx in indices_greater_than_999:
            mixed_indices[idx] = indices_less_than_1000[torch.randint(len(indices_less_than_1000), (1,))]

        return mixed_indices

# https://github.com/FrancescoPinto/RegMixup/blob/main/models/regmixup.py
def mixup_data(x, y, soft_y, alpha=1.0, mix_strategy='mixup'):
    """Returns mixed inputs, pairs of targets, and lambda."""

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # if lam > 0.5:
    #     lam = 1-lam # [0-0.5]
    # index = torch.randperm(batch_size).cuda()
    if mix_strategy == 'mixup' or mix_strategy == 'cutmix' or mix_strategy == 'manimix' or mix_strategy == 'geomix' or mix_strategy == 'manimixrev':
        batch_size = x.size()[0]
        index = torch.arange(batch_size-1, -1, -1).cuda()
        mask = y == y[index]
        while mask.any():
            # print('refine mask') ## for imagenet, it is seldom to refine mask (less than 1/10000)
            # 获取需要修正的索引
            swap_with = torch.randperm(batch_size).cuda()
            # 对于所有B[i] == A[i]的情况，随机选择一个索引与之交换
            index[mask] = index[swap_with[mask]]
            # 更新掩码
            mask = y == y[index]
        soft_ya, soft_yb = soft_y, soft_y[index]
        if mix_strategy == 'mixup':
            mixed_x = lam * x + (1 - lam) * x[index]
        elif mix_strategy == 'cutmix':
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            mixed_x = x.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        elif mix_strategy == 'manimix':
            mixed_x = lam * x + (1 - lam) * x[index]
            mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
        elif mix_strategy == 'manimixrev':
            mixed_x = lam * x + (1 - lam) * x[index] ## lam <0.5
            mixed_x = x[index] - (mixed_x - x[index]) ## major part:  x[index]
            mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
            ood_label = soft_y.new_zeros((soft_y.size(0), soft_y.size(1))) ## 128 * 1001
            ood_label[:,-1] = 1
            soft_ya = ood_label
        elif mix_strategy == 'geomix':
            mixed_x = sph_inter(x, x[index], lam) 
    elif mix_strategy == 'cross_dis':
        # x  N*512
        # y, N
        # soft_y N*2000
        index = create_mixed_indices(y)
        # pdb.set_trace()
        soft_ya, soft_yb = soft_y, soft_y[index]
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
  
    elif mix_strategy == 'manimix_wccm' or mix_strategy == 'wccm':
        # raise NotImplementedError
        # pdb.set_trace()
        image_features, text_feats = x # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        text_num = text_feats.size()[1]
        random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
        selected_text_feat = text_feats[y, random_indices, :] # same class.
        mixed_x = lam * image_features + (1 - lam) * selected_text_feat # within class, cross modal.
        mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
        soft_ya, soft_yb = soft_y, soft_y
    elif mix_strategy == 'mani_cccm':
        image_features, text_feats = x # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        class_num, text_num = text_feats.size()[0], text_feats.size()[1]
        # data mix cross class.
        random_class_indices = torch.randint(0, class_num-1, (batch_size,)).cuda()
        random_class_indices = torch.where(random_class_indices >= y, random_class_indices + 1, random_class_indices)
        random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
        selected_text_feat = text_feats[random_class_indices, random_indices, :] # same class.
        mixed_x = lam * image_features + (1 - lam) * selected_text_feat # within class, cross modal.
        mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)

        soft_text_y = torch.zeros_like(soft_y)
        soft_text_y[torch.arange(batch_size).cuda(), random_class_indices] = 1
        soft_ya, soft_yb = soft_y, soft_text_y
    elif mix_strategy == 'geomix_wccm':
        image_features, text_feats = x # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        text_num = text_feats.size()[1]
        random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
        selected_text_feat = text_feats[y, random_indices, :] # same class.
        mixed_x = sph_inter(image_features, selected_text_feat, lam) 
        soft_ya, soft_yb = soft_y, soft_y
    elif mix_strategy == 'geo_cccm':
        image_features, text_feats = x # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        class_num, text_num = text_feats.size()[0], text_feats.size()[1]
        # data mix cross class.
        random_class_indices = torch.randint(0, class_num-1, (batch_size,)).cuda()
        random_class_indices = torch.where(random_class_indices >= y, random_class_indices + 1, random_class_indices)
        random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
        selected_text_feat = text_feats[random_class_indices, random_indices, :] # same class.
        mixed_x = sph_inter(image_features, selected_text_feat, lam) 

        soft_text_y = torch.zeros_like(soft_y)
        soft_text_y[torch.arange(batch_size).cuda(), random_class_indices] = 1
        soft_ya, soft_yb = soft_y, soft_text_y
    else:
        raise NotImplementedError
    

    ################# adding jusdgement, whether mixed data belong to the same label. 
    return mixed_x, soft_ya, soft_yb, lam


def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class OodMixupTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.alpha = self.config.trainer.trainer_args.alpha
        self.beta = self.config.trainer.trainer_args.beta # [0,2], if beta=0, degenerates to mixup.
        self.total_gs_num = self.config.trainer.trainer_args.total_gs_num
        self.selected_gs_num = self.config.trainer.trainer_args.selected_gs_num
        self.gs_loss_weight = self.config.trainer.trainer_args.gs_loss_weight 
        self.loss_weights = self.config.trainer.trainer_args.gs_loss_weight.split('-')
        print('loss weights', self.loss_weights)
        for i in range(len(self.loss_weights)):
            self.loss_weights[i] = float(self.loss_weights[i])

        self.gs_flag = self.config.trainer.trainer_args.gs_flag
        self.soft_split = self.config.trainer.trainer_args.soft_split
        # image domain mix: mixup | cutmix 
        # feature domain mix: manimix (manifold mixup) | geomix | mani_intercm | mani_intracm | geo_crossmodal
        # crossmodal: mix image & text data, use the current text or 'a photo of a {}' ? Yes. 
        self.mix_strategy = self.config.trainer.trainer_args.mix_strategy 
        self.loss_components = self.config.trainer.trainer_args.loss_components 
        self.text_feats = self.net.text_features # class * prompts * dim.
        self.extra_ood_class = True
        assert 0. <= self.beta
        if self.beta > 2.:
            print('pay attention that many mixed samples will be treaded as pure ood data with the beta', self.beta)
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        # for guassian modeling and resampling, we first construct a list
        self.class_query = ClassTensorQueues(class_num=self.net.n_cls, capacity=self.config.trainer.trainer_args.queue_capacity, feature_dim=512)
        self.iter_recompuration = self.config.trainer.trainer_args.iter_recompuration
        self.iter_count = self.iter_recompuration
        self.pre_queue =  self.config.trainer.trainer_args.pre_queue
    def setup(self):
        pass 

    def train_epoch(self, epoch_idx):
        # maybe we should first try to fill the class_query before the real time training. 
        if self.gs_flag and self.pre_queue and (not self.class_query.is_full()):
            self.net.eval()
            pre_epoch_idx = 0
            print('Filling the class queue with image features')
            while not self.class_query.is_full():
                train_dataiter = iter(self.train_loader)
                for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Pre Epoch {:03d}: '.format(pre_epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
                    batch = next(train_dataiter)
                    x, y = batch['data'].cuda(), batch['label'].cuda()  # 128*3*224*224; 128
                    with torch.no_grad():
                        image_features, text_features, logit_scale = self.net(x, return_feat=True)  
                    self.class_query.push(image_features, y)
                pre_epoch_idx += 1
                # print(pre_epoch_idx)
            print('class queue prepared.')

        self.net.train()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        n_cls = self.net.n_cls

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            x, y = batch['data'].cuda(), batch['label'].cuda()  # 128*3*224*224; 128
            soft_y = batch['soft_label'].cuda() # 128 * 1000
            # pdb.set_trace()
            batch_size = soft_y.size(0)
            if self.net.n_output == self.net.n_cls: # no extra ood class, ood label is the uniform ID label. 
                self.extra_ood_class=False
                ood_label = soft_y.new_ones((soft_y.size(0), soft_y.size(1))) ## 128 * 1001
                ood_label = ood_label / soft_y.size(1)
            else:
                self.extra_ood_class=True
                ood_expand = soft_y.new_zeros((batch_size, 1)) ## 128 * 1
                soft_y = torch.cat((soft_y, ood_expand), dim=1)
                ood_label = soft_y.new_zeros((soft_y.size(0), soft_y.size(1))) ## 128 * 1001
                ood_label[:,-1] = 1
            
            if self.mix_strategy == 'mixup' or self.mix_strategy == 'cutmix':
                ## image domain data mix.
                mixup_x, part_y_a, part_y_b, lam = mixup_data(x, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([x, mixup_x], dim=0)
                image_features, text_features, logit_scale = self.net(new_x, return_feat=True) # B * 1001
                logits = logit_scale * image_features @ text_features.t() 
                image_features = image_features[:batch_size,:] ## vanilla image features. 
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'vanilla':
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, 'manimix') ## not use
                logits = logit_scale * image_features @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'manimix' or self.mix_strategy == 'geomix':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'manimixrev':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'manimix_wccm' or self.mix_strategy == 'geomix_wccm':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                strategy_one, strategy_two = self.mix_strategy.split('_')
                # pdb.set_trace()
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, strategy_one)
                mixup_x_wccm, _, _, _ = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, self.mix_strategy)
                # soft_y = torch.cat([soft_y, soft_y])
                new_x = torch.cat([image_features, mixup_x_wccm, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                # cossim_id_id = logits[:batch_size*2,:1000] / logit_scale # B*C  whether use the logit_scale here is to be defined.
                cossim_id_ood = logits[:batch_size*2, 1000:] / logit_scale # B*C

                logits = [logits[:batch_size,:], logits[batch_size:batch_size*2,:], logits[batch_size*2:,:]]
                soft_ys = [soft_y, soft_y]
            elif self.mix_strategy == 'mani_cccm' or self.mix_strategy == 'geo_cccm':  # mix cross modal but within the same class [intra-class]
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C

                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'mixup_manimix_wccm' or self.mix_strategy == 'mixup_geomix_wccm':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_mix_strategy, strategy_one, strategy_two = self.mix_strategy.split('_')
                wccm_strategy = strategy_one + '_' + strategy_two
                # print(image_mix_strategy)
                mixup_x_image, part_y_a_image, part_y_b_image, lam_image = mixup_data(x, y, soft_y, self.alpha, image_mix_strategy)

                id_lam_image = 1 - self.beta * lam_image # [0,1]
                ood_lam_image = self.beta * lam_image
                mixup_y_image = ood_lam_image * ood_label + id_lam_image * (lam_image * part_y_a_image + (1-lam_image) * part_y_b_image)

                new_x_image = torch.cat([x, mixup_x_image], dim=0)
                image_features_vanilla_mixed, text_features, logit_scale = self.net(new_x_image, return_feat=True) 
                image_features, mixed_image_features = torch.chunk(image_features_vanilla_mixed, 2, dim=0)

                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, strategy_one)
                mixup_x_wccm, _, _, _ = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, wccm_strategy)
                new_x = torch.cat([image_features, mixup_x_wccm, mixed_image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                logits = [logits[:batch_size,:], logits[batch_size:batch_size*2,:], logits[batch_size*2:batch_size*3,:], logits[batch_size*3:,:]]
                soft_ys = [soft_y, soft_y, mixup_y_image]
            else:
                raise NotImplementedError
            # pdb.set_trace()
            self.class_query.push(image_features, y)
            if self.class_query.is_full() and self.gs_flag:
                 # get additional training data by Gaussian resampling. 
                if self.iter_count == self.iter_recompuration:
                    self.iter_count = 0
                    # recalculate the Gaussian mean and variance.
                    self.class_query.update_guassian()
                    syn_x, syn_y = self.class_query.sampling_guassian(num_samples=self.total_gs_num, selected_number=self.selected_gs_num, soft_y=soft_y)
                else:
                    self.iter_count += 1
                    syn_x, syn_y = self.class_query.sampling_guassian(num_samples=self.total_gs_num, selected_number=self.selected_gs_num, soft_y=soft_y)
                syn_logits = logit_scale * syn_x @ text_features.float().t()
            ############################ 判断是否为同类mix, 同类mix不作为ood??  这里碰到同类的概率很低. 已经彻底避免了该情况。
            ood_lam = self.beta * lam
            if ood_lam > 1:
                ood_lam = 1 
            id_lam = 1 - ood_lam # [0,1]
            
            mixup_y = ood_lam * ood_label + id_lam * (lam * part_y_a + (1-lam) * part_y_b)
            soft_ys.append(mixup_y)
            assert len(soft_ys) >= len(logits) ## when vanilla, soft_ys is longer than logits.
            # pdb.set_trace()
            if self.extra_ood_class: ## with extra class label. 
                loss = softmax_oodmerge(logits[0], soft_ys[0]) *  self.loss_weights[0]
                for i in range(1, len(logits)):
                    loss += softmax_oodmerge(logits[i], soft_ys[i]) *  self.loss_weights[i]
            else:
                loss = soft_cross_entropy(logits[0], soft_ys[0]) *  self.loss_weights[0]
                for i in range(1, len(logits)):
                    loss += soft_cross_entropy(logits[i], soft_ys[i]) *  self.loss_weights[i]
            
            # ########### ood sample, 熵小； id sample 熵大
            # id_sample_ood_logits = []
            # for i in range(0, len(logits)-1):
            #     id_sample_ood_logits.append(logits[i][:, n_cls:])
            #     # id_sample_ood_logits_wccm    = logits[1][:, n_cls:]
            # ood_sample_ood_logits = logits[-1][:, n_cls:]
            # # print(loss)
            # if ood_sample_ood_logits.size(1) > 1:
            #     ## more than one ood class.
            #     loss_entropy = energy(ood_sample_ood_logits, ood_lam)
            #     for i in range(len(id_sample_ood_logits)):
            #         loss_entropy += - energy(id_sample_ood_logits[i])
            #     loss += loss_entropy * self.loss_weights[-1]

            # print(loss_entropy)

            if 'abscos' in self.loss_components:

                # print('calculate abscos')
                # 对所有的 ID data, apply LSA 那篇的binary loss.  ood data 不加，因为ood data 在overall 那里有一个损失，那里约束ood score 比较大
                # max_id_cossim, _ = torch.max(cossim_id_id, dim=1)
                # pdb.set_trace()
                max_ood_cossim, _ = torch.max(cossim_id_ood, dim=1)
                # binary_loss = - torch.log(torch.sigmoid(max_id_cossim)).mean() - torch.log(1 - torch.sigmoid(max_ood_cossim)).mean()
                binary_loss = - torch.log(1 - torch.sigmoid(max_ood_cossim)).mean() # 希望ood cossim 绝对数值越小越好(比如负数)，而不是相对小就行(比如10); 绝对距离
                loss += binary_loss * self.loss_weights[len(soft_ys)]
            # new_y = torch.cat([soft_y, mixup_y], dim=0)
            # forward
            # loss = soft_cross_entropy(logits, new_y) 
            # if self.class_query.is_full() and self.gs_flag:
            #     loss += soft_cross_entropy(syn_logits, syn_y) * self.gs_loss_weight
            # print(loss)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics

############################## the cross entropy is only applied cross idood, do not distinguish between ood classes (and id classes)
############################## only for negative prompt tuning.
class OodMixupTrainer_idood:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.alpha = self.config.trainer.trainer_args.alpha
        self.beta = self.config.trainer.trainer_args.beta # [0,2], if beta=0, degenerates to mixup.
        self.total_gs_num = self.config.trainer.trainer_args.total_gs_num
        self.selected_gs_num = self.config.trainer.trainer_args.selected_gs_num
        self.gs_loss_weight = self.config.trainer.trainer_args.gs_loss_weight 
        self.loss_weights = self.config.trainer.trainer_args.gs_loss_weight.split('-')
        print('loss weights', self.loss_weights)
        for i in range(len(self.loss_weights)):
            self.loss_weights[i] = float(self.loss_weights[i])

        self.gs_flag = self.config.trainer.trainer_args.gs_flag
        self.soft_split = self.config.trainer.trainer_args.soft_split
        # image domain mix: mixup | cutmix 
        # feature domain mix: manimix (manifold mixup) | geomix | mani_intercm | mani_intracm | geo_crossmodal
        # crossmodal: mix image & text data, use the current text or 'a photo of a {}' ? Yes. 
        self.mix_strategy = self.config.trainer.trainer_args.mix_strategy 
        self.loss_components = self.config.trainer.trainer_args.loss_components 
        self.text_feats = self.net.text_features # class * prompts * dim.
        self.text_features = self.net.text_features
        self.text_features_unselected = self.net.text_features_unselected
        self.extra_ood_class = True
        assert 0. <= self.beta
        if self.beta > 2.:
            print('pay attention that many mixed samples will be treaded as pure ood data with the beta', self.beta)
        if config.optimizer.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                net.parameters(),
                config.optimizer.lr,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay,
                nesterov=True,
            )
        elif config.optimizer.name == 'adam':
            # pdb.set_trace()
            self.optimizer = torch.optim.Adam(net.parameters(), lr=config.optimizer.lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=config.optimizer.weight_decay)
        elif config.optimizer.name == 'adamw':
            ## eps=1e-8 lead to nan.
            self.optimizer = torch.optim.AdamW(net.parameters(), lr=config.optimizer.lr, betas=(0.9, 0.999), eps=1e-06, weight_decay=config.optimizer.weight_decay)
        else:
            raise NotImplementedError

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-8,
            ),
        )
        # for guassian modeling and resampling, we first construct a list
        self.class_query = ClassTensorQueues(class_num=self.net.n_cls, capacity=self.config.trainer.trainer_args.queue_capacity, feature_dim=512)
        self.iter_recompuration = self.config.trainer.trainer_args.iter_recompuration
        self.iter_count = self.iter_recompuration
        self.pre_queue =  self.config.trainer.trainer_args.pre_queue
        self.binary_loss = nn.BCELoss()
    def setup(self):
        pass 

    def train_epoch(self, epoch_idx):
        # maybe we should first try to fill the class_query before the real time training. 
        if self.gs_flag and self.pre_queue and (not self.class_query.is_full()):
            self.net.eval()
            pre_epoch_idx = 0
            print('Filling the class queue with image features')
            while not self.class_query.is_full():
                train_dataiter = iter(self.train_loader)
                for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Pre Epoch {:03d}: '.format(pre_epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
                    batch = next(train_dataiter)
                    x, y = batch['data'].cuda(), batch['label'].cuda()  # 128*3*224*224; 128
                    with torch.no_grad():
                        image_features, text_features, logit_scale = self.net(x, return_feat=True)  
                    self.class_query.push(image_features, y)
                pre_epoch_idx += 1
                # print(pre_epoch_idx)
            print('class queue prepared.')

        self.net.train()
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        n_cls = self.net.n_cls

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            x, y = batch['data'].cuda(), batch['label'].cuda()  # 128*3*224*224; 128
            binary_labels = (y  < 1000).long().float()  ## id is given label 1
            # pdb.set_trace()
            soft_y = batch['soft_label'].cuda() # 128 * 1000
            # pdb.set_trace()
            batch_size = soft_y.size(0)
            if self.net.n_output == self.net.n_cls:   # no extra ood class, ood label is the uniform ID label. 
                self.extra_ood_class=False
                ood_label = soft_y.new_ones((soft_y.size(0), soft_y.size(1))) ## 128 * 1001
                ood_label = ood_label / soft_y.size(1)
            else:
                self.extra_ood_class=True
                ood_expand = soft_y.new_zeros((batch_size, 1)) ## 128 * 1
                soft_y = torch.cat((soft_y, ood_expand), dim=1)
                ood_label = soft_y.new_zeros((soft_y.size(0), soft_y.size(1))) ## 128 * 1001
                ood_label[:,-1] = 1
            
            if self.mix_strategy == 'mixup' or self.mix_strategy == 'cutmix':
                ## image domain data mix.
                mixup_x, part_y_a, part_y_b, lam = mixup_data(x, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([x, mixup_x], dim=0)
                image_features, text_features, logit_scale = self.net(new_x, return_feat=True) # B * 1001
                logits = logit_scale * image_features @ text_features.t() 
                image_features = image_features[:batch_size,:] ## vanilla image features. 
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'vanilla':
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, 'manimix') ## not use
                logits_all = logit_scale * image_features @ text_features.t()
                cossim_id_ood = logits_all[:batch_size,1000:] / logit_scale # B*C

                binary_prob = F.softmax(logits_all, dim=1)[:, :1000].sum(1)  ## id prob.
                logits = [binary_prob]
                soft_ys = [binary_labels]   ### 因为图像是合成的，所以binary labels 可能是不对的，所以要么直接用熵减，要么对image进行filter, 把一些生成错误的样本去掉。
                # pdb.set_trace()
                # soft_ys = [soft_y]
            elif self.mix_strategy == 'wccm':
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x_wccm, part_y_a, part_y_b, lam  = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, self.mix_strategy)
                # mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, strategy_one)
                # pdb.set_trace()
                logits_all = logit_scale * mixup_x_wccm @ text_features.t()
                cossim_id_ood = logits_all[:batch_size,1000:] / logit_scale # B*C
                binary_prob = F.softmax(logits_all, dim=1)[:, :1000].sum(1)  ## id prob.
                logits = [binary_prob]
                soft_ys = [binary_labels] ### 因为图像是合成的，所以binary labels 可能是不对的，所以要么直接用熵减，要么对image进行filter, 把一些生成错误的样本去掉。

            elif self.mix_strategy == 'wccm_cd':
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x_wccm, _, _, _  = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, 'wccm')  ##within class cross modal.
                mixup_x_cd, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, 'cross_dis')
                # pdb.set_trace()
                logits_all = logit_scale * mixup_x_wccm @ text_features.t()
                logits_cd = logit_scale * mixup_x_cd @ text_features.t()

                cossim_id_ood = logits_all[:batch_size,1000:] / logit_scale # B*C
                binary_prob = F.softmax(logits_all, dim=1)[:, :1000].sum(1)  ## id prob.
                logits = [binary_prob]
                soft_ys = [binary_labels] ### 因为图像是合成的，所以binary labels 可能是不对的，所以要么直接用熵减，要么对image进行filter, 把一些生成错误的样本去掉。

            elif self.mix_strategy == 'manimix' or self.mix_strategy == 'geomix':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'manimixrev':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C
                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'manimix_wccm' or self.mix_strategy == 'geomix_wccm':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                strategy_one, strategy_two = self.mix_strategy.split('_')
                # pdb.set_trace()
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                
                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, strategy_one)
                mixup_x_wccm, _, _, _ = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, self.mix_strategy)
                # soft_y = torch.cat([soft_y, soft_y])
                new_x = torch.cat([image_features, mixup_x_wccm, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                # cossim_id_id = logits[:batch_size*2,:1000] / logit_scale # B*C  whether use the logit_scale here is to be defined.
                cossim_id_ood = logits[:batch_size*2, 1000:] / logit_scale # B*C

                logits = [logits[:batch_size,:], logits[batch_size:batch_size*2,:], logits[batch_size*2:,:]]
                soft_ys = [soft_y, soft_y]
            elif self.mix_strategy == 'mani_cccm' or self.mix_strategy == 'geo_cccm':  # mix cross modal but within the same class [intra-class]
                image_features, text_features, logit_scale = self.net(x, return_feat=True) 
                mixup_x, part_y_a, part_y_b, lam = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, self.mix_strategy)
                new_x = torch.cat([image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                cossim_id_ood = logits[:batch_size,1000:] / logit_scale # B*C

                logits = [logits[:batch_size,:], logits[batch_size:,:]]
                soft_ys = [soft_y]
            elif self.mix_strategy == 'mixup_manimix_wccm' or self.mix_strategy == 'mixup_geomix_wccm':
                ## feat domain data mix, manimix first mix, then move to sphere by L2 normlization;  geomix mix on the sphere 
                image_mix_strategy, strategy_one, strategy_two = self.mix_strategy.split('_')
                wccm_strategy = strategy_one + '_' + strategy_two
                # print(image_mix_strategy)
                mixup_x_image, part_y_a_image, part_y_b_image, lam_image = mixup_data(x, y, soft_y, self.alpha, image_mix_strategy)

                id_lam_image = 1 - self.beta * lam_image # [0,1]
                ood_lam_image = self.beta * lam_image
                mixup_y_image = ood_lam_image * ood_label + id_lam_image * (lam_image * part_y_a_image + (1-lam_image) * part_y_b_image)

                new_x_image = torch.cat([x, mixup_x_image], dim=0)
                image_features_vanilla_mixed, text_features, logit_scale = self.net(new_x_image, return_feat=True) 
                image_features, mixed_image_features = torch.chunk(image_features_vanilla_mixed, 2, dim=0)

                mixup_x, part_y_a, part_y_b, lam = mixup_data(image_features, y, soft_y, self.alpha, strategy_one)
                mixup_x_wccm, _, _, _ = mixup_data([image_features, self.text_feats], y, soft_y, self.alpha, wccm_strategy)
                new_x = torch.cat([image_features, mixup_x_wccm, mixed_image_features, mixup_x], dim=0)
                logits = logit_scale * new_x @ text_features.t()
                logits = [logits[:batch_size,:], logits[batch_size:batch_size*2,:], logits[batch_size*2:batch_size*3,:], logits[batch_size*3:,:]]
                soft_ys = [soft_y, soft_y, mixup_y_image]
            else:
                raise NotImplementedError
            # pdb.set_trace()
            # self.class_query.push(image_features, y)
            # if self.class_query.is_full() and self.gs_flag:
            #      # get additional training data by Gaussian resampling. 
            #     if self.iter_count == self.iter_recompuration:
            #         self.iter_count = 0
            #         # recalculate the Gaussian mean and variance.
            #         self.class_query.update_guassian()
            #         syn_x, syn_y = self.class_query.sampling_guassian(num_samples=self.total_gs_num, selected_number=self.selected_gs_num, soft_y=soft_y)
            #     else:
            #         self.iter_count += 1
            #         syn_x, syn_y = self.class_query.sampling_guassian(num_samples=self.total_gs_num, selected_number=self.selected_gs_num, soft_y=soft_y)
            #     syn_logits = logit_scale * syn_x @ text_features.float().t()
            ############################ 判断是否为同类mix, 同类mix不作为ood??  这里碰到同类的概率很低. 已经彻底避免了该情况。
            ood_lam = self.beta * lam
            if ood_lam > 1:
                ood_lam = 1 
            id_lam = 1 - ood_lam # [0,1]
            
            mixup_y = ood_lam * ood_label + id_lam * (lam * part_y_a + (1-lam) * part_y_b)
            soft_ys.append(mixup_y)
            assert len(soft_ys) >= len(logits) ## when vanilla, soft_ys is longer than logits.
            # pdb.set_trace()
            if self.loss_components == 'binaryce':
                loss = self.binary_loss(logits[0].float(), soft_ys[0]) *  self.loss_weights[0]
            elif self.loss_components == 'entropy':
                loss = -(logits[0].float() * torch.log(logits[0].float() + 1e-6) + (1-logits[0].float()) * torch.log(1-logits[0].float() + 1e-6)).mean()
            elif self.loss_components == 'multice':
                # pdb.set_trace()
                loss = soft_cross_entropy(logits_all, soft_y) *  self.loss_weights[0]
                if self.mix_strategy == 'wccm_cd':
                    loss += soft_cross_entropy(logits_cd, mixup_y) *  self.loss_weights[1]
            # elif self.loss_components == 'multice_cd':
            #     pdb.set_trace()
            #     loss = soft_cross_entropy(logits_all, soft_y) *  self.loss_weights[0]
            elif self.loss_components == 'textentropy':   #
                # pdb.set_trace()
                batch_size = image_features.size()[0]
                label_num, text_num = self.text_features_unselected.size()[0], self.text_features_unselected.size()[1]  ## 8W, 7
                random_indices = torch.randint(0, text_num, (batch_size,)).cuda() # sample text prompt.
                label_indices = torch.randint(0, label_num, (batch_size,)).cuda() # sample unselected negative labels.

                sampled_unselected_text_feat = self.text_features_unselected[label_indices, random_indices, :]  ## 2k*7*512 --> batch*512
                logits_unseltext = logit_scale * sampled_unselected_text_feat @ text_features.t()
                binary_prob_unseltext = F.softmax(logits_unseltext, dim=1)[:, :1000].sum(1)  ## id prob.
                loss = -(binary_prob_unseltext.float() * torch.log(binary_prob_unseltext.float() + 1e-6) + (1-binary_prob_unseltext.float()) * torch.log(1-binary_prob_unseltext.float() + 1e-6)).mean()
               
                ########## random sample a batch of text features from the unselected text features. 
                # loss = soft_cross_entropy(logits_all, soft_y) *  self.loss_weights[0]
            elif self.loss_components == 'textmultice':
                batch_size = image_features.size()[0]
                label_num, text_num = self.text_features.size()[0], self.text_features.size()[1]  ## 2k, 7
                random_indices = torch.randint(0, text_num, (batch_size,)).cuda() # sample text prompt.
                label_indices = torch.randint(0, label_num, (batch_size,)).cuda() # sample unselected negative labels.
                sampled_text_feat = self.text_features[label_indices, random_indices, :]  ## 2k*7*512 --> batch*512
                logits_text = logit_scale * sampled_text_feat @ text_features.t()
                Y_one_hot = F.one_hot(label_indices, num_classes=label_num)
                loss = soft_cross_entropy(logits_text, Y_one_hot) *  self.loss_weights[0]
            else:
                raise NotImplementedError
            if torch.isnan(text_features).any():
                pdb.set_trace()

            # pdb.set_trace()
            # if self.extra_ood_class: ## with extra class label. 
            #     loss = softmax_oodmerge(logits[0], soft_ys[0]) *  self.loss_weights[0]
            #     for i in range(1, len(logits)):
            #         loss += softmax_oodmerge(logits[i], soft_ys[i]) *  self.loss_weights[i]
            # else:
            #     loss = soft_cross_entropy(logits[0], soft_ys[0]) *  self.loss_weights[0]
            #     for i in range(1, len(logits)):
            #         loss += soft_cross_entropy(logits[i], soft_ys[i]) *  self.loss_weights[i]
            
            # ########### ood sample, 熵小； id sample 熵大
            # id_sample_ood_logits = []
            # for i in range(0, len(logits)-1):
            #     id_sample_ood_logits.append(logits[i][:, n_cls:])
            #     # id_sample_ood_logits_wccm    = logits[1][:, n_cls:]
            # ood_sample_ood_logits = logits[-1][:, n_cls:]
            # # print(loss)
            # if ood_sample_ood_logits.size(1) > 1:
            #     ## more than one ood class.
            #     loss_entropy = energy(ood_sample_ood_logits, ood_lam)
            #     for i in range(len(id_sample_ood_logits)):
            #         loss_entropy += - energy(id_sample_ood_logits[i])
            #     loss += loss_entropy * self.loss_weights[-1]

            # print(loss_entropy)

            if 'abscos' in self.loss_components:

                # print('calculate abscos')
                # 对所有的 ID data, apply LSA 那篇的binary loss.  ood data 不加，因为ood data 在overall 那里有一个损失，那里约束ood score 比较大
                # max_id_cossim, _ = torch.max(cossim_id_id, dim=1)
                # pdb.set_trace()
                max_ood_cossim, _ = torch.max(cossim_id_ood, dim=1)
                # binary_loss = - torch.log(torch.sigmoid(max_id_cossim)).mean() - torch.log(1 - torch.sigmoid(max_ood_cossim)).mean()
                binary_loss = - torch.log(1 - torch.sigmoid(max_ood_cossim)).mean() # 希望ood cossim 绝对数值越小越好(比如负数)，而不是相对小就行(比如10); 绝对距离
                loss += binary_loss * self.loss_weights[len(soft_ys)]
            # new_y = torch.cat([soft_y, mixup_y], dim=0)
            # forward
            # loss = soft_cross_entropy(logits, new_y) 
            # if self.class_query.is_full() and self.gs_flag:
            #     loss += soft_cross_entropy(syn_logits, syn_y) * self.gs_loss_weight
            # print(loss)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_lr())

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics
