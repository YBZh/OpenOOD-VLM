from typing import Any

from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
import pdb


def distance(penultimate, target, metric='inner_product'):

    return torch.cosine_similarity(penultimate, target, dim=1)

    # if metric == 'inner_product':
    #     return torch.sum(torch.mul(penultimate, target), dim=1)
    # elif metric == 'euclidean':
    #     return -torch.sqrt(torch.sum((penultimate - target)**2, dim=1))
    # elif metric == 'cosine':
    #     return torch.cosine_similarity(penultimate, target, dim=1)
    # else:
    #     raise ValueError('Unknown metric: {}'.format(metric))


class SHEOneOodPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(SHEOneOodPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            all_activation_log = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Eval: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label']
                    all_labels.append(deepcopy(labels))

                    # logits, features = net(data, return_feature=True)
                    image_features, text_features, logit_scale = net(data, return_feat=True)
                    logits = logit_scale * image_features @ text_features.t() # batch * class.
                    logits = logits[:, :1000]
                    all_activation_log.append(image_features.cpu())
                    all_preds.append(logits.argmax(1).cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_activation_log = torch.cat(all_activation_log)

            self.activation_log = []
            for i in range(self.num_classes):
                # mask = torch.logical_and(all_labels == i, all_preds == i) ## 取出gt 和 pred 都一致的样本来计算class center. 奇怪，training data 讲道理pred 和 GT是一致的，但是这里有很多不一致
                mask = all_labels == i
                class_correct_activations = all_activation_log[mask]
                self.activation_log.append(
                    class_correct_activations.mean(0, keepdim=True))

            self.activation_log = torch.cat(self.activation_log).cuda()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # output, feature = net(data, return_feature=True)
        # pred = output.argmax(1)
        image_features, text_features, logit_scale = net(data, return_feat=True)
        logits = logit_scale * image_features @ text_features.t() # batch * class.
        pred = logits[:, :1000].argmax(1)
        # conf = distance(image_features, self.activation_log[pred], self.args.metric)  ### 只有这个结果很差 89/75 远差于-conf_out 的79/50
        conf = -distance(image_features, self.activation_log[pred], self.args.metric)  ### 只有这个结果 90.35/92   也很差，奇怪，也就是这个对ID or OOD 没有判别能力？

        # score = torch.softmax(logits, dim=1)
        # conf_out = torch.sum(score[:, 1000:], dim=1)
        # conf = conf * (- conf_out)
        # pdb.set_trace()
        if torch.isnan(conf).any():
            pdb.set_trace()
        return pred, conf

