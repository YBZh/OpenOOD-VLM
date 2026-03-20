from typing import Any
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from transformers import pipeline

from .ants_base_postprocessor import ANTSBasePostprocessor
from openood.networks.clip_fixed_ood_prompt import imagenet_classes
from openood.networks.clip import clip
from collections import Counter
from PIL import Image
import requests
import pdb
import time
import re
import os
import random
import math
from transformers.image_utils import load_image
import copy
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration, BlipForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from transformers import BlipProcessor, BlipForQuestionAnswering, Blip2Processor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoModelForVision2Seq

# imagenet_classes = imagenet_near_classnames


############################ following nnguide!!  besides single points, using its neighbor images. 
class ANTSprocessor(ANTSBasePostprocessor):
    def __init__(self, config):
        super(ANTSprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.eta = self.args.eta
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.random_permute = self.args.random_permute
        self.class_num = None
        self.ens_stop_step = self.args.ens_stop_step

        self.batch_idx = 0
        self.ens_idx = 0
        self.ada_threshold = 0.8
        self.far_queue_max_size = 10000
        self.neglabel_init_flag = self.args.neglabel_init_flag
        self.group_num = self.args.group_num
        self.group_len = int(self.far_queue_max_size/self.group_num)

        # net attributes migrated to self
        self.all_conf_list = []
        self.conf_near_list = []
        self.conf_far_list = []
        self.path_list = []
        self.far_pred_list = []
        
        self.near_nts_features = None
        self.near_nts_list = []

        self.imagenet_features = None
        self.far_negative_feature_queue = None
        self.upper_interval = None
        self.high_freq_pred_dict = {}
        self.pred = []
        

        self.mllm_model_type = "LLAVA"
        
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        class_num = net.n_cls
        self.class_num = class_num

        self.batch_idx = 0
        self.ens_idx = 0
        self.ada_threshold = 0.8
        self.far_queue_max_size = 10000
        self.neglabel_init_flag = self.args.neglabel_init_flag
        self.group_num = self.args.group_num
        self.group_len = int(self.far_queue_max_size/self.group_num)

        # net attributes migrated to self
        self.all_conf_list = []
        self.conf_near_list = []
        self.conf_far_list = []
        self.path_list = []
        self.far_pred_list = []
        
        self.near_nts_features = None
        self.near_nts_list = []

        self.imagenet_features = None
        self.far_negative_feature_queue = None
        self.upper_interval = None
        self.high_freq_pred_dict = {}
        self.pred = []
        self.processor, self.model = self.get_model(self.mllm_model_type)
        if self.neglabel_init_flag:
            self.far_negative_feature_queue = net.text_features[:, self.class_num:].t()  
        else:
            self.far_negative_feature_queue = None
        return

    def reset_memory(self):
        self.reset = True

    def reset_group_num(self, group_num):
        self.group_num = group_num      

    def grouping_score(self, output, group_len=100):
        pos_logit = output[:, :self.class_num] ## B*C
        neg_logit = output[:, self.class_num:] ## B*total_neg_num
        group_num = int(neg_logit.size(1)/group_len)
        drop = neg_logit.size(1) % group_num
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
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], group_num, -1).contiguous()
        scores = []
        for i in range(group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        return conf_in

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, path: Any):
        net.eval()
        self.batch_idx = self.batch_idx + 1
        processor, model = self.processor, self.model

        class_num = net.n_cls
        # pdb.set_trace()
        image_features, text_features, logit_scale = net(data, return_feat=True)
        id_text_features = text_features[:class_num]

        output = logit_scale * image_features @ text_features.t() # batch * class.

        output_only_in = output[:, :class_num]
        score_only_in = torch.softmax(output_only_in, dim=1)

        _, pred_in = torch.max(score_only_in, dim=1)
        _, pred_out = torch.max(output[:, class_num:], dim=1)

        pred_in_list = [pred.item() for pred in pred_in]
        self.pred.extend(pred_in_list)

        #use MCM
        #conf_in, _ = torch.max(score_only_in, dim=1)
        #use NegLabel
        conf_in = self.grouping_score(output)

        self.all_conf_list.extend(conf_in)
        # self.ants_conf_list.extend(conf_in)

        #for ENS generation
        bins = np.arange(0, 1.1, 0.1)
        self.all_conf_list = [i.cpu() for i in self.all_conf_list]
        # self.ants_conf_list = [i.cpu() for i in self.ants_conf_list]

        threshold = self.ada_threshold
        #print("self.ada_threshold", self.ada_threshold)
        for i in range(len(conf_in)):
            if conf_in[i] < threshold:  # 判断分数是否低于 threshold
                self.path_list.append(path[i])  # 存储对应的路径
                self.far_pred_list.append(pred_in[i])

        if self.ens_idx < self.ens_stop_step:
            if len(self.path_list) > 200:
                counts, _ = np.histogram(self.all_conf_list, bins)
                differences = np.abs(np.diff(counts))
                max_diff_index = np.argmax(differences)
                upper_interval = bins[max_diff_index + 1] 
                self.upper_interval = upper_interval
                conf_neg = [conf.cpu() for conf in self.all_conf_list if conf < torch.tensor(upper_interval)]
                #conf_neg = [x for x in self.conf_list if x < upper_interval]
                percentile = self.eta*100
                #percentile = 30 use to calculate lambda
                # update self.fg_thresold
                n_threshold = np.percentile(conf_neg, percentile)
                self.ada_threshold = n_threshold

                self.far_canlabel_generation(net, processor, model)
                self.ens_idx = self.ens_idx + 1
                print("self.ens_idx", self.ens_idx)

        #for VSNL generation
        self.get_high_pred_simlabel(net, processor, model)
        self.near_nts_features = self.get_prompt_text_features(net, self.near_nts_list)
 
        if self.far_negative_feature_queue == None:
            conf_in_far = None
            balance_conf_in_far = None
        else:
            id_ens_text_features = torch.cat([id_text_features, self.far_negative_feature_queue], dim=0)
            output_far = logit_scale * image_features @ id_ens_text_features.t() # batch * class.
            if self.far_negative_feature_queue.shape[0] > self.group_len*10:
                conf_in_far = self.grouping_score(output_far)
                balance_conf_in_far = self.grouping_score(output_far, group_len=self.group_len*10)
            else:
                score_in_far = output_far.softmax(dim=-1)
                conf_in_far = score_in_far[:, :class_num].sum(dim=-1)
                balance_conf_in_far = score_in_far[:, :class_num].sum(dim=-1)
     
        # for near not maintain queue
        if self.near_nts_features!=None:
            id_vsnl_text_features = torch.cat([id_text_features, self.near_nts_features.t()], dim=0)
            balanced_id_vsnl_text_features = torch.cat(
                [id_text_features] + [self.near_nts_features.t()] * 10, dim=0
            )
            output_near = logit_scale * image_features @ id_vsnl_text_features.t() # batch * class.
            balanced_output_near = logit_scale * image_features @ balanced_id_vsnl_text_features.t() # batch * class.
            conf_in_near = self.grouping_score(output_near)
            balance_conf_in_near = self.grouping_score(balanced_output_near, group_len=self.group_len*10)
        else:
            conf_in_near = None 
            balance_conf_in_near = None
  
        conf = []
        if conf_in_far is None:
            conf_in_far = conf_in
            balance_conf_in_far = conf_in

        if conf_in_near is None:
            conf_in_near = conf_in
            balance_conf_in_near = conf_in
        # max in prob - max out prob
        if self.in_score == 'ada':
               
            self.conf_far_list.extend(balance_conf_in_far)  
            self.conf_near_list.extend(balance_conf_in_near) 

            #conf_in_his = torch.tensor(self.ants_conf_list)
            conf_in_far_his = torch.tensor(self.conf_far_list)
            conf_in_near_his = torch.tensor(self.conf_near_list)
            
            #1.using fraction ada_weight
            a = 1 - conf_in_far_his.mean()
            b = 1 - conf_in_near_his.mean()

            ada_weight = a/(a + b)
            print("ada_weight", ada_weight)

            # use ada weight
            conf = ada_weight*conf_in_far + (1-ada_weight)*conf_in_near

        elif self.in_score == 'far_only':
            conf = conf_in_far
            conf = torch.tensor(conf)
        elif self.in_score == 'near_only':
            conf = conf_in_near
            conf = torch.tensor(conf)
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

    def get_model(self, type):
        device = torch.cuda.current_device()
        if type=='BLIP':
            #processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            #model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
            model = model.to(device)
        elif type=='InstructBLIP':
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)
            model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)
        elif type=='QWEN':
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        elif type=='LLAVA':
            model_id = "llava-hf/llava-1.5-7b-hf"
            model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
            # only llava have chat template
            #model_id = "llava-hf/llava-1.5-7b-hf"
            processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True, use_flash_attention_2=True)
        elif type=='SmolVLM':
            device = torch.cuda.current_device()
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).to(device)
        elif type=='InternVL2':
            processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2-2B", torch_dtype=torch.float16)
            model = AutoModelForVision2Seq.from_pretrained("OpenGVLab/InternVL2-2B", torch_dtype=torch.float16)
        return processor, model

    def get_text_features(self, net, batch_far_labels):
        if len(batch_far_labels)!=0:
            classnames = batch_far_labels # imagenet --> imagenet class names
            with torch.no_grad():
                text_features = []
                for classname in classnames:
                    texts = clip.tokenize(classname).cuda()  # tokenize
                    class_embeddings = net.model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
                text_features = torch.stack(text_features, dim=1).cuda() # 512*1000
            return text_features
        else:
            return None   

    def get_prompt_text_features(self, net, batch_far_labels):
        if len(batch_far_labels)!=0:
            template = 'The nice {}.'
            classnames = batch_far_labels # imagenet --> imagenet class names
            with torch.no_grad():
                text_features = []
                for classname in classnames:   
                    texts = [template.format(classname)]  # format with class
                    texts = clip.tokenize(classname).cuda()  # tokenize
                    class_embeddings = net.model.encode_text(texts)  # embed with text encoder
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
                text_features = torch.stack(text_features, dim=1).cuda() # 512*1000
            return text_features
        else:
            return None          


    def far_canlabel_generation(self, net, processor, model):
        batch_far_labels = self.get_far_canlabel(net, self.path_list, self.far_pred_list, processor, model)  # 执行方法
        self.path_list.clear()  # 清空 path_list
        self.far_pred_list.clear()  # 清空 path_list
        batch_far_text_features = self.get_text_features(net, batch_far_labels)
        
        if self.far_negative_feature_queue == None:
            self.far_negative_feature_queue = batch_far_text_features.t()
        else:    
            self.far_negative_feature_queue = torch.cat(
                (batch_far_text_features.t(), self.far_negative_feature_queue), dim=0
            )
            if self.far_negative_feature_queue.shape[0] > self.far_queue_max_size:
                self.far_negative_feature_queue = self.far_negative_feature_queue[-self.far_queue_max_size:, :]

    def get_far_canlabel(self, net, path, far_pred_list, processor, model):
        filter_images = [Image.open(image_path) for image_path in path]
        id_classses = [imagenet_classes[pred] for pred in far_pred_list]
        

        if len(filter_images)!=0:
            if self.mllm_model_type == 'QWEN':
                candidate_label_list = self.get_candidate_label_list_qwen(filter_images, id_classses, processor, model)
            elif self.mllm_model_type == 'LLAVA':
                candidate_label_list = self.get_candidate_label_list_llava(filter_images, id_classses, processor, model)
            elif self.mllm_model_type == 'BLIP2':
                candidate_label_list = self.get_candidate_label_list_blip2(filter_images, id_classses, processor, model)
            elif self.mllm_model_type == 'BLIP':
                candidate_label_list = self.get_candidate_label_list_blip(filter_images, id_classses, processor, model)
            elif self.mllm_model_type == 'SmolVLM':
                candidate_label_list = self.get_candidate_label_list_smolvlm(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_llava(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_blip2(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_candidate_label_list_blip(filter_images, id_classses, processor, model)

            #candidate_label_list = self.get_candidate_label_list_smolvlm(filter_images, id_classses, processor, model)
            #candidate_label_list = self.get_far_canlabel_vllm(filter_images, id_classses, model)
            candidate_label_list = list(dict.fromkeys(candidate_label_list))
            candidate_label_list = list(set(label.rstrip('.') for label in candidate_label_list))
            return candidate_label_list
        else:
            return []
     

    def get_high_pred_simlabel(self, net, processor, model):
        sim_id_classes_num = 40
        class_counts = Counter(self.pred)
        all_counts = len(self.pred)

        filtered_counts = class_counts.most_common(sim_id_classes_num)
        high_freq_pred = [pred for pred, _ in filtered_counts]
        
        save_high_freq_pred = [pred for pred in high_freq_pred if pred not in self.high_freq_pred_dict.keys()]
        if self.batch_idx >= 10 and (self.batch_idx - 10) % 40 == 0:
        #if self.batch_idx%40==0:
            if self.mllm_model_type == 'QWEN':
                simlabel_list = self.get_simlabel_list_qwen(save_high_freq_pred, processor, model)
            elif self.mllm_model_type == 'LLAVA':
                simlabel_list = self.get_simlabel_list_llava(save_high_freq_pred, processor, model)
            elif self.mllm_model_type == 'BLIP2':
                simlabel_list = self.get_simlabel_list_blip2(save_high_freq_pred, processor, model)
            elif self.mllm_model_type == 'BLIP':
                simlabel_list = self.get_simlabel_list_blip(save_high_freq_pred, processor, model)
            elif self.mllm_model_type == 'SmolVLM':
                simlabel_list = self.get_simlabel_list_smolvlm(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_llava(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_tinyllava(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_qwen(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_blip2(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_smolvlm(save_high_freq_pred, processor, model)
            #simlabel_list = self.get_simlabel_list_instructblip(save_high_freq_pred, processor, model)
            new_items = []
            keys_to_remove = []
            for i in range(len(save_high_freq_pred)):
                new_items.append((save_high_freq_pred[i], simlabel_list[i]))
            for key, value in new_items:
                self.high_freq_pred_dict[key] = value    
            for pre_pred in self.high_freq_pred_dict.keys():
                if pre_pred not in high_freq_pred:
                    keys_to_remove.append(pre_pred)
            for key in keys_to_remove:
                self.high_freq_pred_dict.pop(key)
            self.near_nts_list = [label for sublist in self.high_freq_pred_dict.values() for label in sublist] 
            self.near_nts_list = list(dict.fromkeys(self.near_nts_list))       
        

    def get_candidate_label_list_blip2(
        self, images, id_classes, processor, model
    ):
        device = torch.cuda.current_device()
        ood_candidate_label_list = []
        prompt = "Question: Describe this image less than eight words. Answer:"
        #prompt = "Briefly describe this image. Answer:"
        with torch.no_grad():
            batch_prompts = [prompt] * len(images)
            inputs = processor(
                images=images,
                text=batch_prompts,
                return_tensors="pt",
                padding=True
            ).to(device, torch.float16)

            generated_ids = model.generate(
                **inputs,
                max_length=60
            )

            # generated_texts = processor.batch_decode(
            #     generated_ids, skip_special_tokens=True
            # )

            input_len = inputs.input_ids.shape[1]
            new_ids = generated_ids[:, input_len:]
            generated_texts = processor.batch_decode(
                new_ids, skip_special_tokens=True
            )

            ood_candidate_label_list.extend(
                [t.split("Answer:")[-1].strip() for t in generated_texts]
            )

        return ood_candidate_label_list


    def get_candidate_label_list_blip(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        model = model.to(device)
        with torch.no_grad():
            inputs = processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device, torch.float16)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30
            )
            input_len = inputs.pixel_values.shape[0]
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            ood_candidate_label_list = [t.strip() for t in generated_texts]
        return ood_candidate_label_list

    def get_candidate_label_list_qwen(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        model = model.to(device)
        # conversation = "Give me one fine-grained image label to the image, no more than five words."
        ood_candidate_label_list = []
        batch_size = 16
        # 只生成一次 template，所有图片复用
        template = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image less than eight words. Answer:"},
            ]}],
            tokenize=False, add_generation_prompt=True
        )
        template = template[0] if isinstance(template, list) else template

        with torch.no_grad():
            for batch_start in range(0, len(images), batch_size):
                batch_images = [
                    img.resize((56, 56), Image.BILINEAR)
                    for img in images[batch_start: batch_start + batch_size]
                ]
                #texts = [template] * len(batch_images)
                inputs = processor(
                    text=template, images=batch_images, padding=True, return_tensors="pt"
                ).to(device, torch.float16)

                generated_ids = model.generate(**inputs, max_new_tokens=10)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                ood_candidate_label_list.extend(output_text)
                pdb.set_trace()

        return ood_candidate_label_list     


    def get_candidate_label_list_llava(self, images, id_classses, processor, model):
        device = torch.cuda.current_device()
        model = model.to(device)
        candidate_list = []
        ood_candidate_label_list = []

        with torch.no_grad():
            for i in range(len(images)):
                raw_image = images[i]
                classname = id_classses[i]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                            {"type": "image",},
                            {"type": "text", "text": "Provide a short and concise description of this image less than eight words, don't include ###."},
                            ],
                        }
                    ]

                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
                assistant_response = processor.decode(output[0][2:], skip_special_tokens=True)
                #ood_candidate_label = assistant_response.split('ASSISTANT: ')[1].strip()
                parts = assistant_response.split('assistant\n')
                ood_candidate_label = parts[1].strip() if len(parts) > 1 else assistant_response.strip()
                ood_candidate_label_list.append(ood_candidate_label)    
        # acc
        #return
        return ood_candidate_label_list  
      


    def get_simlabel_list_llava(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        model = model.to(device)
        #classnames = [pet_names[i] for i in high_freq_pred]
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        raw_image = Image.open(requests.get(url, stream=True).raw)
        with torch.no_grad():
            for classname in tqdm(classnames):
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type":"image",},
                        {"type": "text", "text": "Give me five different class names that share similar visual features with ###, don't contain ###."},
                        ],
                    },
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classname)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                answer = processor.decode(output[0][2:], skip_special_tokens=True)
                assistant_response = answer.split('ASSISTANT: ')[1].strip()
                # 按行分割字符串
                lines = assistant_response.split('\n')
                # 提取类名
                simclass_list = []
                for line in lines:
                    # 检查行是否以数字开头并包含类名
                    if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        # 提取类名并去除前面的编号
                        class_name = line.split('. ')[1].strip() if len(line.split('. ')) > 1 else None
                        simclass_list.append(class_name)      
                candidate_list.append(simclass_list)   
        return candidate_list    
    
    

    def get_simlabel_list_qwen(self, high_freq_pred, processor, model):
        candidate_list = []
        device = torch.cuda.current_device()
        model = model.to(device)
        # conversation = "Give me one fine-grained image label to the image, no more than five words."
        classnames = [imagenet_classes[i] for i in high_freq_pred]
        # 28x28 = 1 image token，占用最小
        raw_image = Image.new("RGB", (28, 28))
        with torch.no_grad():
            for i in range(len(classnames)):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                        {"type": "image",},
                        {"type": "text", "text": "please suggest five different class names that share visual features with ###."},
                        ],
                    }
                ]
                conversation[0]["content"][1]["text"] = conversation[0]["content"][1]["text"].replace("###", classnames[i])
                text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                text=[text], images=[raw_image], padding=True, return_tensors="pt"
                )
                # image_inputs, video_inputs = process_vision_info(conversation)
                inputs = inputs.to(device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                lines = output_text[0].strip().split('\n')
                simclass_list = [
                    parts[1] for line in lines
                    if len(parts := line.split('. ', 1)) == 2
                ]
                candidate_list.append(simclass_list)   
        return candidate_list 
    
    
    def get_simlabel_list_instructblip(self, high_freq_pred, processor, model):
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        image = Image.open(requests.get(url, stream=True).raw)
        device = torch.cuda.current_device()
        candidate_list = []
        classnames = [imagenet_classes[i] for i in high_freq_pred]

        model = model.to(device)
        with torch.no_grad():
            candidate_list = []
            for classname in tqdm(classnames):
                # context = [
                # ("Give me five different class names share visual features with donut", "1.bagle 2.pastry 3.bread 4.cake 5.cookie"),
                # ]
                # template = "Question: {} Answer: {}."

                conversation = " Question: Give me five different class names share visual features with ###. Answer:"
                conversation = conversation.replace("###", classname)
                #prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + conversation
                inputs = processor(images=image, text=conversation, return_tensors="pt").to("cuda")
                outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
                )
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                simclass_list = []
                candidate_list.append(simclass_list)   
        return candidate_list