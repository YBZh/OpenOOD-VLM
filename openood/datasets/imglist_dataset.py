import ast
import io
import logging
import os

import torch
from PIL import Image, ImageFile
from collections import defaultdict

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pdb
import random

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 few_shot=0, 
                 randseed=0,
                 repeat=True,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)
        self.name = name

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        self.few_shot = few_shot
        # pdb.set_trace()
        imglist_pth = imglist_pth.split('///')
        print('load dataset from', imglist_pth[0])
        with open(imglist_pth[0]) as imgfile:
            self.imglist = imgfile.readlines()
        if self.few_shot != 0:
            # creat the few-shot dataset.
            tracker = self.split_dataset_by_label(self.imglist)
            dataset = []
            random.seed(randseed)
            for label, items in tracker.items():
                if len(items) >= few_shot:
                    sampled_items = random.sample(items, few_shot)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=few_shot)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)
            self.imglist = dataset
        for i in range(1,len(imglist_pth)): 
            print('load aux dataset from', imglist_pth[i])
            with open(imglist_pth[i]) as imgfile:
                aux_imglist = imgfile.readlines()
            self.imglist = self.imglist + aux_imglist


        # pdb.set_trace()
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def split_dataset_by_label(self, imglist):
        data_splited = defaultdict(list)
        for index in range(len(imglist)):
            line = imglist[index].strip('\n')
            tokens = line.split(' ', 1)
            image_name, extra_str = tokens[0], tokens[1]
            data_splited[extra_str].append(imglist[index])
        return data_splited

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
