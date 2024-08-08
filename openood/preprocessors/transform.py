import torchvision.transforms as tvs_trans

normalization_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'clip': [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]], # clip default
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],  # ImageNet default. 
    'imagenet200': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],  # ImageNet default. 
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'aircraft': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cub': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
}

interpolation_modes = {
    'nearest': tvs_trans.InterpolationMode.NEAREST,
    'bilinear': tvs_trans.InterpolationMode.BILINEAR,
    'bicubic':tvs_trans.InterpolationMode.BICUBIC, 
}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


# More transform classes shall be written here
