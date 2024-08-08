import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .base_preprocessor import BasePreprocessor
from .transform import Convert


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config):
        super(TestStandardPreProcessor, self).__init__(config)
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.CenterCrop(self.image_size),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

            # Compose(
    # Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
    # CenterCrop(size=(224, 224))
    # <function _convert_image_to_rgb at 0x7f7fd7caa1f0>
    # ToTensor()
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
