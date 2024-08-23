# OpenOOD-VLM: Benchmarking Generalized OOD Detection with Vision-Language Models

This repository release the **official codes** of our works for out-of-distribution detection with **Vision-Language Models**. For reference, we also reproduce some representative methods of OOD detection.

## 🔥 News
- **23 Aug, 2024**: [LAPT](https://arxiv.org/pdf/2407.08966) is accepted by ECCV 2024, you can easily reproduce our results with [Bash](scripts/ood/lapt/test.sh) and [Checkpoint](https://drive.google.com/file/d/1uHvqAraQDBuc_IjbqyLgSdjFDxesTAA7/view?usp=sharing)


## ⚙️ Dependencies and Installation
<details>
  <summary>Follow OpenOOD to set up the environment, or use our provided instructions below.</summary>

  pip install git+https://github.com/YBZH/OpenOOD-VLM

</details>

## 📷 Datasets
We also follow OpenOOD to manage the training and testing datasets.
If you only use our evaluator, the benchmarks for evaluation will be automatically downloaded by the evaluator (again check out this [tutorial](https://colab.research.google.com/drive/1tvTpCM1_ju82Yygu40fy7Lc0L1YrlkQF?usp=sharing)). If you would like to also use OpenOOD-VLM for training, you can get all data with our [downloading script](https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download). Note that ImageNet-1K training images should be downloaded from its official website.

Besides datasets used in OpenOOD, we also provide evaluation on some popular OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618) curated by [Huang et al. 2021](https://arxiv.org/abs/2105.01879). Please follow instruction from the this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the subsampled datasets where semantically overlapped classes with ImageNet-1k are removed.

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` by default.
```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```

<details>
<summary><b>Supported Datasets for Out-of-Distribution Detection</b></summary>

> - [x] [BIMCV (A COVID X-Ray Dataset)]()
>      > Near-OOD: `CT-SCAN`, `X-Ray-Bone`;<br>
>      > Far-OOD: `MNIST`, `CIFAR-10`, `Texture`, `Tiny-ImageNet`;<br>
> - [x] [MNIST]()
>      > Near-OOD: `NotMNIST`, `FashionMNIST`;<br>
>      > Far-OOD: `Texture`, `CIFAR-10`, `TinyImageNet`, `Places365`;<br>
> - [x] [CIFAR-10]()
>      > Near-OOD: `CIFAR-100`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - [x] [CIFAR-100]()
>      > Near-OOD: `CIFAR-10`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - [x] [ImageNet-200]()
>      > Near-OOD: `SSB-hard`, `NINCO`;<br>
>      > Far-OOD: `iNaturalist`, `Texture`, `OpenImage-O`;<br>
>      > Covariate-Shifted ID: `ImageNet-C`, `ImageNet-R`, `ImageNet-v2`;
> - [x] [ImageNet-1K]()
>      > Near-OOD: `SSB-hard`, `NINCO`;<br>
>      > Far-OOD: `iNaturalist`, `Texture`, `OpenImage-O`;<br>
>      > Covariate-Shifted ID: `ImageNet-C`, `ImageNet-R`, `ImageNet-v2`;
> - [x] [ImageNet-1K Traditional Four Datasets]()
>      > Far-OOD: `iNaturalist`, `SUN`, `Places`, `Texture`;<br>
>      > Covariate-Shifted ID: `ImageNet-C`, `ImageNet-R`, `ImageNet-v2`;
</details>

## 🚀 Training/Testing
We provide training and evaluation scripts for all the methods we support in [scripts folder](https://github.com/YBZH/OpenOOD-VLM/tree/main/scripts).

- [`ECCV2024 Official`] [LAPT]((https://arxiv.org/pdf/2407.08966)) with pre-trained [Checkpoint](https://drive.google.com/file/d/1uHvqAraQDBuc_IjbqyLgSdjFDxesTAA7/view?usp=sharing)

```sh
sh scripts/ood/lapt/test.sh
```

- [NIPS2022 Reproduced] Zero-shot [MCM](https://arxiv.org/pdf/2211.13445)
```sh
sh scripts/ood/mcm/official.sh
```

- [ICLR2024 Reproduced] Zero-shot [NegLabel](https://arxiv.org/pdf/2403.20078)
```sh
sh scripts/ood/neglabel/official.sh
```

<!-- We release the **official codes** of our works for OOD detection with Vision-Language Models, including:
> - [x] [ECCV2024: LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models](https://arxiv.org/pdf/2407.08966) with [Bash](scripts/ood/lapt/test.sh) and [Checkpoint](https://drive.google.com/file/d/1uHvqAraQDBuc_IjbqyLgSdjFDxesTAA7/view?usp=sharing)
</details>

We will also reproduce some interesting works for OOD detection with Vision-Language Models, including:
> - [x] [NIPS2022: Delving into out-of-distribution detection with vision-language representations](https://arxiv.org/pdf/2211.13445) with [Bash](scripts/ood/mcm/official.sh)
> - [x] [ICLR2024: Negative Label Guided OOD Detection with Pretrained Vision-Language Models](https://arxiv.org/pdf/2403.20078)  with [Bash](scripts/ood/neglabel/official.sh)
</details> -->
















## 📖 Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex

@article{zhang2024lapt,
  title={LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models},
  author={Zhang, Yabin and Zhu, Wenjie and He, Chenhang and Zhang, Lei},
  journal={Proceedings of the european conference on computer vision (ECCV)},
  year={2024}
}


```

## 🙏 Acknowledgement

- [OpenOOD](https://github.com/Jingkang50/OpenOOD), an extensible codebase with Vision Models only.




