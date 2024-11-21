# ############################################################ 
## final results. AdaNeg on ImageNet
############################################################ 
tau=1.0
beta=1.0
prompt=nice
in_score=combine
random_permute=True
beta=5.5
thres=0.5
lambdaval=0.1
memleng=10
group_num=5
gap=0.5
backbone=ViT-B/16
# imagenet_traditional_four_ood
for thres in 0.5
do
    for group_num in 5 
    do
        for datasetconfig in imagenet_traditional_four_ood
        do
            python main.py \
            --config configs/datasets/imagenet/${datasetconfig}.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
            --dataset.test.batch_size 256 \
            --ood_dataset.batch_size 256 \
            --dataset.train.few_shot 0 \
            --dataset.num_classes 11000 \
            --ood_dataset.num_classes 11000 \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number 10000 \
            --network.backbone.name  ${backbone} \
            --network.backbone.text_prompt ${prompt} \
            --network.backbone.text_center True \
            --network.pretrained False \
            --postprocessor.APS_mode False \
            --postprocessor.name ttapromptnoadagap \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.tau ${tau}  \
            --postprocessor.postprocessor_args.thres ${thres}  \
            --postprocessor.postprocessor_args.gap ${gap}  \
            --postprocessor.postprocessor_args.samada False  \
            --postprocessor.postprocessor_args.memleng ${memleng}  \
            --postprocessor.postprocessor_args.lambdaval ${lambdaval}  \
            --postprocessor.postprocessor_args.beta ${beta}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --num_gpus 1 --num_workers 6 \
            --merge_option merge \
            --output_dir ./nips_reimp/ \
            --mark ${datasetconfig}_${in_score}_beta${beta}_thres${thres}_gap${gap}_memleng${memleng}_lambda${lambdaval}_group_num_${group_num}_random_${random_permute}_ood
        done
    done
done

# Accuracy 67.15%
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# Generating combined dataset with ID and OOD dataset of ssb_hard, total size 99000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 387/387 [02:53<00:00,  2.23it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 74.26, AUROC: 74.97 AUPR_IN: 75.30, AUPR_OUT: 73.28
# ACC: 65.54
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Generating combined dataset with ID and OOD dataset of ninco, total size 55879
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 219/219 [02:12<00:00,  1.65it/s]
# Computing metrics on ninco dataset...
# FPR@95: 61.23, AUROC: 78.14 AUPR_IN: 96.75, AUPR_OUT: 35.34
# ACC: 65.16
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 67.74, AUROC: 76.55 AUPR_IN: 86.03, AUPR_OUT: 54.31
# ACC: 65.35
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Generating combined dataset with ID and OOD dataset of inaturalist, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [02:48<00:00,  1.39it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 0.55, AUROC: 99.71 AUPR_IN: 99.93, AUPR_OUT: 99.08
# ACC: 65.03
# ──────────────────────────────────────────────────────────────────────
# Performing inference on sun dataset...
# Generating combined dataset with ID and OOD dataset of sun, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:01<00:00,  1.29it/s]
# Computing metrics on sun dataset...
# FPR@95: 10.62, AUROC: 97.35 AUPR_IN: 99.36, AUPR_OUT: 90.48
# ACC: 65.18
# ──────────────────────────────────────────────────────────────────────
# Performing inference on places dataset...
# Generating combined dataset with ID and OOD dataset of places, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [02:28<00:00,  1.58it/s]
# Computing metrics on places dataset...
# FPR@95: 33.92, AUROC: 94.17 AUPR_IN: 98.53, AUPR_OUT: 82.53
# ACC: 65.21
# ──────────────────────────────────────────────────────────────────────
# Performing inference on dtd dataset...
# Generating combined dataset with ID and OOD dataset of dtd, total size 55640
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [02:07<00:00,  1.72it/s]
# Computing metrics on dtd dataset...
# FPR@95: 26.70, AUROC: 95.56 AUPR_IN: 99.39, AUPR_OUT: 80.42
# ACC: 65.32
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 17.95, AUROC: 96.70 AUPR_IN: 99.30, AUPR_OUT: 88.13
# ACC: 65.18




tau=1.0
beta=1.0
prompt=nice
in_score=combine
random_permute=True
beta=5.5
thres=0.5
lambdaval=0.1
memleng=10
group_num=5
gap=0.5
backbone=ViT-B/16
# imagenet_traditional_four_ood
for thres in 0.5
do
    for group_num in 5 
    do
        for datasetconfig in imagenet_traditional_four_ood
        do
            python main.py \
            --config configs/datasets/imagenet/${datasetconfig}.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
            --dataset.test.batch_size 256 \
            --ood_dataset.batch_size 256 \
            --dataset.train.few_shot 0 \
            --dataset.num_classes 11000 \
            --ood_dataset.num_classes 11000 \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number 10000 \
            --network.backbone.name  ${backbone} \
            --network.backbone.text_prompt ${prompt} \
            --network.backbone.text_center True \
            --network.pretrained False \
            --postprocessor.APS_mode False \
            --postprocessor.name ttapromptnoadagap \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.tau ${tau}  \
            --postprocessor.postprocessor_args.thres ${thres}  \
            --postprocessor.postprocessor_args.gap ${gap}  \
            --postprocessor.postprocessor_args.samada True  \
            --postprocessor.postprocessor_args.memleng ${memleng}  \
            --postprocessor.postprocessor_args.lambdaval ${lambdaval}  \
            --postprocessor.postprocessor_args.beta ${beta}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --num_gpus 1 --num_workers 6 \
            --merge_option merge \
            --output_dir ./nips_reimp/ \
            --mark ${datasetconfig}_${in_score}_beta${beta}_thres${thres}_gap${gap}_memleng${memleng}_lambda${lambdaval}_group_num_${group_num}_random_${random_permute}_ood_sampada
        done
    done
done

# Accuracy 67.44%
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# Generating combined dataset with ID and OOD dataset of ssb_hard, total size 99000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 387/387 [03:36<00:00,  1.79it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 74.71, AUROC: 74.85 AUPR_IN: 75.47, AUPR_OUT: 72.87
# ACC: 66.33
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Generating combined dataset with ID and OOD dataset of ninco, total size 55879
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 219/219 [02:17<00:00,  1.59it/s]
# Computing metrics on ninco dataset...
# FPR@95: 61.68, AUROC: 77.95 AUPR_IN: 96.71, AUPR_OUT: 37.41
# ACC: 66.25
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 68.19, AUROC: 76.40 AUPR_IN: 86.09, AUPR_OUT: 55.14
# ACC: 66.29
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Generating combined dataset with ID and OOD dataset of inaturalist, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [02:40<00:00,  1.46it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 0.71, AUROC: 99.71 AUPR_IN: 99.93, AUPR_OUT: 98.96
# ACC: 66.02
# ──────────────────────────────────────────────────────────────────────
# Performing inference on sun dataset...
# Generating combined dataset with ID and OOD dataset of sun, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:17<00:00,  1.19it/s]
# Computing metrics on sun dataset...
# FPR@95: 10.99, AUROC: 97.23 AUPR_IN: 99.34, AUPR_OUT: 89.25
# ACC: 66.10
# ──────────────────────────────────────────────────────────────────────
# Performing inference on places dataset...
# Generating combined dataset with ID and OOD dataset of places, total size 60000
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [02:36<00:00,  1.50it/s]
# Computing metrics on places dataset...
# FPR@95: 34.38, AUROC: 94.10 AUPR_IN: 98.52, AUPR_OUT: 81.58
# ACC: 66.12
# ──────────────────────────────────────────────────────────────────────
# Performing inference on dtd dataset...
# Generating combined dataset with ID and OOD dataset of dtd, total size 55640
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [02:14<00:00,  1.63it/s]
# Computing metrics on dtd dataset...
# FPR@95: 28.84, AUROC: 94.85 AUPR_IN: 99.30, AUPR_OUT: 75.29
# ACC: 66.20
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 18.73, AUROC: 96.47 AUPR_IN: 99.27, AUPR_OUT: 86.27
# ACC: 66.11




