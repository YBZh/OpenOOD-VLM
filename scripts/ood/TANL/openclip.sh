




# ############################################################ 
tau=1.0

######################### clip + ood + NegLabel
prompt=nice
random_permute=False
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \ 1000 100 5000 10000  imagenet_train_ood_ssbhard
in_score=sum
group_num=5
group_size=1000 ## activate only when group_num=-1
ood_number=10000
thres=0.5 ## threshold for mean; maybe we can also use the standard deviation. 
gap=0.5
# alpha=1.0     threshold for value / max.200 100 1000 2000 5000 10000 20000  0.95 0.9 0.8 0.7 0.5 0.3 0.2 0.1   alpha 等于1的结果不正常，难道这么大的扰动是随机噪声？？ 不可能，0.8 0.7 0.5 0.3 0.2 0.1
beta=1000 
gamma=1
cluster_num=0 ## whether use generated negative labels 100 300 500 800 1000 2000 3000 5000 10000 20000
alpha=0.0  ## controal distribution vs. batch activation, alpha=1 means using distribution activations. fixing 0.95 to use both; 如果是transductive 的话，alpha=0 可能更好。
memleng=300



for seed in 1  ## number of used labels. 可能generated labels 在few negative label num 的情况下有效。
do
    for gamma in 1  #  2000 5000 10000 20000 50000 100000 200000 250000   ## activated-aware scores, 0 means activation-agnostics  ptis tisp  ###part_of_speech_tagging, neglabel_wordnet  fixedclip_negoodprompt_fullcorpus
    do
        python main.py \
        --config configs/datasets/imagenet/imagenet_train_ood.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/aaneg.yml \
        --dataset.train.batch_size 128 \
        --dataset.test.batch_size 256 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes $((ood_number + 1000)) \
        --ood_dataset.num_classes $((ood_number + 1000)) \
        --evaluator.name ood_clip_tta \
        --evaluator.ood_scheme ood \
        --network.name fixedclip_negoodprompt_openclip \
        --network.backbone.corpus neglabel_wordnet \
        --network.backbone.name ViT-B/16 \
        --network.backbone.ood_number ${ood_number} \
        --network.backbone.text_prompt ${prompt} \
        --network.backbone.text_center True \
        --network.pretrained False \
        --postprocessor.APS_mode False \
        --postprocessor.name actneg \
        --postprocessor.postprocessor_args.cluster_num ${cluster_num} \
        --postprocessor.postprocessor_args.memleng ${memleng} \
        --postprocessor.postprocessor_args.thres ${thres}  \
        --postprocessor.postprocessor_args.gap ${gap}  \
        --postprocessor.postprocessor_args.group_num ${group_num}  \
        --postprocessor.postprocessor_args.group_size ${group_size}  \
        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
        --postprocessor.postprocessor_args.tau ${tau}  \
        --postprocessor.postprocessor_args.beta ${beta}  \
        --postprocessor.postprocessor_args.alpha ${alpha}  \
        --postprocessor.postprocessor_args.gamma ${gamma}  \
        --postprocessor.postprocessor_args.in_score ${in_score}  \
        --postprocessor.postprocessor_args.cossim False  \
        --num_gpus 1 --num_workers 8 \
        --merge_option merge \
        --output_dir ./aaneg_analyses_cvpr26/ \
        --mark testnumber001_${in_score}_alpha${alpha}_beta${beta}_gamma${gamma}_neg10k_ood_number_${ood_number}_random_${random_permute}_thres${thres}_gap${gap}_gsize${group_size}_seed${seed}
    done
done
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# Generating combined dataset with ID and OOD dataset of ssb_hard, total size 94000
#   0%|                                                                                                                                                                                                                        | 0/368 [00:00<?, ?it/s]cached_pos_features shape: torch.Size([1000, 512])
# cached_neg_features shape: torch.Size([15, 512])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 368/368 [01:56<00:00,  3.15it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 61.21, AUROC: 84.70 AUPR_IN: 83.02, AUPR_OUT: 84.97
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Generating combined dataset with ID and OOD dataset of ninco, total size 50879
#   0%|                                                                                                                                                                                                                        | 0/199 [00:00<?, ?it/s]cached_pos_features shape: torch.Size([1000, 512])
# cached_neg_features shape: torch.Size([15, 512])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [01:12<00:00,  2.76it/s]
# Computing metrics on ninco dataset...
# FPR@95: 70.18, AUROC: 78.94 AUPR_IN: 96.26, AUPR_OUT: 36.28
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 65.69, AUROC: 81.82 AUPR_IN: 89.64, AUPR_OUT: 60.63
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Generating combined dataset with ID and OOD dataset of inaturalist, total size 55000
#   0%|                                                                                                                                                                                                                        | 0/215 [00:00<?, ?it/s]cached_pos_features shape: torch.Size([1000, 512])
# cached_neg_features shape: torch.Size([15, 512])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 215/215 [01:28<00:00,  2.42it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 0.65, AUROC: 99.72 AUPR_IN: 99.94, AUPR_OUT: 98.51
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# Generating combined dataset with ID and OOD dataset of textures, total size 50160
#   0%|                                                                                                                                                                                                                        | 0/196 [00:00<?, ?it/s]cached_pos_features shape: torch.Size([1000, 512])
# cached_neg_features shape: torch.Size([15, 512])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [01:10<00:00,  2.78it/s]
# Computing metrics on textures dataset...
# FPR@95: 24.08, AUROC: 94.73 AUPR_IN: 99.31, AUPR_OUT: 69.24
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# Generating combined dataset with ID and OOD dataset of openimageo, total size 60869
#   0%|                                                                                                                                                                                                                        | 0/238 [00:00<?, ?it/s]cached_pos_features shape: torch.Size([1000, 512])
# cached_neg_features shape: torch.Size([15, 512])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 238/238 [01:22<00:00,  2.88it/s]
# Computing metrics on openimageo dataset...
# FPR@95: 36.56, AUROC: 92.85 AUPR_IN: 97.00, AUPR_OUT: 85.82
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 20.43, AUROC: 95.77 AUPR_IN: 98.75, AUPR_OUT: 84.52
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
