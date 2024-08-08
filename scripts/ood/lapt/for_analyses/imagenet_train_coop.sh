#!/bin/bash
# sh scripts/ood/coop/imagenet_train_coop.sh

################################### ablate ood prompt number and ood entropy loss.
# mix_strategy=mixup
# mix_strategy=cutmix
# mix_strategy=manimix
# mix_strategy=geomix
# mix_strategy=manimix_wccm
# mix_strategy=geomix_wccm
# mix_strategy=mixup_manimix_wccm
# mix_strategy=mixup_geomix_wccm
mix_strategy=wccm

depth=5
alpha=1
beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
total_gs_num=1000
loss_components=multice
gs_loss_weight=1-1-1-1
OOD_NUM=1000
shot_num=16
weight_decay=0.00001
optimizer=sgd
N_CTX=8
# N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
############ learning rate 比较大的时候需要比较多的
for seed in 0 10
do
    for lr in 0.01 0.001 
    do
        for weight_decay in 0.01 0.1 0 0.00001 0.0001 0.001 
        do
            python main.py \
            --config configs/datasets/imagenet/imagenet_train_fsood.yml \
            configs/networks/coop.yml \
            configs/pipelines/train/train_coop.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --network.name coop_negoodprompt \
            --network.backbone.OOD_NUM ${OOD_NUM} \
            --network.backbone.text_prompt tip \
            --network.pretrained False \
            --network.backbone.CSC False \
            --network.backbone.N_CTX ${N_CTX} \
            --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
            --trainer.trainer_args.alpha ${alpha} \
            --trainer.trainer_args.beta ${beta} \
            --trainer.trainer_args.mix_strategy ${mix_strategy} \
            --trainer.trainer_args.total_gs_num ${total_gs_num} \
            --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
            --trainer.trainer_args.loss_components ${loss_components} \
            --trainer.trainer_args.gs_flag False \
            --trainer.trainer_args.queue_capacity 500 \
            --trainer.trainer_args.pre_queue True \
            --trainer.trainer_args.iter_recompuration 10 \
            --trainer.trainer_args.soft_split True \
            --evaluator.name ood_clip \
            --trainer.name oodmixup_idood \
            --postprocessor.name oneoodprompt \
            --postprocessor.APS_mode True \
            --postprocessor.postprocessor_args.tau 1 \
            --postprocessor.postprocessor_args.in_score oodscore \
            --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2_dedup.txt \
            --dataset.interpolation bilinear \
            --dataset.train.batch_size 32 \
            --dataset.num_classes 2000 \
            --dataset.train.few_shot 0 \
            --seed ${seed} \
            --optimizer.name ${optimizer} \
            --optimizer.num_epochs 10 \
            --optimizer.lr ${lr} \
            --optimizer.weight_decay ${weight_decay} \
            --num_gpus 1 --num_workers 4 \
            --merge_option merge \
            --output_dir ./debug_oodval_dedup_scratch_nctx${N_CTX}/ \
            --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
        done
    done
done

# ################################### ablate ood prompt number and ood entropy loss.
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# mix_strategy=wccm

# depth=5
# alpha=1
# beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
# total_gs_num=1000
# loss_components=multice
# gs_loss_weight=1-1-1-1
# OOD_NUM=1000
# shot_num=16
# weight_decay=0.
# optimizer=adam
# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# # ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# for seed in 10 2024
# do
#     for lr in 0.1 0.01 0.001 0.0001 0.00001 0.000001
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.backbone.CTX_INIT 'a photo of a' \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#             --trainer.trainer_args.soft_split True \
#             --evaluator.name ood_clip \
#             --trainer.name oodmixup_idood \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 32 \
#             --dataset.num_classes 2000 \
#             --dataset.train.few_shot 0 \
#             --seed ${seed} \
#             --optimizer.name ${optimizer} \
#             --optimizer.num_epochs 10 \
#             --optimizer.lr ${lr} \
#             --optimizer.weight_decay ${weight_decay} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./debug/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
#     done
# done


# ################################### ablate ood prompt number and ood entropy loss.
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# mix_strategy=vanilla

# depth=5
# alpha=1
# beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-0.1-1-1
# OOD_NUM=1000
# shot_num=16

# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# # ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# for seed in 1 2024
# do
#     for lr in 0.00001 0.0001 0.001 0.01 0.1
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.backbone.CTX_INIT 'a photo of a' \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#             --trainer.trainer_args.soft_split True \
#             --evaluator.name ood_clip \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.num_classes 2000 \
#             --dataset.train.few_shot 0 \
#             --seed ${seed} \
#             --optimizer.num_epochs 50 \
#             --optimizer.lr ${lr} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./debug_synneg_pt/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_lr${lr}
#     done
# done



# ################################### ablate ood prompt number and ood entropy loss.
# mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# # mix_strategy=manimix

# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-1-1-1
# OOD_NUM=1000
# shot_num=16

# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# for seed in 0 1 2024
# do
#     for lr in 0.1 0.01 0.001 0.0001 0.00001 0.000001
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.backbone.CTX_INIT 'a photo of a' \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#             --trainer.trainer_args.soft_split True \
#             --evaluator.name ood_clip \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot ${shot_num} \
#             --seed ${seed} \
#             --optimizer.num_epochs 60 \
#             --optimizer.lr ${lr} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./neg_prompt/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_lr${lr}
#     done
# done



# tau=1.0
# beta=1.0
# prompt=tip
# for in_score in guiscore_img_mul guiscore_img_norm guiscore_img_add guiscore_img_cosonly
# do
#     for beta in 1 2 3 5 10 40 100 200 500 1000 3000
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn2_filtered_acc_pred.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 11000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_image/ \
#         --mark ${in_score}_beta${beta}
#     done
# done


#################################### ablate ood prompt number and ood entropy loss.
# mix_strategy=mixup
# mix_strategy=cutmix
# mix_strategy=manimix
# mix_strategy=geomix
# mix_strategy=manimix_wccm
# mix_strategy=geomix_wccm
# mix_strategy=mixup_manimix_wccm
# mix_strategy=mixup_geomix_wccm
# mix_strategy=manimix

# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-1-1-1
# OOD_NUM=1000
# shot_num=16

# for seed in 0 1 2
# do
#     for OOD_NUM in 10 100 1000
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.backbone.N_CTX 4 --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name mcm \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot ${shot_num} \
#             --seed ${seed} \
#             --optimizer.num_epochs 100 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./neg_prompt/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}
#     done
# done


# #################################### ablate ood prompt number and ood entropy loss.
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm

# depth=5
# alpha=1
# beta=0
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-1-1-1
# OOD_NUM=1
# shot_num=16

# for seed in 0 1 2 
# do
#     for beta in 0 1 2
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name mcm \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot ${shot_num} \
#             --seed ${seed} \
#             --optimizer.num_epochs 100 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./coop_mix/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}
#     done
# done

# ###################################################### mixup + coop.
# for seed in 2 1 0
# do
#     python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/coop.yml \
#         configs/pipelines/train/train_coop.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --trainer.trainer_args.alpha 0.2 \
#         --trainer.name mixup \
#         --postprocessor.postprocessor_args.tau 1  \
#         --network.pretrained False \
#         --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#         --dataset.train.batch_size 128 \
#         --optimizer.num_epochs 150 \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./mixrerun/ \
#         --mark coop_mixup \
#         --seed ${seed}
# done

# for seed in 2 1 0
# do
#     python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/coop.yml \
#         configs/pipelines/train/train_coop.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --trainer.trainer_args.beta 1.0 \
#         --trainer.trainer_args.cutmix_prob 1.0 \
#         --trainer.name cutmix \
#         --postprocessor.postprocessor_args.tau 1  \
#         --network.pretrained False \
#         --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#         --dataset.train.batch_size 128 \
#         --optimizer.num_epochs 150 \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./mixrerun/ \
#         --mark coop_cutmix \
#         --seed ${seed}
# done


# # ###################################################### cutmix + coop.


# ################################################### vanilla coop.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/coop.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --dataset.train.few_shot 16 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --output_dir ./prompt_tuning/ \
#     --merge_option merge \
#     --seed 0

# python main.py \
# --config configs/datasets/imagenet/imagenet_train_fsood.yml \
# configs/networks/coop.yml \
# configs/pipelines/test/test_fsood.yml \
# configs/preprocessors/base_preprocessor.yml \
# configs/postprocessors/mcm.yml \
# --network.backbone.text_prompt tip \
# --network.pretrained True \
# --network.checkpoint ./prompt_tuning/imagenet_coop_base_sgd_e150_lr0.1_default/s0/best.ckpt \
# --postprocessor.postprocessor_args.tau 10  \
# --postprocessor.postprocessor_args.in_score sum  \
# --dataset.train.batch_size 128 \
# --dataset.train.few_shot 16 \
# --num_gpus 1 --num_workers 6 \
# --merge_option merge \
# --output_dir ./test/ \
# --mark alpha${alpha}_beta${beta}_${mix_strategy}_test 

# ######################### clip + MCM.
# python main.py \
# --config configs/datasets/imagenet/imagenet_train_fsood.yml \
# configs/networks/fixed_clip.yml \
# configs/pipelines/test/test_fsood.yml \
# configs/preprocessors/base_preprocessor.yml \
# configs/postprocessors/mcm.yml \
# --network.backbone.text_prompt simple \
# --network.pretrained False \
# --postprocessor.postprocessor_args.tau 3  \
# --postprocessor.postprocessor_args.in_score sum  \
# --dataset.train.batch_size 128 \
# --dataset.train.few_shot 16 \
# --num_gpus 1 --num_workers 6 \
# --merge_option merge \
# --output_dir ./test/ \
# --mark alpha${alpha}_beta${beta}_${mix_strategy}_test 



# #################################### ablate ood prompt number and ood entropy loss.
# mix_strategy=manimix_wccm
# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-1-1-1
# for N_CTX in 16 8 4 2 1
# do
#     for OOD_NUM in 10
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt \
#             --network.name coop_oneoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.N_CTX ${N_CTX} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC True \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 150 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_ntx_num/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth}_oodnum${OOD_NUM}_${gs_loss_weight}_N_CTX${N_CTX}_csc \
#             --seed 0
#     done
# done

# #################################### ablate ood prompt number and ood entropy loss.
# mix_strategy=manimix
# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# # 
# for gs_loss_weight in 1-1-1
# do
#     for OOD_NUM in 1 2 3 5 10 20 50 100 1000
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/maple.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt \
#             --network.name maple_oneoodprompt \
#             --network.backbone.PROMPT_DEPTH ${depth} \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 60 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_oodnum/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth}_oodnum${OOD_NUM}_${gs_loss_weight} \
#             --seed 0
#     done
# done



# mix_strategy=manimix_wccm
# alpha=1
# beta=1
# loss_components=classify
# # 
# # 
# for gs_loss_weight in 1-1-1-1
# do
#     for train_data in train_syn_imagenet_sdxl_gs5 train_syn_imagenet_sdxl_sd3_gs5 train_syn_imagenet_sdxl_sd3_gs75
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/${train_data}.txt\
#             --network.name coop_oneoodprompt \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num 100 \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#              --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split False \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 100 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_synmix_imagenet/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_${train_data}_mixdataset \
#             --seed 0
#     done
# done

# for gs_loss_weight in 1-1-1-1 1-1-1-0.1 1-1-1-0.01 1-1-1-5  
# do
#     for loss_components in abscos
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_oneoodprompt \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num 100 \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#             --trainer.trainer_args.soft_split False \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 100 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_rerun_coop/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_total_gs_num${total_gs_num}_lossw${gs_loss_weight} \
#             --seed 0
#     done
# done

# ###################################################### mani_cc_cm + coop_oneoodprompt.
# mix_strategy=mani_cc_cm
# for alpha in 1
# do
#     for beta in 0.2 1 5 10 50 200
#     do
#         # python main.py \
#         # --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # configs/networks/coop.yml \
#         # configs/pipelines/test/test_fsood.yml \
#         # configs/preprocessors/base_preprocessor.yml \
#         # configs/postprocessors/mcm.yml \
#         # --network.name coop_oneoodprompt \
#         # --network.backbone.text_prompt tip \
#         # --network.pretrained True \
#         # --network.checkpoint ./exp_feat_mix/imagenet_coop_oneoodprompt_oodmixup_sgd_e100_lr0.1_alpha${alpha}_beta${beta}_${mix_strategy}/s0/best.ckpt \
#         # --postprocessor.name oneoodprompt \
#         # --postprocessor.postprocessor_args.tau 1  \
#         # --postprocessor.postprocessor_args.in_score sum  \
#         # --dataset.train.batch_size 128 \
#         # --dataset.train.few_shot 16 \
#         # --num_gpus 1 --num_workers 6 \
#         # --merge_option merge \
#         # --output_dir ./exp_cc_cm/ \
#         # --mark alpha${alpha}_beta${beta}_${mix_strategy}_test 
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_oneoodprompt \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 100 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 6 \
#             --merge_option merge \
#             --output_dir ./exp_cc_cm/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy} \
#             --seed 0
#     done
# done





################################# mixup + ood coop
# for alpha in 0.2 1. 20.
# do
#     for beta in 0 1. 2.
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_oneoodprompt \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 150 \
#             --num_gpus 1 --num_workers 6 \
#             --merge_option merge \
#             --output_dir ./coop_oodmixup/ \
#             --mark alpha${alpha}_beta${beta}_scoresum \
#             --seed 0
#     done
# done