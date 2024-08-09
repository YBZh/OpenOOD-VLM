# ############################################################ 
## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
############################################################ 
tau=1.0
beta=1.0
######################### clip + ood + NegLabel
prompt=nice
in_score=combine
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
        # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
        # --evaluator.name ood_clip \
                # --postprocessor.name oneoodpromptdevelop \
random_permute=False
for thres in 0.9 0.95 
do
    for group_num in 3 5 8 12 15 20 50 100 1000 1
    do
        for beta in 10.5 12 5.5 1.5
        do
            python main.py \
            --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
            --dataset.train.batch_size 8 \
            --dataset.train.few_shot 0 \
            --dataset.num_classes 11000 \
            --ood_dataset.num_classes 11000 \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number 10000 \
            --network.backbone.text_prompt ${prompt} \
            --network.backbone.text_center True \
            --network.pretrained False \
            --postprocessor.APS_mode False \
            --postprocessor.name ttapromptgroup \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.tau ${tau}  \
            --postprocessor.postprocessor_args.thres ${thres}  \
            --postprocessor.postprocessor_args.beta ${beta}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --num_gpus 1 --num_workers 6 \
            --merge_option merge \
            --output_dir ./debug_tta_neglabel_official_groupm/ \
            --mark ${in_score}_beta${beta}_thres${thres}_group_num_${group_num}_random_${random_permute}_10ktta
        done
    done
done


# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# in_score=combine
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \
#                 # --postprocessor.name oneoodpromptdevelop \
# random_permute=False
# for thres in 0.5 0.75
# do
#     for group_num in 3 5 8 12 15 20
#     do
#         for beta in 10.5 12 5.5 
#         do
#             python main.py \
#             --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#             configs/networks/fixed_clip.yml \
#             configs/pipelines/test/test_fsood.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#             --dataset.train.batch_size 8 \
#             --dataset.train.few_shot 0 \
#             --dataset.num_classes 11000 \
#             --ood_dataset.num_classes 11000 \
#             --evaluator.name ood_clip_tta \
#             --network.name fixedclip_negoodprompt \
#             --network.backbone.ood_number 10000 \
#             --network.backbone.text_prompt ${prompt} \
#             --network.backbone.text_center True \
#             --network.pretrained False \
#             --postprocessor.APS_mode False \
#             --postprocessor.name ttaprompt \
#             --postprocessor.postprocessor_args.group_num ${group_num}  \
#             --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#             --postprocessor.postprocessor_args.tau ${tau}  \
#             --postprocessor.postprocessor_args.thres ${thres}  \
#             --postprocessor.postprocessor_args.beta ${beta}  \
#             --postprocessor.postprocessor_args.in_score ${in_score}  \
#             --num_gpus 1 --num_workers 6 \
#             --merge_option merge \
#             --output_dir ./debug_tta_neglabel_official/ \
#             --mark ${in_score}_beta${beta}_thres${thres}_group_num_${group_num}_random_${random_permute}_10ktta
#         done
#     done
# done


# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# in_score=combine
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \
#                 # --postprocessor.name oneoodpromptdevelop \
# random_permute=False
# for thres in 0.5 0.75
# do
#     for group_num in 100 10 1000 1
#     do
#         for beta in  10.5 5.5 12
#         do
#             python main.py \
#             --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#             configs/networks/fixed_clip.yml \
#             configs/pipelines/test/test_fsood.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#             --dataset.train.batch_size 8 \
#             --dataset.train.few_shot 0 \
#             --dataset.num_classes 11000 \
#             --ood_dataset.num_classes 11000 \
#             --evaluator.name ood_clip_tta \
#             --network.name fixedclip_negoodprompt \
#             --network.backbone.ood_number 10000 \
#             --network.backbone.text_prompt ${prompt} \
#             --network.backbone.text_center True \
#             --network.pretrained False \
#             --postprocessor.APS_mode False \
#             --postprocessor.name ttaprompt \
#             --postprocessor.postprocessor_args.group_num ${group_num}  \
#             --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#             --postprocessor.postprocessor_args.tau ${tau}  \
#             --postprocessor.postprocessor_args.thres ${thres}  \
#             --postprocessor.postprocessor_args.beta ${beta}  \
#             --postprocessor.postprocessor_args.in_score ${in_score}  \
#             --num_gpus 1 --num_workers 6 \
#             --merge_option merge \
#             --output_dir ./debug_tta_neglabel/ \
#             --mark ${in_score}_beta${beta}_thres${thres}_group_num_${group_num}_random_${random_permute}_10ktta
#         done
#     done
# done


# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# in_score=combine
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \
#                 # --postprocessor.name oneoodpromptdevelop \

# for thres in 0.5 0.6 0.7 0.8 0.9 0.95
# do
#     for beta in 11 12 13 
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --ood_dataset.num_classes 2000 \
#         --evaluator.name ood_clip_tta \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.ood_number 1000 \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name ttaprompt \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.thres ${thres}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_tta/ \
#         --mark ${in_score}_beta${beta}_thres${thres}_tta
#     done
# done
