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
for thres in 0.5 0.6
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



# ###########################
# # 先用 local feature + global feature 做一个，验证local feature 和global feature 的互补性，直接
# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# in_score=sum
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \
#                 # --postprocessor.name oneoodpromptdevelop \   localglobal_multiply localglobal_add  sum
# beta=10.5
# thres=1.0
# # 
# for localindice in 1
# do
#     for in_score in localonly 
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
#         --network.name fixedclip_negoodprompt_localfeat \
#         --network.backbone.ood_number 1000 \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name ttapromptlocalfeat \
#         --postprocessor.postprocessor_args.tau ${tau} \
#         --postprocessor.postprocessor_args.localindice ${localindice} \
#         --postprocessor.postprocessor_args.thres ${thres}\
#         --postprocessor.postprocessor_args.beta ${beta} \
#         --postprocessor.postprocessor_args.in_score ${in_score} \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_tta_local_mostsim/ \
#         --mark ${in_score}_beta${beta}_thres${thres}_lindice_${localindice}_tta
#     done
# done

# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.
# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# in_score=sum
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \
#                 # --postprocessor.name oneoodpromptdevelop \
# for thres in 0.51
# do
#     for beta in 1
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
#         --output_dir ./debug/ \
#         --mark ${in_score}_beta${beta}_thres${thres}_tta
#     done
# done



# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
#         # --evaluator.name ood_clip \

# for in_score in sum
# do
#     for beta in 1
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
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}_tta
#     done
# done



# # ############################################################ 
# ## 尝试用 10*1000 或者 100*100 groups vote 得到， following NegLabel.

# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         # --evaluator.name ood_clip_tta \
#         # --evaluator.name ood_clip \
# for in_score in sum
# do
#     for beta in 1
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
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.ood_number 1000 \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done



# ############################################################  Final, zero-shot performance.
## 这些都是只基于 farest 1000 OOD text names 得到的结果，并不是最终的结果，最终的结果需要用 10*1000 或者 100*100 groups vote 得到。

# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Start evaluation...
# # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.53it/s]

# # Accuracy 66.77%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.50it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:49<00:00,  3.91it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 77.29, AUROC: 73.14 AUPR_IN: 71.28, AUPR_OUT: 73.32
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.69it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 59.39, AUROC: 80.76 AUPR_IN: 96.88, AUPR_OUT: 42.25
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 68.34, AUROC: 76.95 AUPR_IN: 84.08, AUPR_OUT: 57.79
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.38it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 1.46, AUROC: 99.57 AUPR_IN: 99.90, AUPR_OUT: 98.36
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:10<00:00,  2.06it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 44.34, AUROC: 88.17 AUPR_IN: 98.38, AUPR_OUT: 44.34
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.70it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 36.59, AUROC: 92.04 AUPR_IN: 96.81, AUPR_OUT: 83.65
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 27.46, AUROC: 93.26 AUPR_IN: 98.36, AUPR_OUT: 75.45
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────

# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  Neglabel.
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name fsood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.51it/s]

# # Accuracy 66.77%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.52it/s]
# # Performing inference on imagenetv2 dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:18<00:00,  2.20it/s]
# # Performing inference on imagenetc dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:11<00:00,  3.62it/s]
# # Performing inference on imagenetr dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:48<00:00,  2.42it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Computing accuracy on imagenetv2 dataset...
# # CSID[imagenetv2] accuracy: 60.90%
# # Computing accuracy on imagenetc dataset...
# # CSID[imagenetc] accuracy: 39.68%
# # Computing accuracy on imagenetr dataset...
# # CSID[imagenetr] accuracy: 56.67%
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:47<00:00,  4.05it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 83.91, AUROC: 68.56 AUPR_IN: 79.66, AUPR_OUT: 54.32
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.66it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 68.59, AUROC: 76.99 AUPR_IN: 98.05, AUPR_OUT: 29.16
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 76.25, AUROC: 72.77 AUPR_IN: 88.86, AUPR_OUT: 41.74
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.41it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 1.25, AUROC: 99.56 AUPR_IN: 99.95, AUPR_OUT: 97.46
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:09<00:00,  2.13it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 53.59, AUROC: 85.97 AUPR_IN: 99.02, AUPR_OUT: 27.97
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.74it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 45.04, AUROC: 90.54 AUPR_IN: 98.02, AUPR_OUT: 73.66
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 33.30, AUROC: 92.02 AUPR_IN: 98.99, AUPR_OUT: 66.36
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────