
# # AAneg not work on the Cifar datasets, do not know why.
# tau=1.0

# ######################### clip + ood + NegLabel
# prompt=nice
# random_permute=False
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \ 1000 100 5000 10000  imagenet_train_ood_ssbhard
# in_score=sum
# group_num=5
# group_size=1000 ## activate only when group_num=-1
# ood_number=10000
# thres=0.5 ## threshold for mean; maybe we can also use the standard deviation. 
# gap=0.2
# # alpha=1.0     threshold for value / max.200 100 1000 2000 5000 10000 20000  0.95 0.9 0.8 0.7 0.5 0.3 0.2 0.1   alpha 等于1的结果不正常，难道这么大的扰动是随机噪声？？ 不可能，0.8 0.7 0.5 0.3 0.2 0.1
# beta=1000 
# gamma=0
# cluster_num=0 ## whether use generated negative labels 100 300 500 800 1000 2000 3000 5000 10000 20000
# alpha=0.95  ## controal distribution vs. batch activation, alpha=1 means using distribution activations. fixing 0.95 to use both 
# memleng=300
# #  imagenet_temshift_spti imagenet_temshift_ptis imagenet_temshift_tisp  
# for gap in 0.0   ## number of used labels. 可能generated labels 在few negative label num 的情况下有效。
# do
#     for beta in  10 100 1000 10000 30000 60000 #  2000 5000 10000 20000 50000 100000 200000 250000   ## activated-aware scores, 0 means activation-agnostics  ptis tisp  ###part_of_speech_tagging, neglabel_wordnet  fixedclip_negoodprompt_fullcorpus
#     do
#         python main.py \
#         --config configs/datasets/cifar10/cifar10_clip.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/aaneg.yml \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes $((ood_number + 10)) \
#         --ood_dataset.num_classes $((ood_number + 10)) \
#         --evaluator.name ood_clip_tta \
#         --evaluator.ood_scheme ood \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.corpus neglabel_wordnet \
#         --network.backbone.name ViT-B/16 \
#         --network.backbone.ood_number ${ood_number} \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name actneg \
#         --postprocessor.postprocessor_args.cluster_num ${cluster_num} \
#         --postprocessor.postprocessor_args.memleng ${memleng} \
#         --postprocessor.postprocessor_args.thres ${thres}  \
#         --postprocessor.postprocessor_args.gap ${gap}  \
#         --postprocessor.postprocessor_args.group_num ${group_num}  \
#         --postprocessor.postprocessor_args.group_size ${group_size}  \
#         --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.alpha ${alpha}  \
#         --postprocessor.postprocessor_args.gamma ${gamma}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --postprocessor.postprocessor_args.cossim False  \
#         --num_gpus 1 --num_workers 8 \
#         --merge_option merge \
#         --output_dir ./aaneg_analyses_cifar10/ \
#         --mark ${in_score}_alpha${alpha}_beta${beta}_gamma${gamma}_neg10k_ood_number_${ood_number}_random_${random_permute}_thres${thres}_gap${gap}_gsize${group_size}_memlen${memleng}
#     done
# done

# # #####cifar 10 & cifar 100
# tau=1.0

# ######################### clip + ood + NegLabel
# prompt=nice
# random_permute=False
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \ 1000 100 5000 10000  imagenet_train_ood_ssbhard
# in_score=sum
# group_num=5
# group_size=1000 ## activate only when group_num=-1
# ood_number=10000
# thres=0.5 ## threshold for mean; maybe we can also use the standard deviation. 
# gap=0.2
# # alpha=1.0     threshold for value / max.200 100 1000 2000 5000 10000 20000  0.95 0.9 0.8 0.7 0.5 0.3 0.2 0.1   alpha 等于1的结果不正常，难道这么大的扰动是随机噪声？？ 不可能，0.8 0.7 0.5 0.3 0.2 0.1
# beta=1000 
# gamma=0
# cluster_num=0 ## whether use generated negative labels 100 300 500 800 1000 2000 3000 5000 10000 20000
# alpha=0.95  ## controal distribution vs. batch activation, alpha=1 means using distribution activations. fixing 0.95 to use both 
# memleng=300
# #  imagenet_temshift_spti imagenet_temshift_ptis imagenet_temshift_tisp  
# for gap in 0.0   ## number of used labels. 可能generated labels 在few negative label num 的情况下有效。
# do
#     for beta in 10 100 1000 10000 30000 60000 #  2000 5000 10000 20000 50000 100000 200000 250000   ## activated-aware scores, 0 means activation-agnostics  ptis tisp  ###part_of_speech_tagging, neglabel_wordnet  fixedclip_negoodprompt_fullcorpus
#     do
#         python main.py \
#         --config configs/datasets/cifar100/cifar100_clip.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/aaneg.yml \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes $((beta + 100)) \
#         --ood_dataset.num_classes $((beta + 100)) \
#         --evaluator.name ood_clip_tta \
#         --evaluator.ood_scheme ood \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.corpus neglabel_wordnet \
#         --network.backbone.name ViT-B/16 \
#         --network.backbone.ood_number ${beta} \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name actneg \
#         --postprocessor.postprocessor_args.cluster_num ${cluster_num} \
#         --postprocessor.postprocessor_args.memleng ${memleng} \
#         --postprocessor.postprocessor_args.thres ${thres}  \
#         --postprocessor.postprocessor_args.gap ${gap}  \
#         --postprocessor.postprocessor_args.group_num ${group_num}  \
#         --postprocessor.postprocessor_args.group_size ${group_size}  \
#         --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.alpha ${alpha}  \
#         --postprocessor.postprocessor_args.gamma ${gamma}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --postprocessor.postprocessor_args.cossim False  \
#         --num_gpus 1 --num_workers 8 \
#         --merge_option merge \
#         --output_dir ./aaneg_analyses_cifar100/ \
#         --mark ${in_score}_alpha${alpha}_beta${beta}_gamma${gamma}_neg10k_ood_number_${ood_number}_random_${random_permute}_thres${thres}_gap${gap}_gsize${group_size}_memlen${memleng}
#     done
# done


# # #####cifar 10 & cifar 100
# tau=1.0

# ######################### clip + ood + NegLabel
# prompt=nice
# random_permute=False
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \ 1000 100 5000 10000  imagenet_train_ood_ssbhard
# in_score=sum
# group_num=5
# group_size=1000 ## activate only when group_num=-1
# ood_number=10000
# thres=0.5 ## threshold for mean; maybe we can also use the standard deviation. 
# gap=0.2
# # alpha=1.0     threshold for value / max.200 100 1000 2000 5000 10000 20000  0.95 0.9 0.8 0.7 0.5 0.3 0.2 0.1   alpha 等于1的结果不正常，难道这么大的扰动是随机噪声？？ 不可能，0.8 0.7 0.5 0.3 0.2 0.1
# beta=1000 
# gamma=1
# cluster_num=0 ## whether use generated negative labels 100 300 500 800 1000 2000 3000 5000 10000 20000
# alpha=0.95  ## controal distribution vs. batch activation, alpha=1 means using distribution activations. fixing 0.95 to use both 
# memleng=300
# #  imagenet_temshift_spti imagenet_temshift_ptis imagenet_temshift_tisp  
# for gap in 0.0 0.2 0.5   ## number of used labels. 可能generated labels 在few negative label num 的情况下有效。
# do
#     for beta in  10 100 500 1000 5000 10000 30000 60000 #  2000 5000 10000 20000 50000 100000 200000 250000   ## activated-aware scores, 0 means activation-agnostics  ptis tisp  ###part_of_speech_tagging, neglabel_wordnet  fixedclip_negoodprompt_fullcorpus
#     do
#         python main.py \
#         --config configs/datasets/cifar10/cifar10_clip.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/aaneg.yml \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes $((ood_number + 10)) \
#         --ood_dataset.num_classes $((ood_number + 10)) \
#         --evaluator.name ood_clip_tta \
#         --evaluator.ood_scheme ood \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.corpus neglabel_wordnet \
#         --network.backbone.name ViT-B/16 \
#         --network.backbone.ood_number ${ood_number} \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name actneg \
#         --postprocessor.postprocessor_args.cluster_num ${cluster_num} \
#         --postprocessor.postprocessor_args.memleng ${memleng} \
#         --postprocessor.postprocessor_args.thres ${thres}  \
#         --postprocessor.postprocessor_args.gap ${gap}  \
#         --postprocessor.postprocessor_args.group_num ${group_num}  \
#         --postprocessor.postprocessor_args.group_size ${group_size}  \
#         --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.alpha ${alpha}  \
#         --postprocessor.postprocessor_args.gamma ${gamma}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --postprocessor.postprocessor_args.cossim False  \
#         --num_gpus 1 --num_workers 8 \
#         --merge_option merge \
#         --output_dir ./aaneg_analyses_cifar10/ \
#         --mark ${in_score}_alpha${alpha}_beta${beta}_gamma${gamma}_neg10k_ood_number_${ood_number}_random_${random_permute}_thres${thres}_gap${gap}_gsize${group_size}_memlen${memleng}
#     done
# done

# #####cifar 10 & cifar 100
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
gap=0.2
# alpha=1.0     threshold for value / max.200 100 1000 2000 5000 10000 20000  0.95 0.9 0.8 0.7 0.5 0.3 0.2 0.1   alpha 等于1的结果不正常，难道这么大的扰动是随机噪声？？ 不可能，0.8 0.7 0.5 0.3 0.2 0.1
beta=1000 
gamma=1
cluster_num=0 ## whether use generated negative labels 100 300 500 800 1000 2000 3000 5000 10000 20000
alpha=0.95  ## controal distribution vs. batch activation, alpha=1 means using distribution activations. fixing 0.95 to use both 
memleng=300
#  imagenet_temshift_spti imagenet_temshift_ptis imagenet_temshift_tisp  
for thres in 0.01 0.1 0.3 0.5  ## number of used labels. 可能generated labels 在few negative label num 的情况下有效。
do
    for beta in 10 100 500 1000 5000 10000 30000 60000 #  2000 5000 10000 20000 50000 100000 200000 250000   ## activated-aware scores, 0 means activation-agnostics  ptis tisp  ###part_of_speech_tagging, neglabel_wordnet  fixedclip_negoodprompt_fullcorpus
    do
        python main.py \
        --config configs/datasets/cifar100/cifar100_clip.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/aaneg.yml \
        --dataset.train.batch_size 128 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes $((beta + 100)) \
        --ood_dataset.num_classes $((beta + 100)) \
        --evaluator.name ood_clip_tta \
        --evaluator.ood_scheme ood \
        --network.name fixedclip_negoodprompt \
        --network.backbone.corpus neglabel_wordnet \
        --network.backbone.name ViT-B/16 \
        --network.backbone.ood_number ${beta} \
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
        --output_dir ./aaneg_analyses_cifar100_fixedthres/ \
        --mark ${in_score}_alpha${alpha}_beta${beta}_gamma${gamma}_neg10k_ood_number_${ood_number}_random_${random_permute}_thres${thres}_gap${gap}_gsize${group_size}_memlen${memleng}
    done
done
