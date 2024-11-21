
# # ############################################################ 
############################################################ 
tau=1.0
beta=1.0
######################### setting thres=1.0 in AdaNeg --> NegLabel.
prompt=simple
in_score=combine
random_permute=True
beta=12
gap=0.5
for thres in 1.0
do
    for group_num in 5
    do
        for datasetconfig in covid_ood_idood
        do
            python main.py \
            --config configs/datasets/covid/${datasetconfig}.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.train.batch_size 8 \
            --dataset.train.few_shot 0 \
            --dataset.num_classes 1002 \
            --ood_dataset.num_classes 1002 \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number 1000 \
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
            --postprocessor.postprocessor_args.beta ${beta}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --num_gpus 1 --num_workers 6 \
            --merge_option merge \
            --output_dir ./nips_rebuttal_covid/ \
            --mark neglabel_${datasetconfig}${in_score}_beta${beta}_thres${thres}_gap${gap}_group_num_${group_num}_random_${random_permute}_ood
        done
    done
done

# Computing metrics on ct dataset...  nice prompt
# FPR@95: 100.00, AUROC: 83.13 AUPR_IN: 92.04, AUPR_OUT: 86.36
# ACC: 47.03
# ──────────────────────────────────────────────────────────────────────
# Performing inference on xraybone dataset...
# Generating combined dataset with ID and OOD dataset of xraybone, total size 1091
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:03<00:00,  1.65it/s]
# Computing metrics on xraybone dataset...
# FPR@95: 0.34, AUROC: 99.97 AUPR_IN: 99.99, AUPR_OUT: 99.88
# ACC: 47.03
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 50.17, AUROC: 91.55 AUPR_IN: 96.02, AUPR_OUT: 93.12
# ACC: 47.03


# Processing nearood...   simple prompt.
# Performing inference on ct dataset...
# Generating combined dataset with ID and OOD dataset of ct, total size 1381
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:04<00:00,  1.50it/s]
# Computing metrics on ct dataset...
# FPR@95: 100.00, AUROC: 63.53 AUPR_IN: 85.50, AUPR_OUT: 72.42
# ACC: 45.68
# ──────────────────────────────────────────────────────────────────────
# Performing inference on xraybone dataset...
# Generating combined dataset with ID and OOD dataset of xraybone, total size 1091
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:04<00:00,  1.49it/s]
# Computing metrics on xraybone dataset...
# FPR@95: 0.56, AUROC: 99.68 AUPR_IN: 99.93, AUPR_OUT: 99.48
# ACC: 45.68
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 50.28, AUROC: 81.61 AUPR_IN: 92.71, AUPR_OUT: 85.95
# ACC: 45.68

# ############################################################ 
## abla thres and gap
############################################################ 
tau=1.0
beta=1.0
######################### AdaNeg.
prompt=simple
in_score=combine
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_ood.yml \
#  --config configs/datasets/imagenet/imagenet_train_fsood.yml \
        # --evaluator.name ood_clip_tta \    ## 这个的结果和ood_clip 的结果相同，说明将ID data eval 放到每个ood data paired 是合理的。
        # --evaluator.name ood_clip \
                # --postprocessor.name oneoodpromptdevelop \

# sun397_classes = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'outdoor apartment_building', 'indoor apse', 'aquarium', 'aqueduct', 'arch', 'archive', 'outdoor arrival_gate', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'outdoor athletic_field', 'public atrium', 'attic', 'auditorium', 'auto_factory', 'badlands', 'indoor badminton_court', 'baggage_claim', 'shop bakery', 'exterior balcony', 'interior balcony', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'outdoor basketball_court', 'bathroom', 'batters_box', 'bayou', 'indoor bazaar', 'outdoor bazaar', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'indoor bistro', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'indoor booth', 'botanical_garden', 'indoor bow_window', 'outdoor bow_window', 'bowling_alley', 'boxing_ring', 'indoor brewery', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'outdoor cabin', 'cafeteria', 'campsite', 'campus', 'natural canal', 'urban canal', 'candy_store', 'canyon', 'backseat car_interior', 'frontseat car_interior', 'carrousel', 'indoor casino', 'castle', 'catacomb', 'indoor cathedral', 'outdoor cathedral', 'indoor cavern', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'indoor chicken_coop', 'outdoor chicken_coop', 'childs_room', 'indoor church', 'outdoor church', 'classroom', 'clean_room', 'cliff', 'indoor cloister', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'outdoor control_tower', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'exterior covered_bridge', 'creek', 'crevasse', 'crosswalk', 'office cubicle', 'dam', 'delicatessen', 'dentists_office', 'sand desert', 'vegetation desert', 'indoor diner', 'outdoor diner', 'home dinette', 'vehicle dinette', 'dining_car', 'dining_room', 'discotheque', 'dock', 'outdoor doorway', 'dorm_room', 'driveway', 'outdoor driving_range', 'drugstore', 'electrical_substation', 'door elevator', 'interior elevator', 'elevator_shaft', 'engine_room', 'indoor escalator', 'excavation', 'indoor factory', 'fairway', 'fastfood_restaurant', 'cultivated field', 'wild field', 'fire_escape', 'fire_station', 'indoor firing_range', 'fishpond', 'indoor florist_shop', 'food_court', 'broadleaf forest', 'needleleaf forest', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'indoor garage', 'garbage_dump', 'gas_station', 'exterior gazebo', 'indoor general_store', 'outdoor general_store', 'gift_shop', 'golf_course', 'indoor greenhouse', 'outdoor greenhouse', 'indoor gymnasium', 'indoor hangar', 'outdoor hangar', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'outdoor hot_tub', 'outdoor hotel', 'hotel_room', 'house', 'outdoor hunting_lodge', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'indoor ice_skating_rink', 'outdoor ice_skating_rink', 'iceberg', 'igloo', 'industrial_area', 'outdoor inn', 'islet', 'indoor jacuzzi', 'indoor jail', 'jail_cell', 'jewelry_shop', 'kasbah', 'indoor kennel', 'outdoor kennel', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'outdoor labyrinth', 'natural lake', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'indoor library', 'outdoor library', 'outdoor lido_deck', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'indoor market', 'outdoor market', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'water moat', 'outdoor monastery', 'indoor mosque', 'outdoor mosque', 'motel', 'mountain', 'mountain_snowy', 'indoor movie_theater', 'indoor museum', 'music_store', 'music_studio', 'outdoor nuclear_power_plant', 'nursery', 'oast_house', 'outdoor observatory', 'ocean', 'office', 'office_building', 'outdoor oil_refinery', 'oilrig', 'operating_room', 'orchard', 'outdoor outhouse', 'pagoda', 'palace', 'pantry', 'park', 'indoor parking_garage', 'outdoor parking_garage', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'indoor pilothouse', 'outdoor planetarium', 'playground', 'playroom', 'plaza', 'indoor podium', 'outdoor podium', 'pond', 'establishment poolroom', 'home poolroom', 'outdoor power_plant', 'promenade_deck', 'indoor pub', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'indoor shopping_mall', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'baseball stadium', 'football stadium', 'indoor stage', 'staircase', 'street', 'subway_interior', 'platform subway_station', 'supermarket', 'sushi_bar', 'swamp', 'indoor swimming_pool', 'outdoor swimming_pool', 'indoor synagogue', 'outdoor synagogue', 'television_studio', 'east_asia temple', 'south_asia temple', 'indoor tennis_court', 'outdoor tennis_court', 'outdoor tent', 'indoor_procenium theater', 'indoor_seats theater', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'outdoor track', 'train_railway', 'platform train_station', 'tree_farm', 'tree_house', 'trench', 'coral_reef underwater', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'indoor volleyball_court', 'outdoor volleyball_court', 'waiting_room', 'indoor warehouse', 'water_tower', 'block waterfall', 'fan waterfall', 'plunge waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'barrel_storage wine_cellar', 'bottle_storage wine_cellar', 'indoor wrestling_ring', 'yard', 'youth_hostel']

random_permute=True
beta=5.5
thres=0.5
lambdaval=0.1
memleng=10
group_num=5
gap=0.0
backbone=ViT-B/16
# imagenet_traditional_four_ood
for thres in 0.5
do
    for group_num in 5 
    do
        for datasetconfig in  covid_ood_idood
        do
            python main.py \
            --config configs/datasets/covid/${datasetconfig}.yml \
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
            --output_dir ./nips_rebuttal_covid/ \
            --mark ${datasetconfig}_${in_score}_beta${beta}_thres${thres}_gap${gap}_memleng${memleng}_lambda${lambdaval}_group_num_${group_num}_random_${random_permute}_ood_sampada
        done
    done
done


# Performing inference on ct dataset...  nice prompt
# Generating combined dataset with ID and OOD dataset of ct, total size 1381
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:05<00:00,  1.05it/s]
# Computing metrics on ct dataset...
# FPR@95: 100.00, AUROC: 86.43 AUPR_IN: 93.43, AUPR_OUT: 87.51
# ACC: 47.25
# ──────────────────────────────────────────────────────────────────────
# Performing inference on xraybone dataset...
# Generating combined dataset with ID and OOD dataset of xraybone, total size 1091
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00,  1.01s/it]
# Computing metrics on xraybone dataset...
# FPR@95: 0.00, AUROC: 100.00 AUPR_IN: 100.00, AUPR_OUT: 99.99
# ACC: 46.46
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 50.00, AUROC: 93.21 AUPR_IN: 96.72, AUPR_OUT: 93.75
# ACC: 46.86


# Performing inference on ct dataset...   simple prompt.
# Generating combined dataset with ID and OOD dataset of ct, total size 1381
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:05<00:00,  1.04it/s]
# Computing metrics on ct dataset...
# FPR@95: 100.00, AUROC: 93.48 AUPR_IN: 96.59, AUPR_OUT: 93.96
# ACC: 46.58
# ──────────────────────────────────────────────────────────────────────
# Performing inference on xraybone dataset...
# Generating combined dataset with ID and OOD dataset of xraybone, total size 1091
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:05<00:00,  1.04s/it]
# Computing metrics on xraybone dataset...
# FPR@95: 0.11, AUROC: 99.99 AUPR_IN: 100.00, AUPR_OUT: 99.94
# ACC: 46.58
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 50.06, AUROC: 96.74 AUPR_IN: 98.29, AUPR_OUT: 96.95
# ACC: 46.58