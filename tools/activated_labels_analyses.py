import torch
import pdb

candidate_label_list = torch.load('text_cancidate_list.pth', weights_only=True)

# ssb_hard_act = torch.load('activation_id_ood_ssb_hard.pth',  map_location=torch.device('cpu'))
# ninco_act = torch.load('activation_id_ood_ninco.pth',  map_location=torch.device('cpu'))

# inaturalist_act = torch.load('activation_id_ood_ninco.pth',  map_location=torch.device('cpu'))
# sun_act = torch.load('activation_id_ood_sun.pth',  map_location=torch.device('cpu'))
# places_act = torch.load('activation_id_ood_places.pth',  map_location=torch.device('cpu'))
# dtd_act = torch.load('activation_id_ood_dtd.pth',  map_location=torch.device('cpu'))

# dataset_list = ['ssb_hard', 'ninco', 'inaturalist', 'sun', 'places', 'dtd']
dataset_list = ['inaturalist', 'dtd']
for dataset in dataset_list:
    actscore = torch.load('activation_id_ood_'+ dataset + '.pth',  map_location=torch.device('cpu'), weights_only=True)
    score_from_id = - actscore['id']
    score_from_ood = actscore['ood']
    combined_score = score_from_id
    s2l_value, index = torch.sort(combined_score)
    print(dataset, 'top 3:', ", ".join([
        f"{candidate_label_list[index[-1]]}: {s2l_value[-1]}",
        f"{candidate_label_list[index[-2]]}: {s2l_value[-2]}",
        f"{candidate_label_list[index[-3]]}: {s2l_value[-3]}"
    ]))

    print(dataset, 'bottom 3:', ", ".join([
        f"{candidate_label_list[index[0]]}: {s2l_value[0]}",
        f"{candidate_label_list[index[1]]}: {s2l_value[1]}",
        f"{candidate_label_list[index[2]]}: {s2l_value[2]}"
    ]))
    # pdb.set_trace()