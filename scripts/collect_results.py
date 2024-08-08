import re
import sys
import os
import ipdb
import numpy as np
import glob
def extract_metrics(file_path, metrics):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found. Please make sure the file path is correct.")
        return metrics
    except Exception as e:
        print(f"An error occurred: {e}")
        return metrics
    print(file_path)
    if '\nCompleted!' in text:
        if 'Complete Evaluation, accuracy' in text:
            accuracy_pattern_id = r"Complete Evaluation, accuracy ([0-9.]+)"  ##training time
        else:
            accuracy_pattern_id = r"Accuracy (\d+\.\d+)%"  ## only test.
        
        accuracy_pattern_csid_v2 = r"CSID\[imagenetv2\] accuracy: (\d+\.\d+)%"
        accuracy_pattern_csid_c = r"CSID\[imagenetc\] accuracy: (\d+\.\d+)%"
        accuracy_pattern_csid_r = r"CSID\[imagenetr\] accuracy: (\d+\.\d+)%"

        accuracy_pattern = r"ACC: ([0-9.]+)"
        # fpr_pattern = r"FPR@95: ([0-9.]+)"
        # auroc_pattern = r"AUROC: ([0-9.]+)"
        # aupr_in_pattern = r"AUPR_IN: ([0-9.]+)"
        # aupr_out_pattern = r"AUPR_OUT: ([0-9.]+)"


        nearood_pattern = r"Processing nearood...*?Computing mean metrics...\nFPR@95: (\d+\.\d+), AUROC: (\d+\.\d+) AUPR_IN: (\d+\.\d+), AUPR_OUT: (\d+\.\d+)\nACC: (\d+\.\d+)"
        farood_pattern = r"Processing farood...*?Computing mean metrics...\nFPR@95: (\d+\.\d+), AUROC: (\d+\.\d+) AUPR_IN: (\d+\.\d+), AUPR_OUT: (\d+\.\d+)\nACC: (\d+\.\d+)"
        nearood_match = re.search(nearood_pattern, text, re.DOTALL)
        farood_match = re.search(farood_pattern, text, re.DOTALL)

        accuracy_matches_id = re.findall(accuracy_pattern_id, text)
        accuracy_matches_csid_v2 = re.findall(accuracy_pattern_csid_v2, text)
        accuracy_matches_csid_c = re.findall(accuracy_pattern_csid_c, text)
        accuracy_matches_csid_r = re.findall(accuracy_pattern_csid_r, text)
        # accuracy_matches = re.findall(accuracy_pattern, text)
        # ipdb.set_trace()
        metrics['near_fpr95'].append(float(nearood_match[1]))
        metrics['near_auroc'].append(float(nearood_match[2]))
        metrics['near_auprin'].append(float(nearood_match[3]))
        metrics['near_auprout'].append(float(nearood_match[4]))
        metrics['far_fpr95'].append(float(farood_match[1]))
        metrics['far_auroc'].append(float(farood_match[2]))
        metrics['far_auprin'].append(float(farood_match[3]))
        metrics['far_auprout'].append(float(farood_match[4]))
        metrics['allacc'].append(float(nearood_match[5]))
        if (len(accuracy_matches_csid_v2) > 0):
            metrics['id_acc'].append(float(accuracy_matches_id[0]))
            metrics['csid_v2'].append(float(accuracy_matches_csid_v2[0]))
            metrics['csid_c'].append(float(accuracy_matches_csid_c[0]))
            metrics['csid_r'].append(float(accuracy_matches_csid_r[0]))
        else:
            if 'id_acc' in metrics.keys():
                del metrics['id_acc']
                del metrics['csid_v2']
                del metrics['csid_c']
                del metrics['csid_r']
    else:
        pass
    return metrics

def print_metrics(metrics):
    max_key_length = max(len(key) for key in metrics.keys())
    for key in metrics.keys():
        print(f"{key:>{max_key_length}}", end=' ')
    print()  # 换行
    # ipdb.set_trace()
    for values in metrics.values():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{mean_value:>{max_key_length - 6}.2f}±{std_value:<5.2f}", end=' ')
        # print(f"{mean_value:>9.2f}±{std_value:<5.2f}", end=' ')
    print()  # 换行

def print_metrics_multi(metrics_multi):
    first_flag=True
    for path in metrics_multi.keys():
        metrics = metrics_multi[path]
        # ipdb.set_trace()
        max_key_length = max(len(key) for key in metrics.keys())
        if first_flag:
            for key in metrics.keys():
                print(f"{key:>{max_key_length}}", end=' ')
            print() 
            first_flag = False
        print(path)
        # ipdb.set_trace()
        for values in metrics.values():
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{mean_value:>{max_key_length - 6}.2f}±{std_value:<5.2f}", end=' ')
            # print(f"{mean_value:>9.2f}±{std_value:<5.2f}", end=' ')
        print()  
        print() 

def main(input_path):
    metrics = {}
    criterion_list = ['near_fpr95', 'far_fpr95', 'near_auroc', 'far_auroc', 'near_auprin', 'far_auprin', 'near_auprout', 'far_auprout', 'allacc', 'id_acc', 'csid_v2', 'csid_c', 'csid_r', ]
    for item in criterion_list:
        metrics[item] = []
    if isinstance(input_path, str):
        if os.path.isfile(input_path): # one log file
            metrics = extract_metrics(input_path, metrics)
        elif os.path.isdir(input_path): # dir, calculate the mean results.
            sub_files = os.listdir(input_path)
            if 'log.txt' in sub_files:
                sub_path = os.path.join(input_path, 'log.txt')
                metrics = extract_metrics(sub_path, metrics)
            else:
                for path_seed in sub_files:
                    sub_path = os.path.join(input_path, path_seed, 'log.txt')
                    metrics = extract_metrics(sub_path, metrics)
        else:
            print(f"{input_path} 既不是文件也不是文件夹，或者不存在。")
        print_metrics(metrics)
    elif isinstance(input_path, list):
        # ipdb.set_trace()
        metrics_multi = {}
        for superpath in input_path:
            metrics_multi[superpath] = {}
            for item in criterion_list:
                metrics_multi[superpath][item] = []
            sub_files = os.listdir(superpath)
            if 'log.txt' in sub_files:
                sub_path = os.path.join(superpath, 'log.txt')
                metrics_multi[superpath] = extract_metrics(sub_path, metrics_multi[superpath])
            else:
                for path_seed in sub_files:
                    sub_path = os.path.join(superpath, path_seed, 'log.txt')
                    metrics_multi[superpath] = extract_metrics(sub_path, metrics_multi[superpath])
        print_metrics_multi(metrics_multi)
    else:
        print("input_path is neither a string nor a list")
    


if __name__ == "__main__":
    # print("Usage: python script.py <file_path>")
    if len(sys.argv) > 2:
        path_list = sys.argv[1:]
        main(path_list)
    else:
        main(sys.argv[1])