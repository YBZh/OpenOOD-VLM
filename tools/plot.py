import torch
import ipdb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

color_dict = {
    'dark_blue': '#403990',
    'light_blue': '#80A6E2',
    'yellow': '#FBDD85',
    'light_red': '#F46F43',
    'dark_red': '#CF3D3E'
}

# ########### number of negative labels
# neg_num = [50, 100, 300, 500, 800, 1000, 2000, 3000, 5000, 8000, 10000, 15000] #, 40000]
# # neglabel_near_fpr = [100, 100, 80.3, 76.4, 72.3, 70.9, 68.7, 68.6, 67.4, 68.5, 67.6, 71.1, 74.3]
# neglabel_far_fpr = [52.9, 31.8, 32.5, 29.6, 28.7, 28.0, 26.7, 26.3, 25.5, 25.0, 25.3, 25.5] #, 25.7]
# # aaneg_near_fpr_noaware = [100, 82.4, 59.8, 58.4, 58.0, 58.0, 59.9, 60.9, 65.3, 68.9, 69.8, 72.9, 75.0]
# aaneg_far_fpr_noaware = [37.6, 19.3, 17.6, 17.6, 17.8, 17.6, 19.6, 20.2, 22.2, 23.3, 24.0, 24.4] #, 27.0]
# # aaneg_near_fpr = [83.8, 65.6, 59.8, 59.6, 56.3, 55.75, 57.9, 58.8, 61.0, 65.2, 66.2, 69.7, 71.9]
# aaneg_far_fpr = [24.4, 18.8, 16.7, 16.9, 17.4, 17.6, 18.9, 18.9, 20.7, 21.5, 21.8, 22.4] #, 22.9]

# # Plot
# plt.figure(figsize=(3.2, 2.8))
# # Plot far FPR lines
# plt.plot(neg_num, neglabel_far_fpr, label='NegLabel', color='#403990', linestyle='-', marker='o')
# plt.plot(neg_num, aaneg_far_fpr_noaware, label=r'TANL (w/o $\mathcal{S}_{aa}$)', color='#F46F43', linestyle='--', marker='s')
# plt.plot(neg_num, aaneg_far_fpr, label='TANL', color='#CF3D3E', linestyle='-.', marker='^')

# # Customize the plot
# plt.xlabel('Number $M$ of negative labels')
# plt.ylabel('FPR95 (%)')
# # plt.title('Comparison of Far FPR')
# plt.legend()
# # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # Show plot
# plt.tight_layout()
# plt.savefig("negnum.pdf", dpi=300, bbox_inches="tight")  # 保存图像




# # different label selection strategy. 
# methods = ["neglabel", "pos_only", "neg_only", "neg_pos"]
# near = [70.9, 75.2, 56.39, 55.75]
# far = [28.0, 29.52, 18.92, 17.6]
# # 设置柱状图参数
# x = np.arange(2)  # near 和 far 两个分类
# width = 0.2  # 每个柱子的宽度
# # 创建画布
# fig, ax = plt.subplots(figsize=(3.2,2.8))
# # 绘制柱状图
# bar_neglabel = ax.bar(x - 1.5 * width, [near[0], far[0]], width, label='NegLabel', color=color_dict['dark_blue'])
# bar_pos_only = ax.bar(x - 0.5 * width, [near[1], far[1]], width, label=r'-$Act_b({\mathcal{X}_{pos}, \widehat{y}_i})$', color=color_dict['light_blue'])
# bar_neg_only = ax.bar(x + 0.5 * width, [near[2], far[2]], width, label=r'$Act_b({\mathcal{X}_{neg}}, \widehat{y}_i)$', color='#F46F43')
# bar_neg_pos = ax.bar(x + 1.5 * width, [near[3], far[3]], width, label=r'$\widehat{Act}_b(\widehat{y}_i)$', color='#CF3D3E')

# # 添加标签、标题和图例
# # ax.set_xlabel('OOD Setting', fontsize=12)
# ax.set_ylabel('FPR95', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(['Near OOD', 'Far OOD'], fontsize=10)
# ax.legend( fontsize=8.5, title_fontsize=10)
# # 调整布局并保存图像
# plt.tight_layout()
# plt.savefig("label_selection_strategy_grouped.pdf", dpi=300, bbox_inches="tight")


# # different similarity criterion
# methods = ["Vanilla cosine similarity", "Normalized cosine similarity"]
# near = [76.25, 58.83]
# far = [29.67, 17.21]
# # 设置柱状图参数
# x = np.arange(2)  # near 和 far 两个分类
# width = 0.3  # 每个柱子的宽度
# # 创建画布
# fig, ax = plt.subplots(figsize=(4.1, 3.7))
# # 绘制柱状图

# vanilla_cosine = ax.bar(x - 0.5 * width, [near[0], far[0]], width, label='Vanilla cosine similarity', color=color_dict['dark_blue'])
# normalized_cosine = ax.bar(x + 0.5 * width, [near[1], far[1]], width, label='Normalized cosine similarity', color=color_dict['dark_red'])

# # 添加标签、标题和图例
# # ax.set_xlabel('OOD Setting', fontsize=12)
# ax.set_ylabel('FPR95', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(['Near OOD', 'Far OOD'], fontsize=10)
# ax.legend( fontsize=10, title_fontsize=10)
# # 调整布局并保存图像
# plt.tight_layout()
# plt.savefig("similarity_metric.pdf", dpi=300, bbox_inches="tight")




# # circular enhancement
# methods = ["w/o mutual enhancement", "with mutual enhancement"]
# near = [63.00, 58.83]
# far = [19.04, 17.21]
# # 设置柱状图参数
# x = np.arange(2)  # near 和 far 两个分类
# width = 0.3  # 每个柱子的宽度
# # 创建画布
# fig, ax = plt.subplots(figsize=(4.1, 3.7))
# # 绘制柱状图

# vanilla_cosine = ax.bar(x - 0.5 * width, [near[0], far[0]], width, label=methods[0], color=color_dict['dark_blue'])
# normalized_cosine = ax.bar(x + 0.5 * width, [near[1], far[1]], width, label=methods[1], color=color_dict['dark_red'])

# # 添加标签、标题和图例
# # ax.set_xlabel('OOD Setting', fontsize=12)
# ax.set_ylabel('FPR95', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(['Near OOD', 'Far OOD'], fontsize=10)
# ax.legend( fontsize=10, title_fontsize=10)
# # 调整布局并保存图像
# plt.tight_layout()
# plt.savefig("mutual_enhancement.pdf", dpi=300, bbox_inches="tight")


# # analyses on the value of alpha.
# alpha = [1.0, 0.99, 0.95, 0.9, 0.8, 0.5]
# fpr95_near = [58.53, 57.88, 56.62, 56.61, 58.92, 59.11]
# fpr95_far = [11.48, 10.98, 11.40, 12.58, 12.78, 12.80]  # Far OOD
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(4.5,4.0 ))
# # 绘制 Near OOD 曲线
# color_near = 'darkred'
# ax1.plot(alpha, fpr95_near, marker='o', label='Near OOD', color=color_near)
# ax1.set_xlabel(r'$\alpha$ value', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_near)
# ax1.tick_params(axis='y', labelcolor=color_near)
# # 设置 x 轴从大到小排列
# ax1.invert_xaxis()
# # 创建第二个 y 轴
# ax2 = ax1.twinx()
# color_far = 'lightcoral'
# ax2.plot(alpha, fpr95_far, marker='s', label='Far OOD', color=color_far)
# ax2.set_ylabel('FPR95 (Far OOD)', fontsize=12, color=color_far)
# ax2.tick_params(axis='y', labelcolor=color_far)
# # 图例
# fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.5), fontsize=12, ncol=1)
# # 美化图表
# plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("alpha_dis_batch_tradeoff.pdf", dpi=300, bbox_inches="tight")



# # value of gamma
# alpha = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95]
# fpr95_near = [68.18, 62.48, 61.10, 60.28, 60.53, 59.86, 57.38, 58.42]
# fpr95_far =           [18.72, 17.61, 17.23, 17.49, 17.67, 17.91, 18.66, 19.32]  # Far OOD
# # far_traditiona_four = [12.29, 9.83,  9.97,  9.94, 10.39, 11.45, 15.66]
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(4.5,4.0 ))
# # 绘制 Near OOD 曲线
# ax1.plot(alpha, fpr95_near, marker='o', label='Near OOD', color=color_dict['dark_blue'])
# ax1.set_xlabel(r'$\gamma$ value', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_dict['dark_blue'])
# ax1.tick_params(axis='y', labelcolor=color_dict['dark_blue'])
# # Add horizontal dashed line for Near OOD baseline
# ax1.axhline(y=58.83, color=color_dict['dark_blue'], linestyle='--', linewidth=1, label='Auto threshold')
# # 设置 x 轴从大到小排列
# # ax1.invert_xaxis()
# # 创建第二个 y 轴
# ax2 = ax1.twinx()
# ax2.plot(alpha, fpr95_far, marker='s', label='Far OOD', color=color_dict['dark_red'])
# ax2.set_ylabel('FPR95 (Far OOD)', fontsize=12, color=color_dict['dark_red'])
# ax2.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# # Add horizontal dashed line for Far OOD baseline
# ax2.axhline(y=17.21, color=color_dict['dark_red'], linestyle='--', linewidth=1, label='Auto threshold')
# # 图例
# fig.legend(loc="upper right", bbox_to_anchor=(0.72, 0.93), fontsize=12, ncol=1)
# # 美化图表
# plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("gamma_value.pdf", dpi=300, bbox_inches="tight")



# error rate in first batch
error_rate =    [0,     0.1,    0.2,   0.3,    0.4,  0.5,   0.6,   0.7,   0.8,   0.9,   1.0]
fpr95_ssbhard = [58.43, 60.11, 62.11, 63.02, 64.12, 65.83, 69.23, 72.64, 75.04, 76.45, 78.85]

# 创建画布
fig, ax1 = plt.subplots(figsize=(3.2, 2.8 ))
# 绘制 Near OOD 曲线
ax1.plot(error_rate, fpr95_ssbhard, marker='o', label='Ours', color=color_dict['dark_red'])
ax1.set_xlabel('Error Rate in First Batch', fontsize=12)
ax1.set_ylabel('FPR95', fontsize=12, color=color_dict['dark_red'])
ax1.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# Add horizontal dashed line for Near OOD baseline
ax1.axhline(y=74.83, color=color_dict['dark_blue'], linestyle='--', linewidth=3, label='NegLabel')
# 图例
fig.legend(loc="upper right", bbox_to_anchor=(0.68, 0.78), fontsize=12, ncol=1)
# 美化图表
plt.grid(alpha=0.3)
plt.tight_layout()
# 保存图表
plt.savefig("early_errors.pdf", dpi=300, bbox_inches="tight")




# number of test samples
number_of_test_samples =    [100,   500, 1000,  2000, 5000, 10000, 50000, 99000]
fpr95_ssbhard =             [72.78, 71.5, 68.0, 66.39, 65.1, 63.9, 62.9,  62.5 ]

# 创建画布
fig, ax1 = plt.subplots(figsize=(3.2, 2.8 ))
# 绘制 Near OOD 曲线
ax1.plot(number_of_test_samples, fpr95_ssbhard, marker='o', label='Ours', color=color_dict['dark_red'])
ax1.set_xlabel('Number of Test Samples', fontsize=12)
ax1.set_ylabel('FPR95', fontsize=12, color=color_dict['dark_red'])
ax1.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# Add horizontal dashed line for Near OOD baseline
ax1.axhline(y=74.83, color=color_dict['dark_blue'], linestyle='--', linewidth=3, label='NegLabel')
# 图例
fig.legend(loc="upper right", bbox_to_anchor=(0.8, 0.78), fontsize=12, ncol=1)
# 美化图表
plt.grid(alpha=0.3)
plt.tight_layout()
# 保存图表
plt.savefig("test_number.pdf", dpi=300, bbox_inches="tight")


# # batch size
# batchsize = [1, 4, 8, 16, 32, 64, 128]
# # fpr95_near = [60.59, 60.01, 59.31, 59.00, 58.91, 58.85, 58.83]
# fpr95_near = [60.59, 60.31, 60.24, 60.11, 60.08, 60.06, 60.06]

# # 创建画布
# fig, ax1 = plt.subplots(figsize=(3.2, 2.8 ))
# # 绘制 Near OOD 曲线
# ax1.plot(batchsize, fpr95_near, marker='o', label='Ours', color=color_dict['dark_red'])
# ax1.set_xlabel('Batch size in testing', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_dict['dark_red'])
# ax1.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# # Add horizontal dashed line for Near OOD baseline
# ax1.axhline(y=69.45, color=color_dict['dark_blue'], linestyle='--', linewidth=3, label='NegLabel')
# # 图例
# fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.7), fontsize=12, ncol=1)
# # 美化图表
# plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("batch_size.pdf", dpi=300, bbox_inches="tight")



# # corpus size
# alpha = [2000, 5000, 10000, 20000, 50000, 100000, 200000]
# fpr95_near = [63.15, 62.28, 62.01, 61.34, 60.85, 60.54, 60.36]
# fpr95_far = [24.21, 23.65, 21.79, 20.37, 18.96, 18.39, 17.59]  # Far OOD
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(4.5,4.0 ))
# # 绘制 Near OOD 曲线
# ax1.plot(alpha, fpr95_near, marker='o', label='Near OOD', color=color_dict['dark_blue'])
# ax1.set_xlabel(r'Size of corpus dataset', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_dict['dark_blue'])
# ax1.tick_params(axis='y', labelcolor=color_dict['dark_blue'])
# # 设置 x 轴从大到小排列
# # ax1.invert_xaxis()
# # 创建第二个 y 轴
# ax2 = ax1.twinx()
# ax2.plot(alpha, fpr95_far, marker='s', label='Far OOD', color=color_dict['dark_red'])
# ax2.set_ylabel('FPR95 (Far OOD)', fontsize=12, color=color_dict['dark_red'])
# ax2.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# # 图例
# fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.5), fontsize=12, ncol=1)
# # 美化图表
# plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("corpus_size.pdf", dpi=300, bbox_inches="tight")



# # value of gap
# alpha = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
# fpr95_near = [59.45, 59.89, 60.67, 60.93, 61.77, 61.21, 70.04]
# fpr95_far = [17.58, 17.41, 17.33, 17.61, 17.56, 17.69, 18.78]  # Far OOD
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(4.5,4.0 ))
# # 绘制 Near OOD 曲线
# ax1.plot(alpha, fpr95_near, marker='o', label='Near OOD', color=color_dict['dark_blue'])
# ax1.set_xlabel(r'$g$ value', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_dict['dark_blue'])
# ax1.tick_params(axis='y', labelcolor=color_dict['dark_blue'])
# # 设置 x 轴从大到小排列
# # ax1.invert_xaxis()
# # 创建第二个 y 轴
# ax2 = ax1.twinx()
# ax2.plot(alpha, fpr95_far, marker='s', label='Far OOD', color=color_dict['dark_red'])
# ax2.set_ylabel('FPR95 (Far OOD)', fontsize=12, color=color_dict['dark_red'])
# ax2.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# # 图例
# fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.5), fontsize=12, ncol=1)
# # 美化图表
# plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("gap_value.pdf", dpi=300, bbox_inches="tight")



# # analyses on the length of queue.
# queue_len = [10, 30, 50, 100, 200, 300]
# fpr95_near = [62.69, 61.70, 61.54, 60.82, 60.56, 60.55]
# fpr95_far = [18.82, 17.79, 17.63, 17.51, 17.19, 17.17]  # Far OOD
# # 创建画布
# fig, ax1 = plt.subplots(figsize=(4.5,4.0 ))
# # 绘制 Near OOD 曲线
# ax1.plot(queue_len, fpr95_near, marker='o', label='Near OOD', color=color_dict['dark_blue'])
# ax1.set_xlabel(r'Queue length', fontsize=12)
# ax1.set_ylabel('FPR95 (Near OOD)', fontsize=12, color=color_dict['dark_blue'])
# ax1.tick_params(axis='y', labelcolor=color_dict['dark_blue'])

# # 设置 x 轴从大到小排列
# # ax1.invert_xaxis()
# # 创建第二个 y 轴
# ax2 = ax1.twinx()
# ax2.plot(queue_len, fpr95_far, marker='s', label='Far OOD', color=color_dict['dark_red'])
# ax2.set_ylabel('FPR95 (Far OOD)', fontsize=12, color=color_dict['dark_red'])
# ax2.tick_params(axis='y', labelcolor=color_dict['dark_red'])
# # 图例
# fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.5), fontsize=12, ncol=1)
# # 美化图表
# # plt.grid(alpha=0.3)
# plt.tight_layout()
# # 保存图表
# plt.savefig("queue_length.pdf", dpi=300, bbox_inches="tight")




# import matplotlib.pyplot as plt

# # 数据
# percentile_values = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.9, 1.0]
# ninco_auroc = [78.90, 78.95, 79.49, 80.36, 80.05, 79.65, 75.12, 74.36]  # Near-OOD

# # 创建子图
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 6), gridspec_kw={'width_ratios': [4, 1]})
# fig.subplots_adjust(wspace=0.05)  # 调整子图间距

# # 数据分段
# x_left = percentile_values[:6]
# y_left = ninco_auroc[:6]
# x_right = percentile_values[6:]
# y_right = ninco_auroc[6:]

# # 绘制左侧子图
# ax1.plot(x_left, y_left, 's-', linewidth=2, markersize=8, color='#9BD0E2', label='Near-OOD')
# ax1.set_xlim(0.02, 0.12)  # 左侧 X 轴范围
# ax1.set_ylim(72, 85)  # Y 轴范围
# ax1.set_xlabel("Ratio", fontsize=12)
# ax1.set_ylabel("AUROC", fontsize=12)
# ax1.legend(loc='upper right', fontsize=10)

# # 绘制右侧子图
# ax2.plot(x_right, y_right, 's-', linewidth=2, markersize=8, color='#9BD0E2', label='Near-OOD')
# ax2.set_xlim(0.9, 1.0)  # 右侧 X 轴范围
# # ax2.set_xlabel("Percentile (Right)", fontsize=12)
# # ax2.legend(loc='upper right', fontsize=10)

# # turn off spines
# ax1.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)

# ax1.tick_params(bottom=False)
# ax2.tick_params(bottom=True)

# # 添加 X 轴的断裂线
# d = 0.015  # 断裂线的长度
# # 左侧子图右边的断裂线
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.5)
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 左下断裂线
# ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 左上断裂线

# # 右侧子图左边的断裂线
# kwargs.update(transform=ax2.transAxes)
# ax2.plot((-d, +d), (-d, +d), **kwargs)  # 右下断裂线
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 右上断裂线
# # ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 左下断裂线
# # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 左上断裂线


# # 调整布局
# plt.tight_layout()


# # 保存图形
# plt.savefig('analysis_k_broken.pdf', format='pdf', dpi=300, bbox_inches='tight')
# # plt.show()



# # ######### AANeg motivation, 
# # ood_score = torch.load('ood_activation_imagenet_places.pth', map_location=torch.device('cpu')) ## 10K
# # id_score = torch.load('id_activation_imagenet_places.pth', map_location=torch.device('cpu')) ## 10K

# # # Sort the tensor in descending order
# # sorted_ood_score, sorted_index = torch.sort(ood_score, descending=True)
# # sorted_id_score = id_score[sorted_index]

# # # 配置科研风格配色
# # # plt.style.use('seaborn-darkgrid')  # 使用适合科研的样式

# # # 绘制曲线图
# # plt.figure(figsize=(5, 3))
# # plt.plot(
# #     sorted_ood_score.numpy(),
# #     label="$Act(\mathcal{X}_{ood})$",
# #     color='#403990',  # 蓝色
# #     alpha=1.0,  # 添加透明度
# #     linewidth=1.5
# # )
# # plt.plot(
# #     sorted_id_score.numpy(),
# #     label="$Act(\mathcal{X}_{id})$",
# #     color='#CF3D3E',  # 橙色
# #     alpha=0.8,  # 添加透明度
# #     linewidth=0.5,
# #     linestyle='--'  # 使用虚线区分
# # )

# # # 格式化 Y 轴为科学计数法
# # ax = plt.gca()
# # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # 设置科学计数法范围

# # # 添加图例、标题和标签
# # # plt.title("OOD vs ID Scores", fontsize=16)
# # plt.xlabel("Negative Labels Sorted by $Act(\mathcal{X}_{ood})$", fontsize=14)
# # plt.ylabel("Activation Scores", fontsize=14)
# # plt.legend(fontsize=12, loc="upper right")
# # plt.grid(True, linestyle='--', alpha=0.6)

# # # 增加边框线宽和刻度大小
# # plt.gca().spines['top'].set_linewidth(1.2)
# # plt.gca().spines['right'].set_linewidth(1.2)
# # plt.gca().spines['left'].set_linewidth(1.2)
# # plt.gca().spines['bottom'].set_linewidth(1.2)
# # plt.tick_params(axis='both', which='major', labelsize=12)

# # # 显示图形
# # plt.tight_layout()
# # plt.savefig("actscore_ood_id.pdf", dpi=300, bbox_inches="tight")  # 保存图像


# # score = ood_score - id_score
# # sorted_scores = torch.sort(score, descending=True)[0]
# # topN = [10, 100, 200, 300, 600, 800, 1000, 2000, 3000, 5000, 7000, 8000, 9000, 10000]
# # AUROC = [92.61, 95.33, 95.85, 96.07, 96.22, 96.22, 96.15, 95.85, 95.48, 94.92,94.45, 93.89, 93.09, 91.31 ]
# # FPR = [36.76, 25.58, 21.71, 20.11, 19.01, 18.92, 19.43, 21.30, 23.02, 25.68, 27.67, 29.80, 32.77, 40.90]


# # # 创建图形
# # fig, ax1 = plt.subplots(figsize=(5, 3))

# # # 绘制 score 曲线 (左边Y轴)
# # line1, = ax1.plot(
# #     sorted_scores, label="Sorted Scores", color="#403990"
# # )  # 保存 Line2D 对象
# # ax1.set_xlabel(
# #     "Negative Labels Sorted by $Act(\mathcal{X}_{ood}) - Act(\mathcal{X}_{id})$",
# #     fontsize=12,
# # )
# # ax1.set_ylabel(
# #     "$Act(\mathcal{X}_{ood}) - Act(\mathcal{X}_{id})$", fontsize=12, color="#403990"
# # )
# # ax1.tick_params(axis="y", labelcolor="#403990")
# # ax1.grid(True, linestyle="--", alpha=0.6)

# # # 格式化 Y 轴为科学计数法
# # ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))  # 设置科学计数法范围

# # # 绘制 AUROC 曲线 (右边Y轴)
# # ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
# # line2, = ax2.plot(
# #     topN, FPR, label="FPR95", color="#CF3D3E", marker="o", linestyle="--"
# # )  # 保存 Line2D 对象
# # ax2.set_ylabel("FPR95", fontsize=12, color="#CF3D3E")
# # ax2.tick_params(axis="y", labelcolor="#CF3D3E")

# # # 添加合并的图例
# # fig.legend(
# #     handles=[line1, line2],  # 使用两个曲线的 Line2D 对象
# #     labels=["$Act(\mathcal{X}_{ood}) - Act(\mathcal{X}_{id})$", "FPR95"],  # 自定义图例标签
# #     loc="upper right",  # 自动选择最优位置
# #     bbox_to_anchor=(0.73, 0.85), # 图例的锚点
# #     fontsize=12,
# # )

# # # 调整布局避免重叠
# # fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以适应图例位置

# # # 保存图像
# # plt.savefig("score_auroc_curve.pdf", dpi=300, bbox_inches="tight")