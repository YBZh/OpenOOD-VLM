import csv
import pdb

# 文件名
input_file = "words_pos.csv"
nn_output_file = "nn_words.txt"
nnp_output_file = "nnp_words.txt"
jj_output_file = "jj_words.txt"

# 初始化存储 NN 和 JJ 的列表
nn_words = []
nnp_words = []
jj_words = []

# 读取 CSV 文件
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过表头
    # pdb.set_trace()
    for row in reader:
        word = row[1]  # 第二列是单词
        pos_tag = row[2]  # 第三列是 POS 标签
        if pos_tag == "NN":  # 筛选 NN
            nn_words.append(word)
        elif pos_tag == 'NNP':
            nnp_words.append(word)
        elif pos_tag == "JJ":  # 筛选 JJ
            jj_words.append(word)

print('len of NN, NNP, JJ are', len(nn_words), len(nnp_words), len(jj_words))

# 写入 NN 文件
with open(nn_output_file, mode="w", encoding="utf-8") as file:
    for word in nn_words:
        file.write(word + "\n")

with open(nnp_output_file, mode="w", encoding="utf-8") as file:
    for word in nnp_words:
        file.write(word + "\n")

# 写入 JJ 文件
with open(jj_output_file, mode="w", encoding="utf-8") as file:
    for word in jj_words:
        file.write(word + "\n")

print("筛选完成，结果已写入 nn_words.txt 和 jj_words.txt")