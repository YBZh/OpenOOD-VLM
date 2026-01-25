import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

# 提取所有单词
all_words = set()  # 使用集合去重
for synset in wn.all_synsets():  # 遍历所有同义词集
    for lemma in synset.lemmas():  # 遍历同义词集中的每个词条
        all_words.add(lemma.name())  # 添加词条到集合中

# 将单词保存到本地文件
output_file = "wordnet_words.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for word in sorted(all_words):  # 按字母排序
        f.write(word + "\n")

print(f"WordNet 中的单词已保存到 {output_file}，共计 {len(all_words)} 个单词。")

nouns = set()
for synset in wn.all_synsets(pos='n'):  # 只提取名词
    for lemma in synset.lemmas():
        nouns.add(lemma.name())

# 保存名词到文件
with open("wordnet_nouns.txt", "w", encoding="utf-8") as f:
    for noun in sorted(nouns):
        f.write(noun + "\n")

print(f"名词保存到 wordnet_nouns.txt，数量：{len(nouns)}")

adjs = set()
for synset in wn.all_synsets(pos='a'):  # 只提取名词
    for lemma in synset.lemmas():
        adjs.add(lemma.name())

# 保存名词到文件
with open("wordnet_adjs.txt", "w", encoding="utf-8") as f:
    for adj in sorted(adjs):
        f.write(adj + "\n")

print(f"形容词保存到 wordnet_adjs.txt，数量：{len(adjs)}")