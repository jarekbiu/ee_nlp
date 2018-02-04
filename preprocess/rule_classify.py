# -*- coding: utf-8 -*-

import gensim
import jieba
import numpy as np

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def load_rule():
    with open('rule.txt', 'r') as f:
        tag_allow = {}
        for line in f.readlines():
            line = line.strip()
            words = line.split("\t")
            tag_allow[words[0]] = set(words[1:len(words) - 1])
    return tag_allow


# tf-idf提取每个标题的关键词
def tf_idf(segs, k):
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(segs))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    keywords = []
    for w in weight:
        loc = np.argsort(-w)
        temp = []
        for i in range(k):
            temp.append([word[loc[i]], w[loc[i]]])
        keywords.append(temp)
    return keywords


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return set(stopwords)


tags_allow = load_rule()


def tag_filter(title, tag_table):
    tags = set()
    for dep, dep_tag in tags_allow.items():
        if title.find(dep) >= 0:
            tags = tags.union(dep_tag)

    filter_res = {}
    for tag in tags:
        if tag not in filter_res:
            filter_res[tag] = set()
        filter_res[tag] = filter_res[tag].union(tag_table[tag])

    if len(filter_res) == 0:
        filter_res = tag_table

    return filter_res


def max_cos(title_seg, model, tag_table, stopwords):
    sim = 0.0
    res = ['', '']
    for word in title_seg:
        # weight = w[1]
        if word not in model:
            continue

        for k, se in tag_table.items():
            for v in se:
                temp = model.wv.similarity(v, word)
                if temp > sim:
                    sim = temp
                    res = [k, word]
    return res


def max_total_cos(title_seg, model, tag_table, stopwords):
    table = {}
    res = ''

    for word in title_seg:

        if word not in model:  # 未登陆词
            continue

        for k, se in tag_table.items():
            for v in se:
                if v not in model:  # 未登陆词
                    continue
                temp = model.wv.similarity(v, word)
                if k not in table:
                    table[k] = temp
                else:
                    table[k] = table[k] + temp

    max = 0
    for k, value in table.items():
        if value > max:
            max = value
            res = k
    return [res]


def judgeTag(title, title_seg, model, tag_table, stopwords, keywords=[]):
    tag_table = tag_filter(title, tag_table)
    # res = max_total_cos(title_seg, model, tag_table, stopwords)
    # keywords = [keyword[0] for keyword in keywords]
    # res = max_cos(keywords, model, tag_table, stopwords)
    res = max_cos(title_seg, model, tag_table, stopwords)

    return res


if __name__ == "__main__":

    model = gensim.models.Word2Vec.load("../wiki.zh.text.vector")

    titles = []
    titles_seg = []
    results = []
    dest = []
    keys = []
    tags = set()

    with open('../policy_corpus/title.seg', 'r') as f:
        for line in f.readlines():
            words = line.strip().split(" ")
            if len(words) < 2:
                continue
            titles.append(words[0])
            titles_seg.append(words[1:len(words) - 1])
            results.append(words[len(words) - 1])
            tags.add(words[len(words) - 1])

    stopwords = stopwordslist('stopwords.txt')  # 这里加载停用词的路径

    jieba.load_userdict('../dict/dict.txt')
    tag_table = {}
    for tag in tags:
        words = jieba.cut(tag)
        tag_table[tag] = []
        for word in words:
            if word not in stopwords and word in model:
                tag_table[tag].append(word)
    #
    title_seg = []
    for title in titles:
        title_seg.append(' '.join(jieba.cut(title)))

    keywords = tf_idf(title_seg, 3)
    # with open('keywords', 'w', encoding="utf-8") as f:
    #     for i in range(len(titles)):
    #         f.write("%s\t%s\n" % (titles[i], '\t'.join(keywords[i]))

    # print(tag_table)
    # print("pre finish")

    cnt = 0
    acc = 0
    right = []

    for i in range(len(titles)):
        # if cnt == 5000:
        #     break
        res = judgeTag(titles[i], titles_seg[i], model, tag_table, stopwords, keywords[i])
        dest.append(res[0])
        keys.append(res[1])
        if res[0] == results[i]:
            acc = acc + 1
            right.append("%s\t%s\t%s" % (titles[i], dest[i], results[i]))
        cnt = cnt + 1
        if cnt % 1000 == 0:
            print("acc:" + str(acc) + " cnt:" + str(cnt) + " %:" + str(acc * 1.0 / cnt))

    with open('rule_result.txt', 'w') as f:
        f.write("titel\tactual\tmodel\tkeyword\n")
        # f.write("titel\tmodel\tactual\n")
        for i in range(len(dest)):
            # f.write("%s\t%s\t%s\n" % (titles[i], dest[i], results[i]))
            f.write("%s\t%s\t%s\t%s\n" % (titles[i], dest[i], results[i], keys[i]))

    with open('rule_result_right.txt', 'w') as f:
        f.write("\n".join(right))
    print(acc * 1.0 / cnt)
