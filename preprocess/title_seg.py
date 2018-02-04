import jieba
import os

from sklearn.preprocessing import LabelEncoder


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return set(stopwords)


if __name__ == "__main__":
    titles = []
    tags = []
    titles_seg = []
    jieba.load_userdict('../dict/dict.txt')
    stopwords = stopwordslist('stopwords.txt')

    # content = open("../policy_corpus/raw.txt", "r", encoding="utf-8").read()
    with open('../policy_corpus/raw.txt', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            words = line.strip().split("$ipolicy$")
            if len(words) == 2:
                titles.append(words[1])

                titles_seg.append(filter(lambda item: item not in stopwords, jieba.cut(words[1])))

                tags.append(words[0])

    le = LabelEncoder()
    print(set(tags))
    print(len(set(tags)))
    tags = le.fit_transform(tags)

    with open('../policy_corpus/big_data.txt', 'w', encoding="utf-8") as f:
        lines = [' '.join(line) for line in titles_seg]
        f.write('\n'.join(lines))

    with open('../policy_corpus/big_label.txt', 'w', encoding="utf-8") as f:
        lines = [str(line) for line in tags]
        f.write('\n'.join(lines))
