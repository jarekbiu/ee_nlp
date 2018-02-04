import jieba
import os


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return set(stopwords)


if __name__ == "__main__":
    titles = []
    tags = []
    titles_seg = []
    jieba.load_userdict('../dict/dict.txt')
    stopwords = stopwordslist('stopwords.txt')

    content = open("test.txt", "r", encoding="utf-8").read()
    with open('test.txt', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            words = line.strip().split("$ipolicy$")
            if len(words) == 2:
                titles.append(words[1])

                titles_seg.append(filter(lambda item: item not in stopwords, jieba.cut(words[1])))

                tags.append(words[0])

    with open('title.seg', 'w', encoding="utf-8") as f:
        for i in range(len(tags)):
            f.write(titles[i] + " " + " ".join(titles_seg[i]) + " " + tags[i] + os.linesep)
