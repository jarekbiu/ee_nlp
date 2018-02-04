# -*- coding: utf-8 -*-

if __name__ == "__main__":

    titles = []
    # results = []
    tags = set()

    with open('test.txt', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            words = line.strip().split("$ipolicy$")
            if len(words) == 2:
                titles.append(words[1])
                # results.append(words[0])
                # tags.add(words[0])
    with open('../policy.text', 'w', encoding="utf-8") as f:
        f.write("\n".join(titles))
