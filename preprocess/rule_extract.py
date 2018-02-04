# -*- coding: utf-8 -*-
import re


def load_rule():
    with open('rule.txt', 'r', encoding="utf-8") as f:
        tag_allow = {}
        for line in f.readlines():
            line = line.strip()
            words = line.split("\t")
            tag_allow[words[0]] = set(words[1:len(words)])
    return tag_allow


if __name__ == "__main__":

    tab = {}
    tags = set()
    with open('test.txt', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            words = line.strip().split("$ipolicy$")
            if len(words) == 2:
                tag = words[0]
                tags.add(tag)
                title = words[1]
                index = title.rfind(u"部")
                if index >= 0:
                    st = title[0:index + 1]
                    deps = st.split("、")
                    for dep in deps:
                        if dep not in tab:
                            tab[dep] = set()
                        tab[dep].add(tag)

    print(tags)
    print(len(tags))
    with open('rule.txt', 'w', encoding="utf-8") as f:
        sort_res = sorted(tab.items(), key=lambda item: item[0])

        res_tab = {}
        for temp in sort_res:
            dep = re.sub("《|》|中华人民共和国", '', temp[0])
            tags = temp[1]
            if dep in res_tab:
                res_tab[dep] = res_tab[dep].union(tags)
            else:
                flag = False
                for k, v in res_tab.items():
                    if dep.find(k) >= 0:
                        res_tab[k] = res_tab[k].union(tags)
                        flag = True
                        break
                if not flag:
                    res_tab[dep] = tags

        sort_res = sorted(res_tab.items(), key=lambda item: item[0])
        for temp in sort_res:
            f.write(temp[0] + "\t" + "\t".join(temp[1]) + "\n")
