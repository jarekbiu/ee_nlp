# -*- coding: utf-8 -*-
import os
import pandas as pd
import re

if __name__ == "__main__":
    dict = set()

    # 科技
    with open('科技.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            words = re.split(u"、|（|）",line)
            dict = dict.union(set(words))
    dict.remove("")
    # print(dict)

    rootdir = 'dict_raw'

    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件

    for i in range(0, len(list)):

        path = os.path.join(rootdir, list[i])

        if os.path.isfile(path):
            df = pd.read_excel(path,header=None,names = ['word'])
            dict = dict.union(set(df['word']))

    with open('dict.txt', 'w', encoding='utf-8') as f:
        for item in dict:
            f.write("%s\n" % item)


