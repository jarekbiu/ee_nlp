import time
import jieba
if __name__ == "__main__":
    jieba.load_userdict('../dict/dict.txt')

    # jieba.enable_parallel(8)

    input = '../policy.text'
    output = '../policy.seg'

    content = open(input,"rb").read()
    t1 = time.time()
    words = " ".join(jieba.cut(content))          # 精确模式

    t2 = time.time()
    tm_cost = t2-t1
    if tm_cost==0:
        tm_cost = 1

    log_f = open(output,"wb")
    log_f.write(words.encode('utf-8'))


    print('speed %s bytes/second' % (len(content)/tm_cost))