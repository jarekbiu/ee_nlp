# coding:utf-8
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    VECTOR_DIR = '../wiki.zh.text.vector'

    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 300
    TEST_SPLIT = 0.2

    print('(1) load texts...')
    texts = open('../policy_corpus/data.txt', encoding="utf-8").read().split("\n")
    labels = open('../policy_corpus/label.txt', encoding="utf-8").read().split("\n")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=TEST_SPLIT,
        random_state=42)

    print('(2) doc to var...')
    import gensim
    import numpy as np

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR)
    x_train = []
    x_test = []
    for train_doc in train_texts:
        words = train_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_train.append(vector)
    for test_doc in test_texts:
        words = test_doc.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if word in w2v_model:
                vector += w2v_model[word]
                word_num += 1
        if word_num > 0:
            vector = vector / word_num
        x_test.append(vector)
    print('train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0])))
    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))
    y_train = train_labels
    y_test = test_labels

    print('(3) SVM...')
    from sklearn.svm import SVC

    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))
