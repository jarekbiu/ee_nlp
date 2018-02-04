# coding:utf-8
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    VECTOR_DIR = '../wiki.zh.text.vector'

    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 300
    TEST_SPLIT = 0.2

    print('(1) load texts...')
    texts = open('../policy_corpus/big_data.txt', encoding="utf-8").read().split("\n")
    labels = open('../policy_corpus/big_label.txt', encoding="utf-8").read().split("\n")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels,
        test_size=TEST_SPLIT,
        random_state=42)

    print('(2) doc to var...')
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    count_v0 = CountVectorizer()
    counts_all = count_v0.fit_transform(texts)
    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_train = count_v1.fit_transform(train_texts)
    print("the shape of train is " + repr(counts_train.shape))
    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_test = count_v2.fit_transform(test_texts)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit(counts_train).transform(counts_train)
    test_data = tfidftransformer.fit(counts_test).transform(counts_test)

    x_train = train_data
    y_train = train_labels
    x_test = test_data
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
