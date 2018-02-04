# coding:utf-8
from sklearn.model_selection import train_test_split


def train_d2v_model():
    # all_docs = train_docs + test_docs
    # fout = open('all_contents.txt', 'w')
    # fout.write('\n'.join(all_docs))
    # fout.close()
    import gensim
    sentences = gensim.models.doc2vec.TaggedLineDocument('../policy_corpus/data.txt')
    model = gensim.models.Doc2Vec(sentences, size=200, window=5, min_count=5)
    model.save('doc2vec.model')
    print('num of docs: ' + str(len(model.docvecs)))


if __name__ == '__main__':

    EMBEDDING_DIM = 200
    TEST_SPLIT = 0.2

    texts = open('../policy_corpus/data.txt', encoding="utf-8").read().split('\n')
    labels = open('../policy_corpus/label.txt', encoding="utf-8").read().split('\n')

    print('(1) training doc2vec model...')
    # train_d2v_model()
    print('(2) load doc2vec model...')
    import gensim

    model = gensim.models.Doc2Vec.load('doc2vec.model')

    train_docs, test_docs, train_labels, test_labels = train_test_split(
        model.docvecs, labels,
        test_size=TEST_SPLIT,
        random_state=42)

    x_train = train_docs
    x_test = test_docs
    y_train = train_labels
    y_test = test_labels

    print('train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0])))
    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))

    print('(3) SVM...')
    from sklearn.svm import SVC

    svclf = SVC(kernel='rbf')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))
