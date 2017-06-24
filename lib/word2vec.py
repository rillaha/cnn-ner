import gensim
model = gensim.models.Word2Vec.load('model/word2vec.model')


def load_embedding():
    return model
