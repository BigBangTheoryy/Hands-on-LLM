import gensim.downloader as api

#Loading Model:
model = api.load("glove-wiki-gigaword-50")

print(model.most_similar([model['bike']], topn = 5))

