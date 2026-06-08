from sentence_transformers import SentenceTransformer

#Load Model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

text = "Best Movie Ever!"
vector = model.encode(text)

print(vector.shape) #(768,) ->The Text is now encoded in this one vector with a dimension of 768 numerical values