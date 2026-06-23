"""
Zero Shot Classification
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from datasets import load_dataset
import numpy as np


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

data = load_dataset("rotten_tomatoes")

#Giving Description to the models
label_embeddings = model.encode(['A negative review', "A positive review"])

# Text to embeddings:
test_embeddings = model.encode(data['test']['text'], show_progress_bar = True)


#Using cosine comparison of the document embeddings  versus the label description embeddings

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis = 1)


def evaluate_performance(y_true, y_pred):
    performance = classification_report(y_true, y_pred, target_names = ['Negative Review', "Positive Review"])
    return performance


print(evaluate_performance(data['test']['label'], y_pred))

