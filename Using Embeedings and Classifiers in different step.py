"""
In this program, I will be generating impeding and then feed those embedding features to the classify to generate text classification. In the previous example, we used a pre-Train transformer model that generated the classification and understood the meaning of the sentence in a single step. But here we are doing this two step approach..
"""

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Step 1: Creating Embedding Features from the text

#load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#load data:
data = load_dataset("rotten_tomatoes")

#Convert Text input into embeddings:
train_embeddings = model.encode(data['train']['text'], show_progress_bar = True)

test_embeddings = model.encode(data['test']['text'])

print(train_embeddings.shape) #(8530, 768): This shows that each of the  8,530 input documents has an embedding dimension of 768 and therefore each embedding contains 768 numerical values.

#Step 2: Using Classifier to classify text as positive or negative review.

clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data['train']['label'])

#Prediction
y_pred = clf.predict(test_embeddings)

def evaluate_performance(y_true, y_pred):
    performance = classification_report(y_true, y_pred, target_names = ['Negative Review', 'Positive Review'])
    return performance


print(evaluate_performance(data['test']['label'], y_pred))