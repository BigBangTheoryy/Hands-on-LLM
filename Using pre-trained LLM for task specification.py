from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
from sklearn.metrics import classification_report
#Loading Dataset

data = load_dataset("rotten_tomatoes")


model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

#Load Model into pipeline

pipe = pipeline(model = model, tokenizer =  model, return_all_scores = True)

#Run_iterface to generate predictions, we use our model on the "test" split of our data:

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total = len(data['test'])):
    negative_score = output[0]['score']
    positive_score = output[2]['score']
    assignment =  np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

#function to evaluate our predictions performance

def evaluate_performance(y_true, y_pred):
    performance = classification_report(y_true, y_pred, target_names = ['Negative Reivew', "Positive Review"])
    print(performance)



evaluate_performance(data['test']['label'], y_pred)

