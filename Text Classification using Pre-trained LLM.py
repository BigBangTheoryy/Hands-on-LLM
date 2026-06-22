"this is just a sample and not the full code for text classification. That comes later. Here we are just importing a data sets."


from datasets import load_dataset






data = load_dataset("rotten_tomatoes")

#print(data)

print(data['train'][0, -1])