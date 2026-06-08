from transformers import AutoModel, AutoTokenizer

#Loading Model
model = AutoModel.from_pretrained("microsoft/deberta-v3-small")

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

text = "Hello World"

tokens = tokenizer(text, return_tensors = "pt")

#Process the Tokens
output = model(**tokens)[0]


print(output.shape) # torch.Size([1, 4, 768]) = we can read this as four tokens, each one embedded in a vector of 384 values.

#Decoding the tokens generated
decoding_tokens =  tokens['input_ids'][0]

for token in decoding_tokens:
    print(f"{token} -> {tokenizer.decode(token)}")