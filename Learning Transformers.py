import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


#Load Model and Tokenizer
llm_model = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(llm_model, device_map = "auto", torch_dtype = "auto", trust_remote_code = True)

tokenizer = AutoTokenizer.from_pretrained(llm_model)

#Pipeline
generator =  pipeline("text-generation", model = model, tokenizer = tokenizer, return_full_text = False, max_new_tokens = 200, do_sample = False)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

output = generator(prompt)

print(output[0]["generated_text"])

prompt = "The Capital of France is"

#Tokenize the input:
input_ids =  tokenizer(prompt, return_tensors = "pt")

input_ids = {key:value.to(model.device) for key, value in input_ids.items()}

#Get Model output before lm_head
model_output =  model.model(**input_ids)


#Output of the Lm_head
lm_head_output = model.lm_head(model_output[0])

#
# token_id = lm_head_output[0, -1].argmax(-1)
#
# print(tokenizer.decode(token_id))

token_id = lm_head_output[0, -1].argmax(-1)

print("Token ID:", token_id)
print("Decoded:", repr(tokenizer.decode(token_id)))


print(lm_head_output[0].shape) # LM-Output

print(model_output[0].shape)
