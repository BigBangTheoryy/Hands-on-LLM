from transformers import AutoModelForCausalLM, AutoTokenizer



#Loading Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map   = "auto",
    torch_dtype = "auto",
    trust_remote_code = True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

print("Model device:", model.device)


prompt = "Which is better for gym: Target 1 body muscle a day or 2 muscles target a day in Gym? "

#Tokenize the input:
input_ids = tokenizer( prompt, return_tensors="pt")
print(input_ids)

input_ids = {k: v.to(model.device) for k, v in input_ids.items()}


#Generate_Text:
generate_output = model.generate(**input_ids, max_new_tokens=200)

print(tokenizer.decode(generate_output[0]))

print(model.device) #chatGPT suggested to include this as this helps a lot during debugging. his helps verify whether the model is running on: cpu, mps (Apple GPU), cuda (NVIDIA GPU)
