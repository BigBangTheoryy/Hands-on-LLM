from transformers import AutoModelForCausalLM, AutoTokenizer

#Creating Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map   = "auto",
    torch_dtype = "auto",
    trust_remote_code = True )


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


prompt = "I am very tired today after Gym because there was a lot of work to do in office"

#Converting prompt into token Ids
input_ids =   tokenizer(prompt, return_tensors = "pt")
# print(input_ids)

#Decoding the token ids created:
input_ids = input_ids['input_ids'][0]

for id in input_ids:
    print(f"{id} -> {tokenizer.decode(id)}")

#Generating Text




