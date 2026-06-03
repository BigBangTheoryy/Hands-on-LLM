from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "auto",
    torch_dtype = "auto",
    trust_remote_code = True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#Creating Pipeline
generator = pipeline(
    "text-generation",  # The first argument tells Hugging Face what task you want.
    model = model, # This tells the pipeline: Use this specific LLM for generation.
    tokenizer = tokenizer, #The tokenizer converts text into tokens., After generation, the tokenizer also converts tokens back into readable text.
    return_full_text = False,
    max_new_tokens = 800,
    do_sample = False)

message = [
    {"role": "user", "content": "Tell me about Goku vs Jiren"}
]

output = generator(message)
print(output[0]["generated_text"])

