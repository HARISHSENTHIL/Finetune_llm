import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "/home/ubuntu/Apps/LLM-Attributor/Harish-as-harry/llama3.2-1b-tamil"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

result = moderate([{"role": "user", "content": "Describe the role of scoring metrics in synthetic data validation ?"}])
print(result)
