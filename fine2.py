import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load model and tokenizer with offloading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,          
    load_in_4bit=load_in_4bit,                           
)

# PEFT Model Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Use gradient checkpointing to save memory
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

EOS_TOKEN = tokenizer.eos_token

# Format the dataset prompts
def formatting_prompts_func(examples):
    expert_levels = examples["expert"]
    average_levels = examples["average"]
    beginner_levels = examples["beginner"]
    contents = examples["content"]

    texts = []
    for expert, average, beginner, content in zip(expert_levels, average_levels, beginner_levels, contents):
        text = f"""Instruction: Explain the following smart contract at three levels of expertise.\n
Beginner explanation:\n{beginner}\n
Average explanation:\n{average}\n
Expert explanation:\n{expert}\n
Smart Contract Code:\n{content}""" + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# Load and format the dataset
dataset = load_dataset("parquet", data_files="/home/ollama/Apps/finetune/filtered_Solidity-Dataset.parquet", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


trainer_stats = trainer.train()


{'train_runtime': 189.2367, 'train_samples_per_second': 0.634, 'train_steps_per_second': 0.317, 'train_loss': 0.6614961052934328, 'epoch': 0.0}                                                                       
