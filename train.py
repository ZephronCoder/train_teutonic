import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn
import torch.nn.functional as F

import sys
from pathlib import Path

MODEL_PATH = Path("/root/bigben/models/quasar")
sys.path.insert(0, str(MODEL_PATH))

# ======================
# Configuration
# ======================
MODEL_PATH = "/root/bigben/models/quasar"
OUTPUT_DIR = "/root/bigben/outs/quasar_train"
DATASET_PATH = "/root/bigben/dataset/test.jsonl"

torch.backends.cuda.matmul.allow_tf32 = True
MAX_TOKENS = 1024


# ======================
# Custom Trainer
# ======================
class QuasarSFTTrainer(SFTTrainer):
    def __init__(self, *args, Quasar_loss=None, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


# ======================
# Load Model & Tokenizer
# ======================

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

model.config.use_cache = True
model.gradient_checkpointing_enable()

# ======================
# LoRA Configuration
# ======================
# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=1280,
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"
#     ],
#     modules_to_save=["down_proj"],
#     init_lora_weights="gaussian",
#     ensure_weight_tying=True,
# )

# model = get_peft_model(model, lora_config)

# ======================
# Dataset
# ======================

# def formatting_function(example):
#     """Convert messages to tokenized input_ids + labels"""
#     messages = example["messages"]
    
#     # Apply chat template if your model has one, otherwise just concatenate
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=False
#     )
    
#     # Tokenize
#     tokenized = tokenizer(
#         text,
#         truncation=True,
#         max_length=MAX_TOKENS,
#         padding=False,
#         return_tensors=None,   # Keep as list for dataset
#     )
    
#     # Create labels (standard causal LM: copy input_ids, set padding to -100)
#     labels = tokenized["input_ids"].copy()
    
#     # Optional: mask user prompt if you want assistant-only loss
#     # For now, we train on full sequence (your model can handle it)
    
#     return {
#         "input_ids": tokenized["input_ids"],
#         "attention_mask": tokenized["attention_mask"],
#         "labels": labels,
#     }

dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

print(f"Dataset size after filtering: {len(dataset)}")
print("Sample:", dataset[0])

# ======================
# Training Arguments
# ======================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,                    # More reasonable than 50
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,         # Effective batch size = 4
    learning_rate=1e-6,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.3},
    warmup_steps=10,
    logging_steps=5,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=5,
    save_only_model=True,
    optim="adamw_torch_fused",
    # assistant_only_loss=True,
    # evaluation
    # eval_strategy="steps",
    # eval_steps=200,
    report_to=["none"],
    # ddp_find_unused_parameters=True,
    # ddp_backend="nccl",
)

# ======================
# Trainer
# ======================
trainer = QuasarSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# ======================
# Train
# ======================
trainer.train(resume_from_checkpoint=False)

# Optional: Save final model
model.save_pretrained(f"{OUTPUT_DIR}/final_lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")