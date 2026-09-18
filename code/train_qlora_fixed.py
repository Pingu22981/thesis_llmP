import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

ap=argparse.ArgumentParser()
ap.add_argument("--data",required=True); ap.add_argument("--output",required=True)
ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--lr",type=float,default=2e-4)
ap.add_argument("--max_len",type=int,default=4096)
ap.add_argument("--seed",type=int,default=42)
a=ap.parse_args()
from transformers import set_seed
set_seed(a.seed)

BASE="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(BASE)
tok.pad_token=tok.eos_token

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
model=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16)
model=prepare_model_for_kbit_training(model)
lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=get_peft_model(model,lora)

ds=load_dataset("json",data_files=a.data,split="train")
def fmt(ex):
    return {"text":tok.apply_chat_template(ex["messages"],tokenize=False)}
ds=ds.map(fmt)

# only compute loss on the assistant's reply (everything after this header)
resp_template="<|start_header_id|>assistant<|end_header_id|>"
collator=DataCollatorForCompletionOnlyLM(response_template=resp_template,tokenizer=tok)

cfg=SFTConfig(output_dir=a.output,seed=a.seed,data_seed=a.seed,num_train_epochs=a.epochs,learning_rate=a.lr,
    per_device_train_batch_size=1,gradient_accumulation_steps=8,
    max_seq_length=a.max_len,logging_steps=5,save_strategy="no",
    bf16=True,gradient_checkpointing=True,dataset_text_field="text")
trainer=SFTTrainer(model=model,args=cfg,train_dataset=ds,tokenizer=tok,
    data_collator=collator)
trainer.train()
trainer.save_model(a.output)
tok.save_pretrained(a.output)
print(f"SAVED {a.output}")
