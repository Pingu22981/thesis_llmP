#!/usr/bin/env python3
"""QLoRA fine-tune Llama 3.1 8B on NL->PDDL. Saves LoRA adapter."""
import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_bw.jsonl")
    ap.add_argument("--output", default="lora_bw")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)   # for smoke test
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data, split="train")

    def formatting_func(example):
        return tok.apply_chat_template(example["messages"], tokenize=False)

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_seq_length=args.max_len,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok, formatting_func=formatting_func)
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"saved adapter -> {args.output}")

if __name__ == "__main__":
    main()
