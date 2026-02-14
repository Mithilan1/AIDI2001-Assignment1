import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    pipeline,
    set_seed,
)

DEVICE = 0 if torch.cuda.is_available() else -1
print("CUDA:", torch.cuda.is_available())

# load pipeline
gpt_name = "distilgpt2"
gpt = pipeline("text-generation", model=gpt_name, device=DEVICE)

# same prompt generated 3 times
prompt = "I opened the fridge and found"

for i in range(3):
    out = gpt(prompt, max_new_tokens=40, do_sample=True,
              temperature=1.0, top_p=0.95)[0]["generated_text"]
    print(f"\n--- Run {i+1} ---\n{out}")

# compare two
stable = gpt(prompt, max_new_tokens=40, do_sample=False)[
    0]["generated_text"]  # greedy
creative = gpt(prompt, max_new_tokens=40, do_sample=True,
               temperature=1.2, top_p=0.9)[0]["generated_text"]

print("STABLE:\n", stable)
print("\nCREATIVE:\n", creative)

# concrete repitition
rep = gpt("The number 1024 is important because", max_new_tokens=80,
          do_sample=True, temperature=1.3, top_p=0.95)[0]["generated_text"]
rep_fixed = gpt("The number 1024 is important because", max_new_tokens=80, do_sample=True,
                temperature=1.3, top_p=0.95, no_repeat_ngram_size=3)[0]["generated_text"]

print("REPETITION EXAMPLE:\n", rep)
print("\nWITH no_repeat_ngram_size:\n", rep_fixed)
