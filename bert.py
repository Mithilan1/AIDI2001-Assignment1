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

# Sentence table
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

sentences = [
    "Hello world!",
    "HELLO world!",
    "Hello, world!!!",
    "The number 1024 is important.",
    "Supercalifragilisticexpialidocious is a long word.",
    "A very very very very very long sentence with repeated words."
]

rows = []
for s in sentences:
    ids = tok(s, add_special_tokens=True)["input_ids"]
    tokens = tok.convert_ids_to_tokens(ids)
    rows.append({
        "text": s,
        "char_len": len(s),
        "token_len": len(tokens),
        "tokens": " ".join(tokens),
    })

df = pd.DataFrame(rows)
df

# Visualization
plt.figure()
plt.scatter(df["char_len"], df["token_len"])
plt.xlabel("Character length")
plt.ylabel("Token count (BERT)")
plt.title("Char length vs token count")
plt.show()

# tips dataset
tips = sns.load_dataset("tips").copy()


def row_to_text(r):
    return (f"Table size {r['size']} paid total {r['total_bill']:.2f} "
            f"with tip {r['tip']:.2f}. Smoker={r['smoker']}, "
            f"Day={r['day']}, Time={r['time']}, Sex={r['sex']}.")


tips["nl"] = tips.apply(row_to_text, axis=1)
tips["token_len"] = tips["nl"].apply(lambda s: len(tok(s)["input_ids"]))

tips[["nl", "token_len"]].head()

# Visualization
plt.figure()
plt.hist(tips["token_len"], bins=20)
plt.xlabel("Token count (BERT)")
plt.ylabel("Rows")
plt.title("Token lengths for tips rows (as text)")
plt.show()
