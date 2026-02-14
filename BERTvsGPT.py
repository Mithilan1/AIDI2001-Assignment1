print(gpt("Deep learning models are very", max_new_tokens=40,
      do_sample=True, temperature=1.0, top_p=0.9)[0]["generated_text"])

# Bert fail continuation
fill = pipeline("fill-mask", model="bert-base-uncased", device=DEVICE)

masked = "Deep learning models are very [MASK]."
fill(masked)[:5]

# Mismatch experimentation
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt_tok = AutoTokenizer.from_pretrained("distilgpt2")
gpt_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

enc = bert_tok("The capital of France is", return_tensors="pt")

try:
    out_ids = gpt_model.generate(enc["input_ids"], max_new_tokens=10)
    print("Decoded with BERT tokenizer:\n", bert_tok.decode(out_ids[0]))
except Exception as e:
    print("ERROR:", type(e).__name__, e)


# Summary
summ = pipeline("summarization",
                model="facebook/bart-large-cnn", device=DEVICE)

text1 = (
    "Toronto’s public transit faced delays due to winter storms. Riders reported longer commutes. "
    "Officials urged extra travel time. Service improved by Friday evening, though weekend maintenance may cause closures."
)

peng = sns.load_dataset("penguins").dropna().sample(8, random_state=0)
text2 = "Penguins rows:\n" + "\n".join(
    f"species={r.species}, island={r.island}, bill_length_mm={r.bill_length_mm}, "
    f"flipper_length_mm={r.flipper_length_mm}, body_mass_g={r.body_mass_g}, sex={r.sex}"
    for r in peng.itertuples(index=False)
)

text3 = (
    "Field notes: Most penguins seemed active.\nMeasurements:\n" +
    "\n".join(f"{r.species} on {r.island}: flipper={r.flipper_length_mm}mm, mass={r.body_mass_g}g"
              for r in peng.itertuples(index=False)) +
    "\nConclusion: Windy conditions may add measurement noise."
)


def summarize(t):
    return summ(t, max_length=70, min_length=25, do_sample=False)[0]["summary_text"]


print("SUMMARY 1:\n", summarize(text1))
print("\nSUMMARY 2:\n", summarize(text2))
print("\nSUMMARY 3:\n", summarize(text3))
