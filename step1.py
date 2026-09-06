import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
model.eval()

print(f"loaded in {time.time() - t0: .1f}s")

print("device:", DEVICE)
print("layers:", model.config.num_hidden_layers)
print("hidden size:", model.config.hidden_size)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the Capital of France?"}],
    tokenize=False,
    add_generation_prompt=True
)

print("\n--- what the model actually sees ---")
print(repr(prompt))

enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)

print("\n--- reply ---")
print(tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))