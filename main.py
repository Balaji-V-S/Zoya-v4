from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 🔹 Choose your base model (any 2B–3.5B one)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"   # Example (7B, change to 3B)
# model_name = "microsoft/Phi-3-mini-4k-instruct"  # ~3.8B params
# model_name = "tiiuae/falcon-3b-instruct"         # ~3B params

# Load tokenizer & model (quantized for smaller RAM use if needed)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # Auto places on GPU if available
    torch_dtype="auto"        # Uses best precision (fp16/bf16 if GPU supports it)
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
)

# 🔹 Try chatting with Zoya
prompt = "Hello Zoya, how are you today?"
response = generator(prompt)[0]["generated_text"]

print("Zoya:", response)
