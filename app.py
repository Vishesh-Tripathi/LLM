from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load model and tokenizer
model_name = "distilgpt2"  # Small and Render-friendly
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

@app.route("/")
def home():
    return "LLM API is running!"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"output": result})
