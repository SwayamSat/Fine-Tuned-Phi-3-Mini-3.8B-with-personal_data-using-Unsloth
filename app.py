from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

app = FastAPI()

# -----------------------
# Load Model ONCE
# -----------------------
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "swayam_sat-phi3-unsloth-lora"

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
FastLanguageModel.for_inference(model)

# -----------------------
# Request schema
# -----------------------
class ChatRequest(BaseModel):
    message: str

# -----------------------
# Chat endpoint
# -----------------------
@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"<|user|>\n{req.message}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# -----------------------
# Serve UI
# -----------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
