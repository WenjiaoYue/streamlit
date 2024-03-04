import torch
import os
import shutil
import PIL
import streamlit as st
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# path to git-base model
git_base_model = "/workspace/AI-for-Enterprise/base_model/git-base"

# save finetuned model to
#git_ft_model = "/workspace/AI-for-Enterprise/finetuned_model/med-git"
git_ft_model = "/workspace/AI-for-Enterprise/finetuned_model/med-git-bf16-gpu"

# path to alpaca base model
alpaca_base_model = "/workspace/AI-for-Enterprise/base_model/decapoda-research-llama-7B-hf"

# path to finetuned med-alpaca model
med_alpaca_ft_model = "/workspace/AI-for-Enterprise/finetuned_model/med-alpaca"

# path to finetuned med-alpaca model with LORA
med_alpaca_ft_lora_model = "/workspace/AI-for-Enterprise/finetuned_model/med-alpaca-lora"

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_base_git_model():
    print(f"Loading base git model...")
    base_model = AutoModelForCausalLM.from_pretrained(git_base_model).eval().to(device)
    base_processor = AutoProcessor.from_pretrained(git_base_model)
    return base_model, base_processor

@st.cache_resource
def load_ft_git_model():
    print(f"Loading finetuned git model...")
    ft_model = AutoModelForCausalLM.from_pretrained(git_ft_model).eval().to(device)
    ft_processor = AutoProcessor.from_pretrained(git_ft_model)
    return ft_model, ft_processor

@st.cache_resource
def load_base_alpaca_model():
    print(f"Loading base alpaca model...")
    model = AutoModelForCausalLM.from_pretrained(alpaca_base_model).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(alpaca_base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer

@st.cache_resource
def load_ft_alpaca_model():
    print(f"Loading finetuned alpaca model...")
    model = AutoModelForCausalLM.from_pretrained(med_alpaca_ft_model).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(med_alpaca_ft_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer

@st.cache_resource
def load_ft_lora_alpaca_model():
    print(f"Loading LORA finetuned alpaca model...")
    model = AutoModelForCausalLM.from_pretrained(med_alpaca_ft_lora_model).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(med_alpaca_ft_lora_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer

def infer_git(img_file):
    # Load model and processor
    model, processor = load_ft_git_model() if st.session_state['git-finetuned'] else load_base_git_model()

    # Process input image to get pixel values
    with PIL.Image.open(img_file) as img:
        pixel_values = processor(images=img, return_tensors="pt").pixel_values

    # Move input data (pixel values) to device
    pixel_values = pixel_values.to(device)

    # Ask model to "read" X-Ray image and generate response 
    generated_ids = model.generate(pixel_values=pixel_values, max_length=256)

    # Detonkenize
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption

def infer_alpaca(question, caption):
    # Load model
    if st.session_state['LLM-model'] == "Alpaca base":
        model, tokenizer = load_base_alpaca_model()
    elif st.session_state['LLM-model'] == "Alpaca FT":
        model, tokenizer = load_ft_alpaca_model()
    elif st.session_state['LLM-model'] == "Alpaca LORA FT":
        model, tokenizer = load_ft_lora_alpaca_model()
    else:
        assert False, "LLM model is undefined!"

    # Create prompt
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
            f"### Instruction:\n{question}\n\n"
    if caption is not None and caption != "":
        prompt += f"### Input:\n{caption}\n\n"
    prompt += "### Response:\n"
    
    # Tokenzing prompt and move it to device
    inputs = tokenizer(prompt, padding="longest", max_length=128, return_tensors="pt").to(device)

    # Generating output
    print(f"Generating output...")
    output_ids = model.generate(**inputs, max_length=256, do_sample=False)

    # Decoding output
    print(f"Decoding output...")
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    return outputs
