# Load necessary libraries
# couse6-IncrementalCapstone.py
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          pipeline, TextStreamer)
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

# Task: 1 : Understand generative AI & LLMs
# LLM can be used to generate text, answer questions, and more. LLM can be used for automated marketing, content creation, and customer support.
# LanChain is a framework for building applications with LLMs. It provides a simple interface for working with LLMs and allows 
# user to easily create complex applications.

# Task: 2 : Desing the Advertisement Generator pipeline

# --- Model and tokenizer setup --------------------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"      # or "HuggingFaceH4/zephyr-7b-beta"

# Load the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# --- Text-generation pipeline --------------------------------
# Create a text-generation pipeline using the model and tokenizer

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,       # only generate up to 120 fresh tokens
    do_sample=True,
    temperature=0.9,          # sampling temperature,lower temperature means more deterministic output
    top_p=0.95,               # nucleus sampling, controls diversity of output
    repetition_penalty=1.15,  # penalize repeated phrases, eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id, # pad token for sequences shorter than max length
    streamer=TextStreamer(tokenizer, skip_prompt=True) # stream output token by token
)

# Wrap the pipeline in a LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=gen_pipe)

# --- ChatML-style prompt (Phi-3/Zephyr understand this natively) ----
AD_PROMPT = PromptTemplate(
    input_variables=["bike_specs", "discount_options", "theme"],
    template=(
        "<|system|>\n"
        "You are an award-winning copywriter who writes vibrant, eco-friendly ads.\n<|end|>\n"
        "<|user|>\n"
        "Write a punchy bike-rental advert (≤ 70 words).\n"
        "Theme: {theme}\n"
        "Bike features: {bike_specs}\n"
        "Promos: {discount_options}\n"
        "End with a strong call-to-action.\n<|end|>\n"
        "<|assistant|>"
    )
)

# Create a LangChain LLMChain with the prompt and the LLM
ad_chain = LLMChain(llm=llm, prompt=AD_PROMPT)

# Function to generate a bike rental advertisement
def generate_bike_rental_ad(bike_specs: str, discount_options: str, theme: str) -> str:
    """Return a short, engaging bike-rental advertisement."""
    return ad_chain.run(
        bike_specs=bike_specs,
        discount_options=discount_options,
        theme=theme
    ).strip()

# Example usage
bike_specs = input("Enter bike specifications (e.g., Electric, Lightweight, GPS Enabled): ")
discount_options = input("Enter discount options (e.g., 20% off for first-time users): ")
theme = input("Enter ad theme (e.g., Adventure in the City, Family Fun, Eco-Friendly Travel): ")
ad = generate_bike_rental_ad(bike_specs, discount_options, theme)
print("Generated ad:")
print(ad)

# Observations:
# The output will be a short, engaging advertisement based on the provided inputs.
# The generated ad will be printed to the console.
# The code above sets up a text generation pipeline using a pre-trained model and tokenizer,
# defines a prompt template for generating bike rental advertisements,
# and provides a function to generate an ad based on user inputs.
# The generated ad will be printed to the console.
# The code above sets up a text generation pipeline using a pre-trained model and tokenizer,
# defines a prompt template for generating bike rental advertisements,
# and provides a function to generate an ad based on user inputs.
