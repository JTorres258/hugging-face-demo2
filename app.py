import os
os.environ["TRANSFORMERS_NO_PYTORCH"] = "1" 

from transformers import pipeline
import gradio as gr

model = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

demo.launch()