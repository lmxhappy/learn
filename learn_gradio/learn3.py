# coding: utf-8
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

def greet2(name):
    return "Hi " + name + "!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output)

    greet_btn2 = gr.Button("Greet2")
    greet_btn2.click(fn=greet2, inputs=name, outputs=output)


demo.launch(share=True)