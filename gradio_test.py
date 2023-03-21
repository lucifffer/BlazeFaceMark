import gradio as gr
from inference import inference

def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}

demo = gr.Interface(fn=inference, inputs="image", outputs="image")
demo.launch(server_name='0.0.0.0', server_port=3395)

