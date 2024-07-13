import gradio as gr
from transformers import pipeline

# Load the summarization model
summarizer = pipeline(task="summarization",
                      model="sshleifer/distilbart-cnn-12-6")


# Function to summarize input text
def summarize(input,
              min_length,
              max_length):
    output = summarizer(input,
                        min_length = min_length,
                        max_length = max_length)
    return output[0]['summary_text']

# Create the Gradio interface
SUMMARIZER = gr.Interface(
    fn=summarize,
    inputs=[gr.Textbox(label='Text to summarize', lines=6),
            gr.Slider(label='Min Length', minimum=10, maximum=50, value=10),
            gr.Slider(label='Max Length', minimum=50, maximum=200, value=100)],
    outputs=[gr.Textbox(label='Result', lines=3)],
    allow_flagging='never'
)

# Add Markdown content
markdown_content_summarizer = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Text Summarization with DistilBART-CNN</h1>
        <h3 style='color: #4682B4;'>Model: sshleifer/distilbart-cnn-12-6</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
summarizer_with_markdown = gr.Blocks()
with summarizer_with_markdown:
    markdown_content_summarizer.render()
    SUMMARIZER.render()