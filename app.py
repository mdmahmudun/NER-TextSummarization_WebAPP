import warnings
warnings.filterwarnings('ignore')
import gradio as gr

from ner_utils import ner_with_markdown
from summerizer_utils import summarizer_with_markdown


# Combine both the app
demo = gr.Blocks()
with demo:
    gr.TabbedInterface(
        [ner_with_markdown, summarizer_with_markdown],
        ['Named Entity Recognition', 'Text Summarization']
    )


if __name__ == "__main__":
    demo.launch()