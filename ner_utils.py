import gradio as gr
from transformers import pipeline

#Load the NER model
named_entity_recognizer= pipeline(task = 'ner', model = 'dslim/bert-base-NER')
# NER helper functions
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens
def ner(input):
    output = named_entity_recognizer(input)
    merged_tokens = merge_tokens(output)
    return {'text': input, 'entities': merged_tokens}


# NER App
NER = gr.Interface(
    fn = ner,
    inputs = [gr.Textbox(label = "Text to find entities", lines = 3)],
    outputs = [gr.HighlightedText(label = 'Text with entities')],
    allow_flagging = 'never',
    examples=[
        "My name is Nabi, I'm building NER Application",
    "My name is Emon, I live in Rajshahi and study at RUET"
    ]
)

# Add Markdown content
markdown_content_ner = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Named Entity Recognition APP</h1>
        <h3 style='color: #4682B4;'>Model: dslim/bert-base-NER</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
ner_with_markdown = gr.Blocks()
with ner_with_markdown:
    markdown_content_ner.render()
    NER.render()