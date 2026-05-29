from transformers import pipeline
import gradio as gr
import torch

device = 0 if torch.cuda.is_available() else -1

# Translator model
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    device=device
)

# Languages
languages = {
    "English": "eng_Latn",
    "Urdu": "urd_Arab",
    "Hindi": "hin_Deva",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Bengali": "ben_Beng",
    "Arabic": "arb_Arab",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Russian": "rus_Cyrl",
    "Turkish": "tur_Latn",
    "Korean": "kor_Hang",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn"
}

def translate(text, src, tgt):
    try:
        result = translator(
            text,
            src_lang=languages[src],
            tgt_lang=languages[tgt]
        )
        return result[0]["translation_text"]
    except Exception as e:
        return str(e)

with gr.Blocks() as app:

    gr.Markdown("# 🌍 AI Translator")

    src = gr.Dropdown(
        choices=list(languages.keys()),
        value="English",
        label="Source Language"
    )

    tgt = gr.Dropdown(
        choices=list(languages.keys()),
        value="Urdu",
        label="Target Language"
    )

    text = gr.Textbox(
        label="Enter Text",
        placeholder="Type text here..."
    )

    btn = gr.Button("Translate")

    output = gr.Textbox(label="Output")

    btn.click(
        translate,
        inputs=[text, src, tgt],
        outputs=output
    )

app.launch()
