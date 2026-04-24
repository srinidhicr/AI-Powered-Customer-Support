"""Entry point. Run: python src/ui/app.py"""
 
from src.ui.layout import build
import gradio as gr
from src.ui.styles import CUSTOM_CSS



if __name__ == "__main__":
    build().launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    )