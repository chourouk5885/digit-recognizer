"""Live demo: draw a digit, the CNN predicts it as you draw.

    python app.py              # opens http://127.0.0.1:7860
    python app.py --share      # also creates a temporary public link to send to someone
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from digits.inference import DigitPredictor  # noqa: E402

MODEL_PATH = ROOT / "models" / "mnist_cnn.pt"
if not MODEL_PATH.exists():
    sys.exit(f"No trained model at {MODEL_PATH}. Run `python train.py` first.")

predictor = DigitPredictor(MODEL_PATH)


def predict(editor_value):
    """Gradio's Sketchpad sends {'background', 'layers', 'composite'}; we use the composite."""
    image = editor_value.get("composite") if isinstance(editor_value, dict) else editor_value
    if image is None:
        return None, None
    probs, img28 = predictor.predict(image)
    if not probs:
        return None, None
    preview = Image.fromarray(img28).resize((196, 196), Image.NEAREST)  # show the pixels the model sees
    return probs, np.asarray(preview)


with gr.Blocks(title="Handwritten Digit Recognizer") as demo:
    gr.Markdown(
        "# ✏️ Handwritten Digit Recognizer\n"
        "Draw a single digit (0–9) in the box. A convolutional neural network trained on MNIST "
        "predicts it **live** as you draw. Use the trash icon to clear the canvas."
    )
    with gr.Row():
        with gr.Column(scale=3):
            pad = gr.Sketchpad(
                label="Draw here",
                type="numpy",
                image_mode="RGBA",
                canvas_size=(280, 280),
                fixed_canvas=True,
                brush=gr.Brush(default_size=20, colors=["#000000"], color_mode="fixed"),
                layers=False,
                transforms=(),
            )
        with gr.Column(scale=2):
            label = gr.Label(num_top_classes=3, label="Prediction")
            seen = gr.Image(label="What the model sees (28×28)", interactive=False, height=220)
    gr.Markdown(
        "**Tips:** draw large and centred, one digit at a time. "
        "Before predicting, the drawing is cropped, scaled to 20×20 and centred in a 28×28 frame "
        "exactly like the MNIST training images."
    )
    pad.change(predict, inputs=pad, outputs=[label, seen], show_progress="hidden")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    demo.launch(share=parser.parse_args().share)
