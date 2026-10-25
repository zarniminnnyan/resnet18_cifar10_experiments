
import os
import tempfile
import gradio as gr
import torch
import torch._inductor.codecache
import matplotlib.pyplot as plt
from html import escape

from config.exp_config import (
    temperature_path, 
    original_model_path,
    compiled_directory
    )
from src.grad_cam import process_input_image, GradCAM
from src.test import load_model_for_test
from utils.helper_utils import measure_inference_mb_ms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

COMPILED_PATHS = {
    "QAT": f"{compiled_directory}/AOT_compiled_QAT.pt2",
    "Static PTQ": f"{compiled_directory}/AOT_compiled_static.pt2",
    "Baseline": f"{compiled_directory}/AOT_compiled_original.pt2",
}

TEMPERATURE_EXPERIMENT_NAMES = {
    "QAT": "QAT",
    "Static PTQ": "Static",
    "Baseline": "Original",
}

_baseline_model = None
_gradcam = None
_compiled_models = {}


def _get_gradcam():
    global _baseline_model, _gradcam

    if _baseline_model is None:
        loaded = load_model_for_test(
            is_baseline=False,
            model_dir=original_model_path,
            experiment_name="exp5_v3",
            num_classes=10,
            device=device,
        )
        _baseline_model = loaded.model.to(device).eval()

    if _gradcam is None:
        _gradcam = GradCAM(
            _baseline_model,
            _baseline_model.layer4[-1],
        )

    return _baseline_model, _gradcam


def _load_compiled_model(model_name):
    if model_name not in _compiled_models:
        path = COMPILED_PATHS[model_name]

        if not os.path.isfile(path):
            raise gr.Error(f"Compiled model not found: {path}")

        import torch._inductor

        _compiled_models[model_name] = (
            torch._inductor.aoti_load_package(path)
        )

    return _compiled_models[model_name]


def _format_top5(predictions):
    """Convert Top-5 class scores into the format expected by gr.Label."""
    if predictions is None:
        return {}

    if isinstance(predictions, dict):
        result = {}
        for label, probability in predictions.items():
            if hasattr(probability, "item"):
                probability = probability.item()
            result[str(label)] = float(probability)
        return result

    if isinstance(predictions, (list, tuple)):
        result = {}

        for item in predictions:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                label, probability = item

                if hasattr(probability, "item"):
                    probability = probability.item()

                result[str(label)] = float(probability)
            else:
                result[str(item)] = 0.0

        return result

    return {}


def image_classifier(input_image, model_name):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    tmp_path = None

    try:
        model = _load_compiled_model(model_name)

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp:
            tmp_path = tmp.name

        input_image.save(tmp_path)

        with torch.inference_mode():
            pred_idx, pred_conf, top5, _, input_tensor = (
                process_input_image(
                    image_path=os.path.dirname(tmp_path),
                    temperature_path=temperature_path,
                    image_name=os.path.basename(tmp_path),
                    gradcam=None,
                    model=model,
                    classes=cifar10_classes,
                    experiment_name=TEMPERATURE_EXPERIMENT_NAMES[
                        model_name
                    ],
                    device=device,
                    add_temperature=True,
                )
            )

        timing = measure_inference_mb_ms(
            model=model,
            input_shape=input_tensor,
            is_production=True,
            device=device,
        )

        pred_idx = int(
            pred_idx.item()
            if hasattr(pred_idx, "item")
            else pred_idx
        )

        pred_conf = float(
            pred_conf.item()
            if hasattr(pred_conf, "item")
            else pred_conf
        )

        pred_class = cifar10_classes[pred_idx]
        inference_time = timing.get("model_time_ms", 0.0)

        result_text = f"""
        <div class="prediction-summary">
            <div class="summary-item">
                <span class="summary-label">Model</span>
                <strong>{escape(model_name)}</strong>
            </div>
            <div class="summary-item">
                <span class="summary-label">Prediction</span>
                <strong>{escape(pred_class)}</strong>
            </div>
            <div class="summary-item">
                <span class="summary-label">Confidence</span>
                <strong>{pred_conf * 100:.2f}%</strong>
            </div>
            <div class="summary-item">
                <span class="summary-label">Inference</span>
                <strong>{inference_time:.2f} ms</strong>
            </div>
        </div>
        """

        return _format_top5(top5), result_text

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Prediction failed: {exc}"
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_original_gradcam(input_image):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    tmp_path = None
    fig = None
    output_path = None

    try:
        baseline_model, gradcam = _get_gradcam()

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as tmp:
            tmp_path = tmp.name

        input_image.save(tmp_path)

        _, _, top5, fig, _ = process_input_image(
            image_path=os.path.dirname(tmp_path),
            temperature_path=temperature_path,
            image_name=os.path.basename(tmp_path),
            gradcam=gradcam,
            model=baseline_model,
            classes=cifar10_classes,
            experiment_name="Original",
            device=device,
            add_temperature=True,
        )

        if fig is None:
            raise gr.Error(
                "Grad-CAM did not generate a visualization."
            )

        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        fig.savefig(output_path, bbox_inches="tight")

        # Return BOTH outputs: the Grad-CAM image and Top-5 predictions.
        return output_path, _format_top5(top5)

    except gr.Error:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise

    except Exception as exc:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

        raise gr.Error(
            f"Grad-CAM failed: {exc}"
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        if fig is not None:
            plt.close(fig)


model_profiles = [
    {
        "name": "QAT",
        "tag": "Quantization-Aware Training",
        "accuracy": "92.53",
        "size": "11.28",
        "latency": "20.38",
        "precision": "INT8",
    },
    {
        "name": "Static PTQ",
        "tag": "Post-Training Quantization",
        "accuracy": "91.80",
        "size": "11.28",
        "latency": "21.51",
        "precision": "INT8",
    },
    {
        "name": "Baseline",
        "tag": "Full-Precision Reference",
        "accuracy": "94.31",
        "size": "44.80",
        "latency": "42.93",
        "precision": "FP32",
    },
]

stack_cards = [
    {
        "title": "Model & Training",
        "items": [
            "PyTorch",
            "TorchVision",
            "ResNet18",
            "Transfer Learning",
            "CIFAR-10 Dataset",
        ],
    },
    {
        "title": "Techniques",
        "items": [
            "Data Augmentation",
            "CutMix",
            "MixUp",
            "Random Erasing",
            "Cosine Annealing Warm Restarts",
            "Hyperparameter Fine-Tuning",
            "PyTorch Lightning",
        ],
    },
    {
        "title": "Quantization",
        "items": [
            "PyTorch PT2E Quantization",
            "Quantization-Aware Training (QAT)",
            "Static Post-Training Quantization (PTQ)",
            "INT8 Calibration",
        ],
    },
    {
        "title": "Explainability",
        "items": [
            "Grad-CAM",
            "Confusion Matrix",
            "PIL",
            "Matplotlib",
            "Pandas",
            "Per-Class Accuracy Plot",
            "F1/Precision/Recall Plot",
            "Validation/Test Accuracy Plot",
            "Validation/Training Loss Plot",
        ],
    },
    {
        "title": "Model Reliability",
        "items": [
            "Model Calibration",
            "Negative Log-Likelihood (NLL)",
            "Log Loss",
            "Expected Calibration Error (ECE)",
            "Temperature Scaling",
        ],
    },
    {
        "title": "Deployment & UI",
        "items": [
            "Gradio",
            "GitHub",
            "AOTInductor Compiler for CPU inference",
            "Python",
        ],
    },
]


def profile_block(p):
    return f"""
### {escape(p['name'])}

*{escape(p['tag'])}*

**Test Accuracy:** {escape(p['accuracy'])}%

**Model Size:** {escape(p['size'])} MB

**Latency on CPU:** {escape(p['latency'])} ms

**Precision:** {escape(p['precision'])}
"""


def stack_markdown(stack):
    title = escape(stack["title"])
    items = "\n".join(
        f"- {escape(item)}" for item in stack["items"]
    )
    return f"#### {title}\n\n{items}"


custom_css = """
:root {
    --bg: #f5f4f0;
    --surface: #ffffff;
    --surface-hover: #eeece6;
    --text: #252525;
    --muted: #626262;
    --accent: #e85d24;
    --accent-hover: #cf4c18;
}

html,
body {
    margin: 0 !important;
    min-height: 100% !important;
    background: #f5f4f0 !important;
    color: #252525 !important;
    color-scheme: light;
}

body {
    font-family: Inter, sans-serif !important;
}

.gradio-container,
.gradio-container main,
.gradio-container .main,
.gradio-container footer,
.gradio-container .contain {
    background: #f5f4f0 !important;
    color: #252525 !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1500px !important;
    min-height: 100vh !important;
    margin: 0 auto !important;
    padding: 32px !important;
    font-family: Inter, sans-serif !important;
    font-size: 16px !important;
}

.gradio-container * {
    box-sizing: border-box;
}

.gradio-container .prose,
.gradio-container .prose * {
    background: transparent;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4 {
    color: #252525 !important;
}

.gradio-container .prose h1 {
    font-size: 34px !important;
    font-weight: 750 !important;
    letter-spacing: -0.7px !important;
    line-height: 1.25 !important;
    margin: 0 0 18px 0 !important;
}

.gradio-container .prose h2 {
    font-size: 26px !important;
    font-weight: 700 !important;
    margin: 26px 0 14px 0 !important;
}

.gradio-container .prose h3 {
    font-size: 21px !important;
    font-weight: 650 !important;
    margin: 20px 0 10px 0 !important;
}

.gradio-container .prose h4 {
    color: #c94d1c !important;
    font-size: 18px !important;
    font-weight: 650 !important;
    margin: 18px 0 8px 0 !important;
}

.gradio-container .prose p,
.gradio-container .prose li {
    color: #55534f !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

.gradio-container .prose strong {
    color: #252525 !important;
}

.gradio-container .prose ul {
    padding-left: 22px !important;
    margin-top: 8px !important;
}

.gradio-container .prose li {
    margin-bottom: 5px !important;
}

.gradio-container .prose a {
    color: #c94d1c !important;
    text-decoration: none !important;
}

.gradio-container .prose a:hover {
    color: #a93d13 !important;
    text-decoration: underline !important;
}

.plain-section,
.plain-section > *,
.plain-section .prose,
.plain-section .prose * {
    background: transparent !important;
    box-shadow: none !important;
}

.plain-section {
    border: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
}

.plain-section h3,
.plain-section h4 {
    color: #252525 !important;
}

.plain-section p,
.plain-section li {
    color: #55534f !important;
}

.plain-section strong {
    color: #252525 !important;
}

.gradio-container label,
.gradio-container label span,
.gradio-container [data-testid="block-label"] {
    color: #33312d !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    color: #252525 !important;
    border: 1px solid #d5d2ca !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    border-color: #e85d24 !important;
    box-shadow: 0 0 0 2px rgba(232, 93, 36, 0.12) !important;
}

.gradio-container button.primary {
    background: #e85d24 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    min-height: 46px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    transition: background .2s ease,
                transform .2s ease,
                box-shadow .2s ease !important;
}

.gradio-container button.primary:hover {
    background: #cf4c18 !important;
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(232, 93, 36, .18) !important;
}

.gradio-container button.secondary {
    background: #e9e7e0 !important;
    color: #33312d !important;
    border: 1px solid #d5d2ca !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    font-size: 15px !important;
}

.gradio-container button.secondary:hover {
    background: #dedbd3 !important;
    color: #252525 !important;
    border-color: #d0c9bc !important;
}

.square-panel {
    width: 100% !important;
    overflow: hidden !important;
    border-radius: 15px !important;
    background: #eeede8 !important;
    border: 1px solid #d8d6cf !important;
}

.square-panel img,
.square-panel .image-container {
    max-width: 100% !important;
    border-radius: 12px !important;
}

.gradcam-panel {
    width: 100% !important;
    min-height: 600px !important;
    background: #eeede8 !important;
    border: 1px solid #d8d6cf !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    transition: background .25s ease,
                box-shadow .25s ease !important;
}

.gradcam-panel:hover {
    background: #e9e7e0 !important;
    box-shadow: 0 8px 24px rgba(65, 55, 40, .08) !important;
}

.gradcam-panel img,
.gradcam-panel .image-container {
    width: 100% !important;
    min-height: 560px !important;
    height: 100% !important;
    object-fit: contain !important;
    border-radius: 12px !important;
}

.prediction-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
    width: 100%;
    padding: 20px 0;
    background: transparent !important;
    color: #252525;
    border: none !important;
    border-radius: 0;
}

.summary-item {
    display: flex;
    flex-direction: column;
    gap: 7px;
    min-width: 100px;
}

.summary-label {
    color: #66635d !important;
    font-size: 12px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.summary-item strong {
    color: #252525 !important;
    font-size: 17px !important;
    font-weight: 700;
}

.gradio-container [data-testid="label"] {
    background: #eeede8 !important;
    color: #252525 !important;
    border: 1px solid #dedbd3 !important;
    border-radius: 13px !important;
}

.gradio-container [data-testid="label"] * {
    color: #33312d !important;
    font-size: 15px !important;
}

.gradio-container hr {
    display: none !important;
}

.footer-text,
.footer-text * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.footer-text {
    color: #706d66 !important;
    padding: 18px 0 0 0 !important;
    margin: 22px 0 0 0 !important;
    font-size: 13px !important;
}

footer {
    display: none !important;
}

@media (hover: none) {
    .gradcam-panel:active {
        background: #e4e1d8 !important;
        box-shadow: 0 6px 18px rgba(65, 55, 40, .10) !important;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 16px !important;
    }

    .gradio-container .prose h1 {
        font-size: 27px !important;
    }

    .gradio-container .prose h2 {
        font-size: 22px !important;
    }

    .gradio-container .prose p,
    .gradio-container .prose li {
        font-size: 15px !important;
    }

    .gradcam-panel {
        min-height: 360px !important;
    }

    .gradcam-panel img,
    .gradcam-panel .image-container {
        min-height: 320px !important;
    }

    .prediction-summary {
        gap: 18px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        transition-duration: .01ms !important;
        animation-duration: .01ms !important;
    }
}
"""


with gr.Blocks(
    theme=gr.themes.Default(
        font=[gr.themes.GoogleFont("Inter")],
        primary_hue="orange",
        neutral_hue="zinc",
    ),
    css=custom_css,
) as demo:

    gr.Markdown(
        "# Production-Ready Grade ResNet18 CIFAR-10 Classifier"
    )

    gr.Markdown(
        "This project adapts ResNet18 for CIFAR-10 using transfer learning, "
        "data augmentation, CutMix, MixUp, and hyperparameter tuning. "
        "It evaluates reliability through ECE, NLL, Log Loss, and temperature "
        "scaling, then explores deployment optimization with QAT and PTQ. "
        "The model classifies **airplane, automobile, bird, cat, deer, dog, "
        "frog, horse, ship, and truck.**"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                height=320,
                elem_classes="square-panel",
            )

            with gr.Row():
                qat_btn = gr.Button(
                    "QAT Model",
                    variant="primary",
                )
                ptq_btn = gr.Button(
                    "Static PTQ Model",
                    variant="primary",
                )

            original_btn = gr.Button(
                "Baseline Model",
                variant="primary",
            )

            gradcam_btn = gr.Button(
                "Request FP32 Original Grad-CAM",
                variant="primary",
            )

            gr.Markdown(
                "**About Grad-CAM:** The FP32 original model is the reference "
                "model used to generate Grad-CAM visualizations. QAT and Static "
                "PTQ models are derived from this model, then quantized and "
                "compiled with AOTInductor. This demo does not generate "
                "Grad-CAM directly from the compiled models."
            )

            output_label = gr.Label(
                label="Top 5 Predictions",
                num_top_classes=5,
            )

            result_display = gr.HTML()

        with gr.Column(scale=2, min_width=360):
            gr.Markdown("## Grad-CAM")

            output_gradcam = gr.Image(
                label=None,
                type="filepath",
                height=640,
                elem_classes="gradcam-panel",
            )

            gr.Markdown(
                "The heatmap visualizes regions influencing the original "
                "FP32 model. It may not explain the exact prediction made "
                "by a quantized or compiled model."
            )

    qat_btn.click(
        fn=lambda image: image_classifier(image, "QAT"),
        inputs=input_image,
        outputs=[output_label, result_display],
    )

    ptq_btn.click(
        fn=lambda image: image_classifier(image, "Static PTQ"),
        inputs=input_image,
        outputs=[output_label, result_display],
    )

    original_btn.click(
        fn=lambda image: image_classifier(image, "Baseline"),
        inputs=input_image,
        outputs=[output_label, result_display],
    )

    gradcam_btn.click(
        fn=generate_original_gradcam,
        inputs=input_image,
        outputs=[output_gradcam, output_label],
    )

    gr.Markdown("## Model Profiles")

    with gr.Row(equal_height=False):
        for profile in model_profiles:
            with gr.Column(
                scale=1,
                min_width=180,
                elem_classes="plain-section",
            ):
                gr.Markdown(profile_block(profile))

    gr.Markdown("## Engineering Stack")

    with gr.Row(equal_height=False):
        with gr.Column(
            scale=1,
            min_width=200,
            elem_classes="plain-section",
        ):
            gr.Markdown(stack_markdown(stack_cards[0]))
            gr.Markdown(stack_markdown(stack_cards[3]))

        with gr.Column(
            scale=1,
            min_width=200,
            elem_classes="plain-section",
        ):
            gr.Markdown(stack_markdown(stack_cards[1]))
            gr.Markdown(stack_markdown(stack_cards[4]))

        with gr.Column(
            scale=1,
            min_width=200,
            elem_classes="plain-section",
        ):
            gr.Markdown(stack_markdown(stack_cards[2]))
            gr.Markdown(stack_markdown(stack_cards[5]))

    gr.Markdown(
        "Resnet18 Cifar10 Classifier "
        "[View Source on GitHub](https://github.com/zarniminnnyan/resnet18_cifar10_experiments.git)",
        elem_classes="footer-text",
    )


if __name__ == "__main__":
    demo.launch()
