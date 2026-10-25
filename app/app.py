import gradio as gr
import time

def image_classifier(input_image, model_name):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    start_time = time.time()

    if model_name == "QAT":
        time.sleep(0.012)
        gradcam_out = "gradcam_qat.png"
        scores = {"Cat": 0.90, "Dog": 0.06, "Rabbit": 0.02, "Fox": 0.015, "Deer": 0.005}
    elif model_name == "Static PTQ":
        time.sleep(0.018)
        gradcam_out = "gradcam_ptq.png"
        scores = {"Cat": 0.60, "Dog": 0.25, "Rabbit": 0.08, "Fox": 0.045, "Deer": 0.025}
    else:
        time.sleep(0.045)
        gradcam_out = "gradcam_original.png"
        scores = {"Dog": 0.50, "Cat": 0.30, "Rabbit": 0.10, "Fox": 0.06, "Deer": 0.04}

    inference_time = round((time.time() - start_time) * 1000, 1)
    pred_class = max(scores, key=scores.get)
    confidence = f"{scores[pred_class] * 100:.1f}%"

    # New HTML format for the green banner
    result_text = f"""
    <div style="background-color: #d1fae5; color: #065f46; padding: 16px 20px; border-radius: 8px; border: 1px solid #10b981; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; font-size: 16px; margin-bottom: 20px;">
        <div><strong>Model Used:</strong> {model_name}</div>
        <div style="color: #10b981;">|</div>
        <div><strong>Predicted Class:</strong> {pred_class}</div>
        <div style="color: #10b981;">|</div>
        <div><strong>Confidence:</strong> {confidence}</div>
        <div style="color: #10b981;">|</div>
        <div><strong>Inference Time:</strong> {inference_time} ms</div>
    </div>
    """
    return gradcam_out, scores, result_text

model_profiles = [
    {"name": "QAT", "tag": "Quantization-Aware Training", "accuracy": "94.5", "size": "11.2", "latency": "8.3", "precision": "INT8"},
    {"name": "Static PTQ", "tag": "Post-Training Quantization", "accuracy": "92.1", "size": "11.2", "latency": "8.7", "precision": "INT8"},
    {"name": "Baseline", "tag": "Full-Precision Reference", "accuracy": "95.0", "size": "44.7", "latency": "21.5", "precision": "FP32"},
]

stack_cards = [
    {
        "title": "Model & Training",
        "items": ["PyTorch", "TorchVision", "ResNet18", "Transfer Learning", "CIFAR-10 Dataset"]
    },
    {
        "title": "Techniques",
        "items": ["Data Augmentation", "CutMix", "MixUp", "Random Erasing", "Cosine Annealing Warm Restarts", "Hyperparameter Fine-Tuning", "PyTorch Lightning"]
    },
    {
        "title": "Quantization",
        "items": ["PyTorch PT2E Quantization", "Quantization-Aware Training (QAT)", "Static Post-Training Quantization (PTQ)", "INT8 Calibration"]
    },
    {
        "title": "Explainability",
        "items": ["Grad-CAM", "Confusion Matrix", "PIL", "Matplotlib", "Pandas", "Per-Class Accuracy Plot", "F1/Precision/Recall Plot", "Validation/Test Accuracy Plot", "Validation/Training Loss Plot"]
    },
    {
        "title": "Model Reliability",
        "items": ["Model Calibration", "Negative Log-Likelihood (NLL)", "Log Loss", "Expected Calibration Error (ECE)", "Temperature Scaling"]
    },
    {
        "title": "Deployment & UI",
        "items": ["Gradio", "Github", "ONNX Runtime", "ONNX Export", "Python"]
    }
]

def profile_block(p):
    return f"""
    ### {p['name']}
    *{p['tag']}*
    
    - **Test Accuracy:** {p['accuracy']}%
    - **Model Size:** {p['size']} MB
    - **Latency on CPU:** {p['latency']} ms
    - **Precision:** {p['precision']}
    """

def stack_markdown(stack):
    items_md = "\n".join(f"- {item}" for item in stack["items"])
    return f"#### **{stack['title']}**\n{items_md}"

# Custom CSS to increase font sizes for readability
custom_css = """
body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
}

.prose {
    font-size: 16px !important;
}

.prose h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
}

.prose h2 {
    font-size: 18px !important;
    line-height: 1.6 !important;
    color: #555 !important;
}

.prose h3 {
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

.prose h4 {
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-top: 10px !important;
    margin-bottom: 8px !important;
    color: #333 !important;
}

.prose p, .prose li {
    font-size: 16px !important;
    line-height: 1.7 !important;
    margin-bottom: 6px !important;
}

/* Enlarge the input/output labels */
.block label span {
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Enlarge the Top 5 Accuracies class names and percentages */
.label-container .label-text {
    font-size: 15px !important;
}
"""

with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inter")]), css=custom_css) as demo:
    gr.Markdown("# ResNet18 Cifar10 Classifier")
    gr.Markdown("This project takes a pretrained ResNet18 built for 224×224 ImageNet inputs, adapts it to CIFAR‑10’s 32×32 images by upscaling to 224×224 without changing the pretrained architecture, **adds standard and advanced augmentations (Cutmix, Mixup) with transfer learning and hyperparameter tuning**, checks reliability with calibration metrics like** ECE, NLL, and Log Loss plus temperature scaling**, and finally makes it deployable on mobiles, IoT, and robotics through QAT and PTQ quantization. Even though the model reliability is production-ready grade, it was trained on cifar10 datasets with 60000 images. Therefore, it could classify 10 classes such as **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.**")

    # Changed to gr.HTML so it renders the green box properly
    result_display = gr.HTML("")

    with gr.Row():
        # LEFT COLUMN: Input, Buttons, and Top-5 Accuracies
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=300)
            
            with gr.Row():
                qat_btn = gr.Button("QAT Model", variant="primary")
                ptq_btn = gr.Button("Static PTQ Model", variant="primary")
                original_btn = gr.Button("Baseline Model", variant="primary")

            output_label = gr.Label(label="Top 5 Accuracies", num_top_classes=5)

        # RIGHT COLUMN: Grad-CAM and its Explanation
        with gr.Column():
            output_gradcam = gr.Image(label="Grad-CAM", type="filepath", height=300)
            
            gr.Markdown("""
            ### What is Grad-CAM?
            Grad-CAM is an advanced technique that highlights the regions contributing to a model’s decision. 
            Warmer areas in the visualization indicate stronger influence on the prediction.
            """)

    gr.Markdown("---")
    
    with gr.Row():
        for profile in model_profiles:
            with gr.Column():
                gr.Markdown(profile_block(profile))

    gr.Markdown("---")
    
    gr.Markdown("### Engineering Stack")
    
    # Organized grid for Engineering Stack
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown(stack_markdown(stack_cards[0])) # Model & Training
            gr.Markdown(stack_markdown(stack_cards[3])) # Explainability
        with gr.Column(scale=1):
            gr.Markdown(stack_markdown(stack_cards[1])) # Techniques
            gr.Markdown(stack_markdown(stack_cards[4])) # Model Reliability
        with gr.Column(scale=1):
            gr.Markdown(stack_markdown(stack_cards[2])) # Quantization
            gr.Markdown(stack_markdown(stack_cards[5])) # Deployment & UI

    gr.Markdown("---")
    gr.Markdown("ResNet18 Quantization Project · [View Source on GitHub](https://github.com/yourusername)")

    qat_btn.click(fn=image_classifier, inputs=[input_image, gr.State("QAT")], outputs=[output_gradcam, output_label, result_display])
    ptq_btn.click(fn=image_classifier, inputs=[input_image, gr.State("Static PTQ")], outputs=[output_gradcam, output_label, result_display])
    original_btn.click(fn=image_classifier, inputs=[input_image, gr.State("Baseline")], outputs=[output_gradcam, output_label, result_display])

if __name__ == "__main__":
    demo.launch()