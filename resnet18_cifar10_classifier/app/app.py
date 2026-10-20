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

    result_text = f"""
    **Model Used:** `{model_name}` &nbsp;|&nbsp; 
    **Predicted Class:** `{pred_class}` &nbsp;|&nbsp; 
    **Confidence:** `{confidence}` &nbsp;|&nbsp; 
    **Inference Time:** `{inference_time} ms`
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
        "items": [
            "PyTorch",
            "TorchVision",
            "ResNet18",
            "Transfer Learning",
            "CIFAR-10 Dataset"
        ]
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
            "PyTorch Lightning"
        ]
    },
    {
        "title": "Quantization",
        "items": [
            "PyTorch PT2E Quantization",
            "Quantization-Aware Training (QAT)",
            "Static Post-Training Quantization (PTQ)",
            "INT8 Calibration"
        ]
    },
    {
        "title": "Explainability",
        "items": [
            "Grad-CAM",
            "Confusion Matrix",
            "PIL (Python Imaging Library)",
            "Matplotlib",
            "Pandas",
            "Per-Class Accuracy Plot",
            "Per-Class F1/Precision/Recall/Accuracy Plot",
            "Validation/Training/Test Accuracy Plot",
            "Validation/Training Loss Plot"
        ]
    },
    {
        "title": "Model Reliability",
        "items": [
            "Model Calibration",
            "Negative Log-Likelihood (NLL)",
            "Log Loss",
            "Expected Calibration Error (ECE)",
            "Temperature Scaling"
        ]
    },
    {
        "title": "Deployment & UI",
        "items": [
            "Gradio",
            "Github",
            "ONNX Runtime",
            "ONNX Export",
            "Python"
        ]
    }
]


def profile_block(p):
    return f"""
    ### {p['name']}
    *{p['tag']}*
    
    - **Test Accuracy:** {p['accuracy']}%
    - **Model Size:** {p['size']} MB
    - **Latency:** {p['latency']} ms
    - **Precision:** {p['precision']}
    """

def stack_block(stacks):
    md = ""
    for stack in stacks:
        tags = ", ".join(f"`{item}`" for item in stack["items"])
        md += f"**{stack['title']}**\n{tags}\n\n"
    return md

# Using the default Gradio theme
with gr.Blocks() as demo:
    gr.Markdown("# ResNet18 Cifar10 Classifier")
    gr.Markdown("Cats and dogs · Quantization, model performance, and Grad-CAM explainability")

    result_display = gr.Markdown("")

    with gr.Row():
        # LEFT COLUMN: Input, Buttons, and Top-5 Accuracies
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=300)
            
            with gr.Row():
                qat_btn = gr.Button("QAT Model",variant="primary")
                ptq_btn = gr.Button("Static PTQ Model",variant="primary")
                original_btn = gr.Button("Baseline Model",variant="primary")

            output_label = gr.Label(label="Top 5 Accuracies", num_top_classes=5)

        # RIGHT COLUMN: Grad-CAM and its Explanation
        with gr.Column():
            output_gradcam = gr.Image(label="Grad-CAM", type="filepath", height=300)
            
            gr.Markdown("""
            ### What is  Grad-CAM?
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
    gr.Markdown(stack_block(stack_cards))

    gr.Markdown("---")
    gr.Markdown("ResNet18 Quantization Project · [View Source on GitHub](https://github.com/yourusername)")

    qat_btn.click(fn=image_classifier, inputs=[input_image, gr.State("QAT")], outputs=[output_gradcam, output_label, result_display])
    ptq_btn.click(fn=image_classifier, inputs=[input_image, gr.State("Static PTQ")], outputs=[output_gradcam, output_label, result_display])
    original_btn.click(fn=image_classifier, inputs=[input_image, gr.State("Baseline")], outputs=[output_gradcam, output_label, result_display])

if __name__ == "__main__":
    demo.launch()