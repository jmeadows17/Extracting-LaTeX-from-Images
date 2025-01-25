---
license: mit
datasets:
- unsloth/LaTeX_OCR
language:
- en
base_model:
- meta-llama/Llama-3.2-1B
- google/siglip-so400m-patch14-384
tags:
- vlm
- vision
- multimodal
- AnyModal
---
# AnyModal/LaTeX-OCR-Llama-3.2-1B

**AnyModal/LaTeX-OCR-Llama-3.2-1B** is an experimental model designed to convert images of handwritten and printed mathematical equations into LaTeX representations. Developed within the [AnyModal](https://github.com/ritabratamaiti/AnyModal) framework, this model combines a `google/siglip-so400m-patch14-384` image encoder with the Llama 3.2-1B language model. It has been trained on 20% of the [unsloth/LaTeX_OCR dataset](https://huggingface.co/datasets/unsloth/LaTeX_OCR), which itself is a subset of the [linxy/LaTeX_OCR dataset](https://huggingface.co/datasets/linxy/LaTeX_OCR).

---

## Trained On

This model was trained on the [unsloth/LaTeX_OCR](https://huggingface.co/datasets/unsloth/LaTeX_OCR) dataset. The dataset contains 1% of samples from the larger [linxy/LaTeX_OCR dataset](https://huggingface.co/datasets/linxy/LaTeX_OCR), which includes images of mathematical equations annotated with their corresponding LaTeX expressions. The current model was trained on 20% of the unsloth dataset.

---

## How to Use

### Installation

Clone the AnyModal Project:

```bash
git clone https://github.com/ritabratamaiti/AnyModal.git
```

Navigate to the LaTeX OCR Project (see https://github.com/ritabratamaiti/AnyModal/tree/main/LaTeX%20OCR)

Install the required dependencies:

```bash
pip install torch transformers torchvision huggingface_hub tqdm matplotlib Pillow
```

### Inference

Below is an example of generating LaTeX code from an image:

```python
import llm
import anymodal
import torch
import vision
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

# Load language model and tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B",
    access_token="GET_YOUR_OWN_TOKEN_FROM_HUGGINGFACE",
    quantized=False,
    use_peft=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Load vision model components
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder(
    "google/siglip-so400m-patch14-384", use_peft=False
)

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Initialize MultiModalModel
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    prompt_text="The latex expression of the equation in the image is: ",
)

# Load pre-trained weights
if not os.path.exists("latex_ocr"):
    os.makedirs("latex_ocr")

snapshot_download("AnyModal/latex-ocr-Llama-3.2-1B", local_dir="latex_ocr")
multimodal_model._load_model("latex_ocr")

# Generate LaTeX expression from an image
image_path = "example_equation.jpg"  # Path to your image
image = Image.open(image_path).convert("RGB")
processed_image = image_processor(image, return_tensors="pt")
processed_image = {key: val.squeeze(0) for key, val in processed_image.items()}

# Generate LaTeX caption
generated_caption = multimodal_model.generate(processed_image, max_new_tokens=120)
print("Generated LaTeX Caption:", generated_caption)
```

---

## Project and Training Scripts

This model is part of the [AnyModal LaTeX OCR Project](https://github.com/ritabratamaiti/AnyModal/tree/main/LaTeX%20OCR).

- **Training Script**: [train.py](https://github.com/ritabratamaiti/AnyModal/blob/main/LaTeX%20OCR/train.py)  
- **Inference Script**: [inference.py](https://github.com/ritabratamaiti/AnyModal/blob/main/LaTeX%20OCR/inference.py)  

Refer to the project repository for further implementation details.

---

## Project Details

- **Vision Encoder**: The `google/siglip-so400m-patch14-384` model, pre-trained for visual feature extraction, was used as the image encoder.  
- **Projector Network**: A dense projection network aligns visual features with the Llama 3.2-1B text generation model.  
- **Language Model**: Llama 3.2-1B, a small causal language model, generates the LaTeX expression.

This implementation highlights a proof-of-concept approach using a limited training subset. Better performance can likely be achieved by training on more samples and incorporating a text-conditioned image encoder.