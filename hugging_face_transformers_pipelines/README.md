
---

```markdown
# Hugging Face Transformers Pipelines

This project explores Hugging Face's `transformers` library and its `pipeline` API for quick experimentation with state-of-the-art models.

## Tasks Covered

### NLP
- Sentiment analysis
- Text generation
- Text classification
- Summarization
- Translation
- Question answering
- Mask Filling
- Feature Extraction

### Computer Vision
- Image classification
- Image-to-text
- Object detection

### Audio
- Automatic speech recognition
- Text-to-speech
- Audio classification

## Gradio App

All experiments are wrapped in a Gradio app for interactive testing.

- [Gradio Demo (Hugging Face Space)](https://huggingface.co/spaces/kajolshah310/Transformers-Pipelines)  
- [GitHub Code](https://github.com/kajolshah310/AI-Experiments-Portfolio/tree/main/hugging_face_transformers)

You can try it yourself:  
- Type a sentence and run NLP models.  
- Upload an image and see classification, captioning, or object detection results.  
- Record or upload audio for transcription, classification, or text-to-speech.  

## Key Takeaways

- Hugging Face’s pipeline API allows using powerful pretrained models with just a few lines of code.
- Pretrained models provide strong baseline results without fine-tuning.
- A unified API across text, vision, and audio simplifies multi-modal experimentation.

## Getting Started Locally

```
Clone this repo and install dependencies:

```bash
git clone https://github.com/kajolshah310/AI-Experiments-Portfolio.git
cd AI-Experiments-Portfolio/hugging_face_transformers
pip install -r requirements.txt
```

## Run the Gradio app:
```bash
python app.py
```
