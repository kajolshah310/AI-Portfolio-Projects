# AI-Portfolio-Projects
Daily AI/ML learning journey — building projects with LLMs, Generative AI, and Machine Learning
# Hugging Face Transformers Pipelines
This repository contains examples of using Hugging Face's Transformers library for NLP, computer vision, and audio tasks. It demonstrates pipelines, pre-trained models, and generating outputs for common AI tasks.

---

## Installation

Install the Transformers library with optional dependencies:

```bash
pip install transformers[sentencepiece]
```

## NLP Pipelines

- **Sentiment Analysis**: Classify text as positive or negative.  
- **Text Generation**: Generate text based on a prompt.  
- **Text Classification**: Classify text into predefined categories.  
- **Summarization**: Generate concise summaries from longer text.  
- **Translation**: Translate text between languages (e.g., English → Hindi).  
- **Zero-shot Classification**: Classify text without prior training on specific labels.  
- **Feature Extraction**: Extract vector representations of text.  
- **Question Answering**: Answer questions based on a given context.  
- **Mask Filling**: Fill in blanks in text using predicted tokens.  

## Image Pipelines

- **Image Classification**: Classify objects in images.  
- **Image-to-Text**: Generate descriptive captions for images.  
- **Object Detection**: Detect objects and bounding boxes in images.  

## Audio Pipelines

- **Automatic Speech Recognition**: Convert speech to text.  
- **Audio Classification**: Classify audio into categories.  
- **Text-to-Speech**: Convert text into spoken audio.  

## Key Points

- Hugging Face Transformers provide pre-trained models for NLP, computer vision, and audio tasks.  
- Pipelines allow quick inference without fine-tuning.  
- Generation parameters like `temperature`, `max_length`, `top_k`, and `top_p` control output behavior.  
- This repository serves as a hands-on learning exercise for using Transformers.  

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)  
- [Hugging Face Model Hub](https://huggingface.co/models)  

