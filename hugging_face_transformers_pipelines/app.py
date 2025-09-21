from transformers import pipeline
import gradio as gr
import tempfile
import soundfile as sf
import numpy as np

#NLP Pipelines

sentiment_class = pipeline("sentiment-analysis")
text_gen = pipeline("text-generation", model="gpt2-medium")
text_classify = pipeline("text-classification")
summarize = pipeline("summarization",model="facebook/bart-large-cnn")
translate = pipeline('translation', model='Helsinki-NLP/opus-mt-en-hi') #model for english to hindi translation
zero_shot = pipeline("zero-shot-classification")
feature_ex = pipeline("feature-extraction")
ques_ans = pipeline("question-answering", model="deepset/roberta-base-squad2")
mask_f = pipeline("fill-mask")

#Image Pipelines
image_class = pipeline("image-classification")
image_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
object_detect = pipeline("object-detection")

#Audio Pipelines
speech_rec = pipeline("automatic-speech-recognition")
audio_class = pipeline("audio-classification")
text_speech = pipeline("text-to-speech")

#NLP tasks functions
def sentiment_analysis(text):
  result = sentiment_class(text) #first prediction
  return result[0]['label']

def text_generation(text):
  return text_gen(text, temperature=0.7, max_new_tokens =100)[0]['generated_text']

def text_classification(text):
  result = text_classify(text)
  return result[0]['label']

def text_summarization(text):
  return summarize(text, do_sample=True, max_length = 100)[0]['summary_text']

def text_translation(text):
  return translate(text)[0]['translation_text']

def zero_shot_classification(text, labels):
    labels = [lbl.strip() for lbl in labels.split(",")]  # convert input into list
    result = zero_shot(text, candidate_labels=labels)
    return {lbl: score for lbl, score in zip(result["labels"], result["scores"])}

def feature_extraction(text):
  features = feature_ex(text)
  return str(features)

def question_answering(context, question):
  return ques_ans(question=question, context=context)

def mask_fill(text):
  return mask_f(text)

print(sentiment_analysis("I feel like I am on top of the world"))

#Image tasks functions
def image_classification(image):
  results = image_class(image)
  return results[0]["label"]

def image_to_text(image):
  return image_text(image)[0]["generated_text"]

def object_detection(image):
  results = object_detect(image)
  return results[0]["label"]

#Audio tasks functions
def speech_recognition(audio):
  return speech_rec(audio)

def audio_classification(audio):
  result = audio_class(audio)  # returns list of dicts
  # Take only the top prediction
  top = result[0]
  return {top['label']: float(top['score'])}

def text_to_speech(text):
  result = text_speech(text)  # {'audio': np.array, 'sampling_rate': int}
  # Get audio array and sampling rate from Hugging Face pipeline

  audio_array = result['audio']

  # Ensure float32 array and 1D
  audio_array = np.array(audio_array, dtype=np.float32).flatten()

  # Save to temp WAV file compatible with Gradio
  temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
  sf.write(temp_file.name, audio_array, samplerate=result['sampling_rate'], subtype='PCM_16')
  return temp_file.name

with gr.Blocks() as demo:
  gr.Markdown("<h2 style='text-align: center; font-weight: bold;'>Hugging Face Transformers Pipelines Demo</h2>")

  with gr.Tab("NLP"):
    sentiment_input = gr.Textbox(label="Sentiment Analysis")
    sentiment_output = gr.Label(label="Sentiment Analysis Output")
    sentiment_input.submit(sentiment_analysis, sentiment_input, sentiment_output)

    text_gen_input = gr.Textbox(label="Text Generation")
    text_gen_output = gr.Textbox(label="Text Generation Output")
    text_gen_input.submit(text_generation, text_gen_input, text_gen_output)

    text_class_input = gr.Textbox(label="Text Classification")
    text_class_output = gr.Label(label="Text Classification Output")
    text_class_input.submit(text_classification, text_class_input, text_class_output)

    text_sum_input = gr.Textbox(label="Text Summarization")
    text_sum_output = gr.Textbox(label="Summary of the text")
    text_sum_input.submit(text_summarization, text_sum_input, text_sum_output)

    text_trans_input = gr.Textbox(label="Text Translation")
    text_trans_output = gr.Textbox(label="Tranlated text in hindi")
    text_trans_input.submit(text_translation, text_trans_input, text_trans_output)

    # Zero-Shot Classification Section
    zero_shot_input = gr.Textbox(label="Text to Classify")
    labels_input = gr.Textbox(label="Candidate Labels (comma-separated)")
    zero_shot_output = gr.Label(label="Zero shot text classification output")

    # Bind function: both text and labels are passed
    zero_shot_input.submit(zero_shot_classification, [zero_shot_input, labels_input], zero_shot_output)

    feature_input = gr.Textbox(label="Feature Extraction")
    feature_output = gr.Textbox(label="Extracted features")
    feature_input.submit(feature_extraction, feature_input, feature_output)

    # Question Answering Section
    ques_ans_input = gr.Textbox(label="Question")
    context_input = gr.Textbox(label="Context", lines=4, placeholder="Paste context here...")
    ques_ans_output = gr.Textbox(label="Answer")

    # Bind function: both Question + Context go into the function
    ques_ans_input.submit(question_answering, [ques_ans_input, context_input], ques_ans_output)

    mask_input = gr.Textbox(label="Mask Filling")
    mask_output = gr.Textbox(label="Mask Filling output")
    mask_input.submit(mask_fill, mask_input, mask_output)


  with gr.Tab("Image"):
      image_class_input = gr.Image(type="pil", label="Image Classification")
      image_class_output = gr.Label(label="Image Classification Output")
      image_class_input.change(image_classification, image_class_input, image_class_output)

      image_to_text_input = gr.Image(type="pil", label="Image to Text")
      image_to_text_output = gr.Textbox(label="Image to Text Output")
      image_to_text_input.change(image_to_text, image_to_text_input, image_to_text_output)

      object_detect_input = gr.Image(type="pil", label="Object Detection")

      object_detect_output = gr.Label(label="Objects detected")
      object_detect_input.change(object_detection, object_detect_input, object_detect_output)

  with gr.Tab("Audio"):
      speech_rec_input = gr.Audio(type="filepath", label="Speech Recognition")
      speech_rec_output = gr.Textbox(label="Speech Recognition Output")
      speech_rec_input.change(speech_recognition, speech_rec_input, speech_rec_output)

      audio_class_input = gr.Audio(type="filepath", label="Audio Classification")
      audio_class_output = gr.Label(label="Audio Classification Output")
      audio_class_input.change(audio_classification, audio_class_input, audio_class_output)

      text_to_speech_input = gr.Textbox(label="Text to Speech")
      audio_output = gr.Audio(label="Generated Speech")
      text_to_speech_input.submit(text_to_speech, text_to_speech_input, audio_output)

demo.launch(debug=True)