import streamlit as st
import numpy as np
import os
import tempfile
import soundfile as sf
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from scipy import signal

# Page config
st.set_page_config(page_title="speech Emotion Detection", layout="centered")

# Emotion Detector Class
class EmotionDetector:
    def __init__(self, model_name="Dpngtm/wav2vec2-emotion-recognition"):
        st.info("Loading model...")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Full label list in the model
        self.full_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.selected_emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
        self.selected_indices = [self.full_labels.index(e) for e in self.selected_emotions]

        st.success("Model loaded successfully!")

    def detect_emotions(self, audio_array, sample_rate):
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
            audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))
        if sample_rate != 16000:
            num_samples = round(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

        # Filter only selected emotions
        selected_probs = probs[self.selected_indices]
        total = selected_probs.sum().item()
        normalized = [round((p.item() / total) * 100, 2) for p in selected_probs]

        emotion_scores = dict(zip([e.capitalize() for e in self.selected_emotions], normalized))

        # Get top emotion
        top_idx = np.argmax(normalized)
        top_emotion = self.selected_emotions[top_idx].capitalize()

        return top_emotion, emotion_scores

# Load the model once
@st.cache_resource
def load_model():
    return EmotionDetector()

emotion_detector = load_model()

# UI
st.title("🎙️ speech Emotion Detection")
st.write("Upload a `.wav` audio file to detect emotions in your voice.")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name

    try:
        audio_array, sample_rate = sf.read(audio_path, always_2d=False)
        top_emotion, emotion_scores = emotion_detector.detect_emotions(audio_array, sample_rate)

        # Display output
        st.subheader("🎯 Detected Emotion")
        st.markdown(f"**{top_emotion}**")

        st.subheader("📋Percentages ")
        for emotion, percent in emotion_scores.items():
            st.markdown(f"- **{emotion}**: {percent}%")

    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        os.unlink(audio_path)
