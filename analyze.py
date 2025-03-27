import torch
import torchaudio
import numpy as np
import nltk
import Levenshtein
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import cmudict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 📌 Download CMU Pronouncing Dictionary
nltk.download("cmudict")
pron_dict = cmudict.dict()

# 🎙 Load Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# 🌟 Streamlit Page Configuration
st.set_page_config(page_title="AI Read-Aloud Analyzer", page_icon="📖", layout="centered")

# 🎨 Custom CSS for better UI styling
st.markdown("""
    <style>
    .stApp {background: linear-gradient(to right, #ffffff, #e3f2fd);}
    .title {color: #004d99; font-size: 28px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# 🔄 Function to resample audio to 16kHz
def resample_audio(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = transform(waveform)
    return waveform.squeeze().numpy()

# 🎙 Convert speech to text
def transcribe_audio(file_path):
    try:
        speech = resample_audio(file_path)
        input_values = processor(torch.tensor(speech, dtype=torch.float32).unsqueeze(0), return_tensors="pt", sampling_rate=16000).input_values
        
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# 🔡 Convert text to phonemes
def text_to_phonemes(text):
    words = text.lower().split()
    return {word: pron_dict.get(word, [["UNK"]])[0] for word in words}

# ✅ Compare phonemes and suggest corrections
def compare_phonemes(expected_phonemes, spoken_phonemes):
    word_feedback = {}
    total_accuracy = 0
    total_words = len(expected_phonemes)

    for word, expected in expected_phonemes.items():
        spoken = spoken_phonemes.get(word, ["UNK"])
        edit_distance = Levenshtein.distance(" ".join(expected), " ".join(spoken))
        max_length = max(len(expected), len(spoken))
        accuracy = 1 - (edit_distance / max_length) if max_length > 0 else 0
        total_accuracy += accuracy

        correction = f"Try saying '{word}' as /{' '.join(expected)}/" if accuracy < 0.8 else "Good pronunciation!"
        word_feedback[word] = {"Expected": " ".join(expected), "Spoken": " ".join(spoken), "Accuracy": round(accuracy * 100, 2), "Correction": correction}

    overall_accuracy = round((total_accuracy / total_words) * 100, 2) if total_words > 0 else 0
    return overall_accuracy, word_feedback

# 📊 Generate charts for pronunciation analysis
def generate_graphs(word_feedback, overall_accuracy):
    words = list(word_feedback.keys())
    accuracies = [feedback["Accuracy"] for feedback in word_feedback.values()]

    # 📊 Bar Chart (Word-by-Word Accuracy)
    bar_chart = px.bar(x=words, y=accuracies, labels={'x': 'Words', 'y': 'Pronunciation Accuracy (%)'},
                        title="🔍 Word Accuracy", color=accuracies, color_continuous_scale='blues')

    # 🍕 Pie Chart (Pronunciation Distribution)
    pie_chart = go.Figure(data=[go.Pie(
        labels=["Correct Pronunciation", "Mispronounced"], 
        values=[overall_accuracy, 100 - overall_accuracy], 
        hole=0.4)])

    pie_chart.update_layout(title="⚖️ Pronunciation Distribution")

    return bar_chart, pie_chart

# 🏠 Streamlit UI
st.markdown("<div class='title'>📖 AI-Powered Read-Aloud Pronunciation Analyzer</div>", unsafe_allow_html=True)

# 📂 File Uploader
uploaded_file = st.file_uploader("Upload a WAV audio file:", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # 🎙 Transcribe uploaded speech
    spoken_text = transcribe_audio(file_path)

    # 📌 Phoneme Analysis
    spoken_phonemes = text_to_phonemes(spoken_text)

    # ✅ Pronunciation Accuracy
    overall_accuracy, word_feedback = compare_phonemes(spoken_phonemes, spoken_phonemes)

    # 📊 Generate Graphs
    bar_chart, pie_chart = generate_graphs(word_feedback, overall_accuracy)

    # 📌 Display Results
    st.subheader("📑 Pronunciation Analysis")
    st.write(f"🔠 **Overall Pronunciation Accuracy:** {overall_accuracy}%")

    # 📊 Show Graphs
    st.plotly_chart(bar_chart)
    st.plotly_chart(pie_chart)

    # 🔍 Word-by-Word Feedback
    st.subheader("🔍 Word-by-Word Feedback")
    for word, feedback in word_feedback.items():
        color = "green" if feedback["Accuracy"] > 80 else "red"
        st.markdown(f"**{word}**: <span style='color:{color}'>({feedback['Accuracy']}%)</span>", unsafe_allow_html=True)
        st.write(f"📌 **Expected Phonemes:** {feedback['Expected']}")
        st.write(f"🎙 **Spoken Phonemes:** {feedback['Spoken']}")
        st.write(f"🛠 **Correction Suggestion:** {feedback['Correction']}")

st.success("🎉 Upload an audio file and analyze your speech!")