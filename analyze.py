import torch
import numpy as np
import nltk
import Levenshtein
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Download CMU Pronouncing Dictionary
nltk.download("cmudict")
pron_dict = cmudict.dict()

# Load Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Function to capture live speech using SpeechRecognition
def record_audio():
    st.write("🎙️ Recording... Speak now!")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Improve recognition
        audio = recognizer.listen(source)

    return audio

# Convert speech to text using SpeechRecognition
def transcribe_audio(audio):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio)  # Use Google's STT
        return text
    except sr.UnknownValueError:
        return "❌ Speech not recognized"
    except sr.RequestError:
        return "❌ Speech Recognition service unavailable"

# Convert text to phonemes
def text_to_phonemes(text):
    words = text.lower().split()
    return {word: pron_dict.get(word, [["UNK"]])[0] for word in words}

# Compare phonemes and suggest corrections
def compare_phonemes(expected_phonemes, spoken_phonemes):
    total_accuracy = 0
    total_words = len(expected_phonemes)
    word_feedback = {}

    for word, expected in expected_phonemes.items():
        spoken = spoken_phonemes.get(word, ["UNK"])
        edit_distance = Levenshtein.distance(" ".join(expected), " ".join(spoken))
        max_length = max(len(expected), len(spoken))
        accuracy = 1 - (edit_distance / max_length) if max_length > 0 else 0
        total_accuracy += accuracy

        word_feedback[word] = {
            "Expected": " ".join(expected),
            "Spoken": " ".join(spoken),
            "Accuracy": round(accuracy * 100, 2),
            "Correction": f"Try saying '{word}' as /{' '.join(expected)}/" if accuracy < 0.8 else "✅ Good pronunciation!"
        }

    overall_accuracy = round((total_accuracy / total_words) * 100, 2) if total_words > 0 else 0
    return overall_accuracy, word_feedback

# Calculate fluency score
def calculate_fluency(reference_text, duration=5):
    words_per_minute = (len(reference_text.split()) / duration) * 60
    return min(100, words_per_minute / 150 * 100)  # Normalize to percentage

# Function to plot pie chart
def plot_pie_chart(fluency_score, pronunciation_accuracy):
    plt.figure(figsize=(4, 4))
    labels = ["Fluency Score", "Pronunciation Accuracy"]
    sizes = [fluency_score, pronunciation_accuracy]  

    if sum(sizes) == 0:
        st.write("⚠️ Not enough data for pie chart (both values are zero).")
        return  

    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"])
    plt.title("Fluency vs. Pronunciation Accuracy")
    st.pyplot(plt)

# Function to plot bar chart
def plot_bar_chart(word_feedback):
    plt.figure(figsize=(6, 4))
    words = list(word_feedback.keys())
    accuracies = [word_feedback[word]["Accuracy"] for word in words]

    plt.bar(words, accuracies, color=["green" if acc > 80 else "red" for acc in accuracies])
    plt.xlabel("Words")
    plt.ylabel("Pronunciation Accuracy (%)")
    plt.title("Word-by-Word Pronunciation Accuracy")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Store trends across multiple attempts
fluency_scores = []
pronunciation_accuracies = []

# Streamlit UI
st.title("📖 AI-Powered Read-Aloud Pronunciation Analyzer")

# Input reference text
reference_text = st.text_area("Enter the reference text:", "The quick brown fox jumps over the lazy dog")

if st.button("🎤 Start Live Speech"):
    audio_data = record_audio()

    if audio_data is not None:
        spoken_text = transcribe_audio(audio_data)

        # Extract phonemes
        expected_phonemes = text_to_phonemes(reference_text)
        spoken_phonemes = text_to_phonemes(spoken_text)

        # Pronunciation accuracy and correction suggestions
        overall_accuracy, word_feedback = compare_phonemes(expected_phonemes, spoken_phonemes)

        # Fluency scoring
        fluency = calculate_fluency(reference_text, duration=5)

        # Store scores for trend graph
        fluency_scores.append(fluency)
        pronunciation_accuracies.append(overall_accuracy)

        # Display results
        st.subheader("📑 Pronunciation Analysis")
        st.write(f"✅ **You Said:** {spoken_text}")
        st.write(f"🔠 **Overall Pronunciation Accuracy:** {overall_accuracy}%")
        st.write(f"⚡ **Fluency Score:** {round(fluency, 2)}%")

        # Show Graphs
        st.subheader("📊 Visualization")
        plot_pie_chart(fluency, overall_accuracy)
        plot_bar_chart(word_feedback)

        # Word-by-word analysis
        st.subheader("🔍 Word-by-Word Feedback")
        for word, feedback in word_feedback.items():
            st.write(f"**{word}**: ({feedback['Accuracy']}%)")
            st.write(f"📌 **Expected Phonemes:** {feedback['Expected']}")
            st.write(f"🎙 **Spoken Phonemes:** {feedback['Spoken']}")
            st.write(f"🛠 **Correction Suggestion:** {feedback['Correction']}")
