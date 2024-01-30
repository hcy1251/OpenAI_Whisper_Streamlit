import io
import whisper
import streamlit as st
from pydub import AudioSegment
from openai import OpenAI
import math
import tempfile
import os

# Streamlit app configuration
st.set_page_config(
    page_title="Whisper based ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

# API Key input and setting environment variable
api_key = st.text_input('OpenAI API Key')
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

# Audio tags for export
audio_tags = {'comments': 'Converted using pydub!'}

# Function to split large audio files into smaller parts
def split_audio(audio_data, max_size_mb=25):
    max_size_bytes = max_size_mb * 1024 * 1024
    duration_seconds = audio_data.duration_seconds  # Corrected this line
    filesize_bytes = len(audio_data.raw_data)

    if filesize_bytes <= max_size_bytes:
        return [audio_data]

    parts = math.ceil(filesize_bytes / max_size_bytes)
    chunk_length = duration_seconds / parts
    split_files = []

    for i in range(parts):
        start = i * chunk_length * 1000
        end = min((i + 1) * chunk_length * 1000, duration_seconds * 1000)
        split_audio = audio_data[start:end]
        split_files.append(split_audio)

    return split_files

# Function to convert audio file to mp3 format
def to_mp3(audio_file):
    audio_format = audio_file.name.split('.')[-1].lower()
    audio_data = AudioSegment.from_file(io.BytesIO(audio_file.getvalue()), format=audio_format)
    buffer = io.BytesIO()
    audio_data.export(buffer, format="mp3", tags=audio_tags)
    buffer.seek(0)
    return buffer

# Function to process audio using Whisper model
def process_audio(audio_data, model_type, language_option):
    # Convert BytesIO back to AudioSegment
    audio_segment = AudioSegment.from_file(audio_data, format="mp3")

    # Handle large files by splitting
    split_files = split_audio(audio_segment)
    combined_transcript = ""

    for audio in split_files:
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
            audio.export(temp_file.name, format="mp3")
            model = whisper.load_model(model_type)
            result = model.transcribe(temp_file.name, language=language_option)
            combined_transcript += result["text"] + " "

    # Additional processing for Chinese language
    if language_option == "Chinese":
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "請將以下文本進行斷句，以及對文本進行基本的處理（例如專有名詞的矯正、標點符號的添加等）："},
                {"role": "user", "content": combined_transcript}
            ]
        )
        combined_transcript = completion.choices[0].message.content

    return combined_transcript

# Streamlit UI components
st.title("🗣 Automatic Speech Recognition using Whisper by OpenAI ✨")
st.info('✨ Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV 😉')

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"])

# File processing
if uploaded_file is not None:
    with st.spinner(f"Processing Audio ... 💫"):
        processed_audio = to_mp3(uploaded_file)
        st.audio(processed_audio)

        whisper_model_type = st.selectbox("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))
        language_option = st.selectbox("Choose a language", ["English", "Chinese"])

        if st.button("Generate Transcript"):
            transcript = process_audio(processed_audio, whisper_model_type.lower(), language_option)
            output_txt_file = str(uploaded_file.name.split('.')[0]+".txt")

            with open(output_txt_file, "w") as f:
                f.write(transcript)

            with open(output_txt_file, "r") as f:
                output_file_data = f.read()

            st.download_button(label="Download Transcript 📝", data=output_file_data, file_name=output_txt_file, mime='text/plain')

else:
    st.warning('⚠ Please upload your audio file 😯')
