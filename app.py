import os
import whisper
import streamlit as st
from pydub import AudioSegment
from openai import OpenAI
import math

st.set_page_config(
    page_title="Whisper based ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

api_key = st.text_input('OpenAI API Key')
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

audio_tags = {'comments': 'Converted using pydub!'}

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"

def split_audio(file_path, max_size_mb=25, format="mp3"):
    audio = AudioSegment.from_file(file_path, format=format.split('.')[-1].lower())
    max_size_bytes = max_size_mb * 1024 * 1024
    duration_seconds = len(audio) / 1000
    filesize_bytes = os.path.getsize(file_path)

    if filesize_bytes <= max_size_bytes:
        return [file_path]

    parts = math.ceil(filesize_bytes / max_size_bytes)
    chunk_length = duration_seconds / parts

    split_files = []
    for i in range(parts):
        start = i * chunk_length * 1000
        end = min((i + 1) * chunk_length * 1000, duration_seconds * 1000)
        split_audio = audio[start:end]
        split_file_path = f"{file_path.split('.')[0]}_part{i}.{format}"
        split_audio.export(os.path.join(download_path, split_file_path), format=format)
        split_files.append(os.path.join(download_path, split_file_path))

    return split_files


@st.cache_data(persist=True,show_spinner=True)
def to_mp3(audio_file, output_audio_file, upload_path, download_path):
    ## Converting Different Audio Formats To MP3 ##
    if audio_file.name.split('.')[-1].lower()=="wav":
        audio_data = AudioSegment.from_wav(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp3":
        audio_data = AudioSegment.from_mp3(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="ogg":
        audio_data = AudioSegment.from_ogg(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="wma":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"wma")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="aac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"aac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flac":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"flac")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="flv":
        audio_data = AudioSegment.from_flv(os.path.join(upload_path,audio_file.name))
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)

    elif audio_file.name.split('.')[-1].lower()=="mp4":
        audio_data = AudioSegment.from_file(os.path.join(upload_path,audio_file.name),"mp4")
        audio_data.export(os.path.join(download_path,output_audio_file), format="mp3", tags=audio_tags)
    return output_audio_file

@st.cache_data(persist=True,show_spinner=True)
def process_audio(filename, model_type, language_option, max_size_mb=25):
    max_size_bytes = max_size_mb * 1024 * 1024
    filesize_bytes = os.path.getsize(filename)

    if filesize_bytes > max_size_bytes:
        print(f"File size is larger than {max_size_mb}MB, splitting the audio file.")
        split_files = split_audio(filename)
    else:
        split_files = [filename]

    combined_transcript = ""
    for file in split_files:
        model = whisper.load_model(model_type)
        result = model.transcribe(file, language=language_option)
        combined_transcript += result["text"] + " "

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


@st.cache_data(persist=True,show_spinner=True)
def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file),"w") as f:
        f.write(transcript_data)

st.title("🗣 Automatic Speech Recognition using whisper by OpenAI ✨")
st.info('✨ Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV 😉')
uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

audio_file = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path,output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))
    with col3:
        language_option = st.selectbox("Choose a language", ["English", "Chinese"])

    if st.button("Generate Transcript"):
        with st.spinner(f"Generating Transcript... 💫"):
            transcript = process_audio(str(os.path.abspath(os.path.join(download_path,output_audio_file))), whisper_model_type.lower(), language_option)

            output_txt_file = str(output_audio_file.split('.')[0]+".txt")

            save_transcript(transcript, output_txt_file)
            output_file = open(os.path.join(transcript_path,output_txt_file),"r")
            output_file_data = output_file.read()

        if st.download_button(
                             label="Download Transcript 📝",
                             data=output_file_data,
                             file_name=output_txt_file,
                             mime='text/plain'
                         ):
            st.balloons()
            st.success('✅ Download Successful !!')

else:
    st.warning('⚠ Please upload your audio file 😯')


