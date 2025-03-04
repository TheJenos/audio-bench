import streamlit as st
import assemblyai as aai
from elevenlabs.client import ElevenLabs
from io import BytesIO
import pandas as pd
import time
import requests
from openai import OpenAI

st.title("Audio Benchmarking Tool")
st.markdown(
    """ 
    Welcome to the Audio Benchmarking Tool! 

    This tool is designed to help you evaluate and compare the performance of different audio processing algorithms. 

    You can upload your audio files, apply different algorithms, and see the results in real-time. We've also prepared a few examples for you to get started. Just click on the buttons above and discover what you can do with the Audio Benchmarking Tool. 
    """
)

open_ai_client = OpenAI(
    api_key=st.secrets['OPENAI_KEY']
)
aai.settings.api_key = st.secrets['ASSEMBLYAI_KEY']
assembly_ai_client = aai.Transcriber()
eleven_labs_client = ElevenLabs(
    api_key=st.secrets['ELEVENLABS_KEY']
)

record_audio = st.audio_input("Record a voice message")
upload_audio = st.file_uploader("Record a voice message", type=['mp3'])

if record_audio or upload_audio:
    st.audio(upload_audio or record_audio)

transcribe_btn = st.button("Transcribe")

assembly_ai_tab, open_ai_tab, eleven_labs_tab = st.tabs(["AssemblyAI", "OpenAI", "ElevenLabs"])

def eleven_labs_transcription(input_file):
    with eleven_labs_tab:
        with st.spinner('Processing...'):
            start_time = time.time()
            input_file = BytesIO(input_file.read())
            transcript = eleven_labs_client.speech_to_text.convert(
                file=input_file,
                model_id="scribe_v1"
            )
            end_time = time.time()
            duration = end_time - start_time
            st.markdown(f"Processing time: {duration:.2f} seconds")
            st.markdown(transcript.text)
            return duration

def assembly_ai_transcription(input_file):
    with assembly_ai_tab:
        with st.spinner('Processing...'):
            start_time = time.time()
            input_file = BytesIO(input_file.read())
            transcript = assembly_ai_client.transcribe(input_file)
            end_time = time.time()
            duration = end_time - start_time
            st.markdown(f"Processing time: {duration:.2f} seconds")
            st.markdown(transcript.text)
            return duration

def open_ai_transcription(input_file):
    with open_ai_tab:
        with st.spinner('Processing...'):
            start_time = time.time()
            transcript = open_ai_client.audio.transcriptions.create(
                model = "whisper-1",
                file = input_file
            )
            end_time = time.time()
            duration = end_time - start_time
            st.markdown(f"Processing time: {duration:.2f} seconds")
            st.markdown(transcript.text)
            return duration

if transcribe_btn: 
    input_file = upload_audio or record_audio

    if input_file is None: 
        st.error('Input File required')
    else:
        assembly_ai_duration = assembly_ai_transcription(input_file) 
        open_ai_duration = open_ai_transcription(input_file)
        eleven_labs_duration = eleven_labs_transcription(input_file)

        chart_data = pd.DataFrame(
            {
                "durations": [assembly_ai_duration, open_ai_duration, eleven_labs_duration],
                "services": ["AssemblyAI", "OpenAI", "ElevenLabs"],
            }
        )

        st.bar_chart(
            chart_data,
            x="services",
            y="durations",
        )

        st.success("Done!")