import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import azure.cognitiveservices.speech as speechsdk
import streamlit as st

load_dotenv()


llm = AzureChatOpenAI(
    azure_endpoint=st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
    deployment_name=st.secrets.get("AZURE_DEPLOYMENT_NAME", os.getenv("AZURE_DEPLOYMENT_NAME")),
    api_version=st.secrets.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION")),
    openai_api_key=st.secrets.get("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"))

)

def speak_text(text, filename="speech.mp3"):
    speech_config = speechsdk.SpeechConfig(
        subscription=st.secrets.get("AZURE_SPEECH_KEY", os.getenv("AZURE_SPEECH_KEY")),
        region=st.secrets.get("AZURE_SPEECH_REGION", os.getenv("AZURE_SPEECH_REGION"))
    )
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(filename="output.mp3")

    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech was spoken")
    else:
        print(f"Error: {result.reason}")

def ask_and_speak(prompt):
    print(f"Prompt: {prompt}")
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"Response: {response.content}")
    speak_text(response.content)
    return response.content


st.header("Research Tool")

user_input = st.text_input("Enter your Prompt")


if st.button("Summerize"):
    result = ask_and_speak(user_input)
    st.write(result)
    with open("output.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")
