import playsound
import speech_recognition as sr
from gtts import gTTS

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def getAudio(self):
        saidText="" # set initial state of the speech text is empty string
        with sr.Microphone() as source:
            audio = self.recognizer.listen(source) # listen to the user speaking to the microphone
            try:
                saidText = self.recognizer.recognize_google(audio)
            except Exception as error:
                print(f"Exception Error: {str(error)}")
        return saidText

    def speak(self, text:str):
        tts = gTTS(text = text, lang="en")
        filename = "voice-assistant.mp3"
        tts.save(filename)
        playsound.playsound(filename)

