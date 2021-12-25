import os
import requests
from dotenv import load_dotenv
from time import sleep

load_dotenv()
BACK_END_URL = os.environ.get("BACKEND_BASE_URL")

def getClassNameAndEncodings():
    encodedTargetFaces = []
    classNames = []
    try:
        faceIdentities = requests.get(f"{BACK_END_URL}/api/face-identities/encoded").json()
        if (faceIdentities):
            classNames = list(map(lambda fi: fi["name"], faceIdentities))
            encodedTargetFaces = list(map(lambda fi: fi["encodedFace"], faceIdentities))
            return classNames, encodedTargetFaces
    except:
        print("Server Connection Error: Attempting to reconnect")
        sleep(2)
        getClassNameAndEncodings()

