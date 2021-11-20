import cv2
import face_recognition
import numpy as np
import urllib.request
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import face_recognition
import cv2

load_dotenv()
BACK_END_URL = os.environ.get("BACKEND_BASE_URL")

def markAttendance(name):
    with open("../Attendance.csv", "r+") as file:
        dataList = file.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime("%H:%M:%S")
            file.writelines(f"\n{name}, {dateString}")

def openRequestedImage(url, readFlag=cv2.IMREAD_COLOR):
    res = (urllib.request.urlopen(url)).read()
    image = np.asarray(bytearray(res), dtype="uint8")
    return cv2.imdecode(image, readFlag)

def findEncoding(targetImages):
    encodedImages = list(map(lambda img: face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0], targetImages))
    return encodedImages

def getClassNameAndEncodings():
    encodedTargetFaces = []
    classNames = []
    faceIdentities = requests.get(f"{BACK_END_URL}/api/face-identities/encoded").json()
    if(faceIdentities):
        classNames = list(map(lambda fi: fi["name"], faceIdentities))
        encodedTargetFaces = list(map(lambda fi: fi["encodedFace"], faceIdentities))
    return classNames, encodedTargetFaces

def renderRectangleAndText(image, text):
    t, r, b, l = face_recognition.face_locations(image)[0]
    cv2.rectangle(image, (l, t), (r, b), (47, 194, 103), 5)  # create bounding box from the face location
    cv2.rectangle(image, (l - 3, b + 35), (r + 50, b), (47, 194, 103), cv2.FILLED)
    cv2.putText(image, f"{text}", (l, b + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)