import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from urllib.request import urlopen
from dotenv import load_dotenv
import urllib.request
load_dotenv()
path = "resources"
targetImages = []
classNames = []
imageList = os.listdir(path) # go to the resources directory and list out all files as string
CAMERA_CODE = 1
for image in imageList:
    fileName = image.split(".")[0]
    if len(fileName) > 0:
        currentImage = cv2.imread(f"{path}/{image}") # read the image from the given path
        targetImages.append(currentImage)
        classNames.append(fileName) # get the first element which is the main name of the image file

def openReuqestedImage(url, readFlag=cv2.IMREAD_COLOR):
    res = (urllib.request.urlopen(url)).read()
    image = np.asarray(bytearray(res), dtype="uint8")
    return cv2.imdecode(image, readFlag)

def findEncoding(targetImages):
    encodedImages = list(map(lambda img: face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0], targetImages))
    return encodedImages

def markAttendance(name):
    with open("Attendance.csv", "r+") as file:
        dataList = file.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime("%H:%M:%S")
            file.writelines(f"\n{name}, {dateString}")
encodedTargetFaces = findEncoding(targetImages)

# print(os.environ["AWS_ACCESS_KEY"])
# print(os.environ["AWS_REGION"])
cap = cv2.VideoCapture(CAMERA_CODE)
while True:
    try:
        success, image = cap.read()
        imageSmall = cv2.cvtColor(cv2.resize(image, (0, 0), None, 0.25, 0.25),
                                  cv2.COLOR_BGR2RGB)  # resize the image to have width and height of 40% of its original

        faceLocations = face_recognition.face_locations(imageSmall)  # find the location of faces
        encodedFaces = face_recognition.face_encodings(imageSmall,
                                                       faceLocations)  # Send in faces location as there will be many faces capture in a frame

        for encodedFace, faceLocation in zip(encodedFaces, faceLocations):
            matches = face_recognition.compare_faces(encodedTargetFaces, encodedFace)
            faceDistances = face_recognition.face_distance(encodedTargetFaces, encodedFace) # array of face distance to each corresponding encoded target face
            matchIndex = np.argmin(faceDistances) # get the index of the lowest faceDistance value
            minimumDistance = faceDistances[matchIndex]

            if matches[matchIndex]:
                nameThatMatch = classNames[matchIndex]
                top, right, bottom, left = faceLocation[0]*4, faceLocation[1]*4, faceLocation[2]*4, faceLocation[3]*4
                cv2.rectangle(image, (left, top), (right, bottom), (47, 194, 103), 5) # create bounding box from the face location
                cv2.rectangle(image, (left-3, bottom+35), (right+150, bottom), (47, 194, 103), cv2.FILLED)
                cv2.putText(image, f"{nameThatMatch} {round((100 - minimumDistance*100), 2)}%", (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                markAttendance(nameThatMatch)
            else:
                if len(face_recognition.face_locations(image)) > 0:
                    t, r, b, l = face_recognition.face_locations(image)[0]
                    cv2.rectangle(image, (l, t), (r, b), (47, 194, 103),5)  # create bounding box from the face location
        cv2.imshow("Webcam Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("Camera is not connected --> Trying to Reconnect")

cap.release()
cv2.destroyAllWindows()
