import cv2
import numpy as np
import face_recognition
import os
from dotenv import load_dotenv
import re
from utils.aws_s3_config import AwsS3Image
from utils.custom_utils import getClassNameAndEncodings, renderRectangleAndText
from utils.voice_assistant import VoiceAssistant
from time import sleep
load_dotenv()

accessKey = os.environ.get("AWS_ACCESS_KEY")
secretKey = os.environ.get("AWS_SECRET_KEY")
region = os.environ.get("AWS_REGION")
bucketName = os.environ.get("AWS_BUCKET_NAME")

# cap = VideoCameraStream(int(os.environ.get("WEBCAM_CODE")))
classNames, encodedTargetFaces = getClassNameAndEncodings()
cap = cv2.VideoCapture(int(os.environ.get("LAPTOP_CAM_CODE")))
s3Uploader = AwsS3Image(accessKey, secretKey, bucketName, region)
voice = VoiceAssistant()

# flags for data state and system flow management
detectUnrecognizedFace = False
readyToSafe = False
name = ""
triggerVoice = False
while True:
    try:
        key = cv2.waitKey(1) & 0xFF  # define key variable for any keyboard event
        if key == ord('q'):  # break the look when user press q
            break
        elif triggerVoice and detectUnrecognizedFace:

            # when the system detect an unrecognized face and when user press 'a'
            # the bot will ask the face's owner name
            # print("I don't I have met you yet.")
            # print("What is your name?")
            print("Voice Assistant Activate")
            voice.speak("Hey, I don't think we have met yet.")
            voice.speak("What is your name?")
            name = voice.getAudio()
            voice.speak(f"Please wait {name}. I'm attempting to memorize your face")
            readyToSafe = True  # set the flag that data is ready to be safe

            # set the flag the let the system know that the face's owner is not that unrecognizable
            detectUnrecognizedFace = False
            continue  # go back to the start of the while loop
        elif key == ord('s') and readyToSafe:
            # when data is ready to be safe, the bot will ask the user to stay still for it to remember out face
            print("Ready To Upload")
            fileName = re.sub("\s+", "-", name.lower().strip())
            fileLoc = "resources/" + fileName + ".png"
            cv2.imwrite(fileLoc, image)  # temporarily create a png file to resources folder

            if os.path.exists(fileLoc):
                # refer the file from resources folder to be uploaded to AWS S3 Bucket
                className, encodedFace = s3Uploader.uploadImageToS3Bucket(str(fileLoc))
                classNames.append(className)  # add new class name fo the face after successful upload
                encodedTargetFaces.append(encodedFace)  # add new encoded face values
                name = ""  # reset name state
                readyToSafe = False  # reset the state that data is now has been consumed

        _, image = cap.read()
        if not readyToSafe:
            imageSmall = cv2.cvtColor(cv2.resize(image, (0, 0), None, 0.25, 0.25),
                                      cv2.COLOR_BGR2RGB)  # resize the image to have width and height of 40% of its original
            faceLocations = face_recognition.face_locations(imageSmall)  # find the location of faces
            encodedFaces = face_recognition.face_encodings(imageSmall,
                                                           faceLocations)  # Send in faces location as there will be many faces capture in a frame
            for encodedFace, faceLocation in zip(encodedFaces, faceLocations):
                matches = face_recognition.compare_faces(encodedTargetFaces, encodedFace)
                faceDistances = face_recognition.face_distance(encodedTargetFaces,
                                                               encodedFace)  # array of face distance to each corresponding encoded target face
                matchIndex = np.argmin(faceDistances)  # get the index of the lowest faceDistance value
                minimumDistance = faceDistances[matchIndex]
                if matches[matchIndex]:
                    nameThatMatch = classNames[matchIndex]
                    top, right, bottom, left = faceLocation[0]*4, faceLocation[1]*4, faceLocation[2]*4, faceLocation[3]*4
                    cv2.rectangle(image, (left, top), (right, bottom), (47, 194, 103), 5) # create bounding box from the face location
                    cv2.rectangle(image, (left-3, bottom+35), (right+50, bottom), (47, 194, 103), cv2.FILLED)
                    cv2.putText(image, f"{nameThatMatch} {round((100 - minimumDistance*100), 2)}%", (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    if 0 < len(face_recognition.face_locations(image)) <= 1:
                        detectUnrecognizedFace = True
                        triggerVoice = True
                        renderRectangleAndText(image, "Hi, what is your name?")
                        sleep(2)
        else:
            cv2.putText(image, "Please stay still for me to remember you", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Webcam Image", image)
    except:
        print("Camera is not connected --> Trying to Reconnect")
        detectUnrecognizedFace = False
        readyToSafe = False
        name = ""

cap.release()
cv2.destroyAllWindows()
