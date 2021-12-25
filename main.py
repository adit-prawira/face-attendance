import cv2
import numpy as np
import face_recognition
import os
from dotenv import load_dotenv
import re
from utils.aws_s3_image import AwsS3Image
from utils.custom_utils import getClassNameAndEncodings
from utils.voice_assistant import VoiceAssistant
import logging
from threading import Thread, enumerate as enamurateThreads
from queue import Queue

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)

BUF_SIZE = 10
q = Queue(BUF_SIZE)
VOICE_ASSISTANT_THREAD_NAME = "VoiceAssistant"
FACE_RECOGNITION_THREAD_NAME = "FaceRecognition"
S3_IMAGE_UPLOADER_THREAD_NAME = "S3ImageUploader"
ALL_THREADS = "AllThreads"


class Package:
    def __init__(self, sentTo: str, sentFrom: str, consumed: bool, content: any):
        self.sentTo = sentTo
        self.sentFrom = sentFrom
        self.consumed = consumed
        self.content = content

    def json(self):
        return {
            "sentTo": self.sentTo,
            "sentFrom": self.sentFrom,
            "consumed": self.consumed,
            "content": self.content,
        }


class S3ImageUploaderThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(S3ImageUploaderThread, self).__init__()
        self.target = target
        self.name = name
        _accessKey = os.environ.get("AWS_ACCESS_KEY")
        _secretKey = os.environ.get("AWS_SECRET_KEY")
        _region = os.environ.get("AWS_REGION")
        _bucketName = os.environ.get("AWS_BUCKET_NAME")
        self.s3Uploader = AwsS3Image(_accessKey, _secretKey, _bucketName, _region)

    def run(self):
        while True:
            if not q.empty() and len(list(q.queue)) > 0:
                item = list(q.queue)[0]

                if isinstance(item, Package):
                    if item.json()["sentFrom"] == FACE_RECOGNITION_THREAD_NAME and\
                            item.json()["sentTo"] == self.getName():
                        item = q.get().json()["content"]
                        logging.debug(f"Getting {str(item)}: {str(q.qsize())} items in queue")
                        fileName = re.sub("\s+", "-", item["name"].lower().strip())
                        fileLoc = "resources/" + fileName + ".png"
                        cv2.imwrite(fileLoc, item["roi"])  # temporarily create a png file to resources folder

                        if os.path.exists(fileLoc):
                            # refer the file from resources folder to be uploaded to AWS S3 Bucket
                            className, encodedFace = self.s3Uploader.uploadImageToS3Bucket(str(fileLoc))
                            content = {
                                "className": className,
                                "encodedFace": encodedFace
                            }
                            packageToMain = Package(FACE_RECOGNITION_THREAD_NAME, self.getName(), True, content)
                            q.put(packageToMain)
                            logging.debug(f"Putting {str(item)}: {str(q.qsize())} items in queue")

                    elif item.json()["sentFrom"] == FACE_RECOGNITION_THREAD_NAME and\
                            item.json()["sentTo"] == ALL_THREADS:
                        logging.debug("Terminating...")
                        break


class VoiceAssistantThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(VoiceAssistantThread, self).__init__()
        self.target = target
        self.name = name
        self.voice = VoiceAssistant()
        self.faceName = ""

    def __cancel(self):
        self.voice.speak("Cancelling")
        item = Package(FACE_RECOGNITION_THREAD_NAME, self.getName(), True, {"cancel": True, "value": self.faceName})
        q.put(item)
        logging.debug(f"Putting {str(item)}: {str(q.qsize())} items in queue")

    def __save(self):
        self.voice.speak("Saving in")
        self.voice.speak("3")
        self.voice.speak("2")
        self.voice.speak("1")
        item = Package(FACE_RECOGNITION_THREAD_NAME, self.getName(), True, {"cancel": False, "value": self.faceName})
        q.put(item)
        logging.debug(f"Putting {str(item)}: {str(q.qsize())} items in queue")

    def __isCancelling(self, answer):
        return "cancel" in str(answer).strip().lower() or "shut up" in str(answer).strip().lower() or\
               "no" in str(answer).strip().lower()

    def __isProceeding(self, answer): return "yes" in str(answer).strip().lower()

    def run(self):
        while True:
            if not q.empty() and len(list(q.queue)) > 0:
                item = list(q.queue)[0]

                if isinstance(item, Package):
                    if item.json()["sentFrom"] == FACE_RECOGNITION_THREAD_NAME and\
                            item.json()["sentTo"] == self.getName():
                        item = q.get()
                        logging.debug(f"Getting {str(item)}: {str(q.qsize())} items in queue")

                        self.voice.speak("Face is unrecognized")
                        self.voice.speak("Who is that?")
                        self.faceName = self.voice.getAudio()
                        if self.__isCancelling(self.faceName):
                            self.__cancel()
                        # should check whether or not that name exists in the existing classNames
                        else:
                            self.voice.speak(f"Please stay still {self.faceName}. I'm attempting to memorize your face")
                            self.voice.speak("Proceed to safe?")
                            answer = self.voice.getAudio()
                            if self.__isProceeding(answer):
                                self.__save()
                            else:
                                self.__cancel()
                    elif item.json()["sentFrom"] == FACE_RECOGNITION_THREAD_NAME and \
                            item.json()["sentTo"] == ALL_THREADS:
                        logging.debug("Terminating...")
                        break


class FaceRecognition:
    def __init__(self, classNames, encodedTargetFaces):
        self.name = FACE_RECOGNITION_THREAD_NAME
        self.classNames, self.encodedTargetFaces = classNames, encodedTargetFaces
        self.cap = cv2.VideoCapture(int(os.environ.get("LAPTOP_CAM_CODE")))
        self.detectedOnce = False

    def getName(self): return self.name

    @classmethod
    def __getFaces(cls, frame):
        frameSmall = cv2.cvtColor(cv2.resize(frame, (0, 0), None, 0.25, 0.25),
                                  cv2.COLOR_BGR2RGB)  # resize the image to have width and height of 40% of its original
        faceLocations = face_recognition.face_locations(frameSmall)  # find the location of faces

        # Send in faces location as there will be many faces capture in a frame
        encodedFaces = face_recognition.face_encodings(frameSmall, faceLocations)
        return (faceLocations, encodedFaces)


    def __getMatchesAndDistances(self, encodedFace):
        # Array of boolean where some element may be True if similarities of encoded face is reaching
        # a certain threshold
        matches = face_recognition.compare_faces(self.encodedTargetFaces, encodedFace)

        # array of face distance to each corresponding encoded target face
        # this will be used for more explicit face matching decision making
        faceDistances = face_recognition.face_distance(self.encodedTargetFaces, encodedFace)

        return matches, faceDistances

    def __renderMatch(self, frame, matchedIndex, similarityPercentage, faceLocation):
        nameThatMatch = self.classNames[matchedIndex] # get value of classNames's element with index of matchindex

        # extract the points of faceLocation rectangle coordinates
        top, right, bottom, left = faceLocation[0] * 4, faceLocation[1] * 4, faceLocation[2] * 4, \
                                   faceLocation[3] * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (47, 194, 103),
                      5)  # create bounding box from the face location
        cv2.rectangle(frame, (left - 3, bottom + 35), (right + 50, bottom), (47, 194, 103),
                      cv2.FILLED)
        cv2.putText(frame, f"{nameThatMatch} {similarityPercentage}%",
                    (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    def __renderNoMatchFace(self, frame, faceLocation):
        top, right, bottom, left = faceLocation[0] * 4, faceLocation[1] * 4, faceLocation[2] * 4, \
                                   faceLocation[3] * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (47, 194, 103), 5)  # create bounding box from the face location
        cv2.rectangle(frame, (left - 3, bottom + 35), (right + 50, bottom), (47, 194, 103), cv2.FILLED)
        cv2.putText(frame, "Who are you?", (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return top, right, bottom, left

    def __kill(self):
        print("Threads to kill", enamurateThreads())
        terminationPackage = Package(ALL_THREADS, self.getName(), False, {"terminate": True})
        q.put(terminationPackage)
        logging.debug("Putting termination package")

    def run(self):
        while True:
            if not q.full():
                try:
                    _, frame = self.cap.read()
                    key = cv2.waitKey(1) & 0xFF  # define key variable for any keyboard event
                    if key == ord('k'):  # break the look when user press q
                        self.__kill()
                    elif key == ord('q'):
                        if len(enamurateThreads()) > 1:
                            logging.debug("Press 'k' to kill all the remaining threads")
                        else:
                            break

                    faceLocations, encodedFaces = self.__getFaces(frame)
                    for encodedFace, faceLocation in zip(encodedFaces, faceLocations):
                        matches, faceDistances = self.__getMatchesAndDistances(encodedFace)
                        matchedIndex = np.argmin(faceDistances)  # get the index of the lowest faceDistance value
                        minimumDistance = faceDistances[matchedIndex]  # get the value of the lowest distance value
                        similarityPercentage = round((100 - minimumDistance * 100), 2)

                        if matches[matchedIndex] and len(list(filter(lambda x: x , matches))) > 0 and\
                                similarityPercentage > 45:
                            self.__renderMatch(frame, matchedIndex, similarityPercentage, faceLocation)
                        else:
                            if not self.detectedOnce:
                                item = Package(VOICE_ASSISTANT_THREAD_NAME, self.getName(), False, True)
                                q.put(item)
                                logging.debug(f"Putting {str(item)}: {str(q.qsize())} items in queue")
                                self.detectedOnce = True
                            top, right, bottom, left = self.__renderNoMatchFace(frame, faceLocation)
                            if not q.empty() and len(list(q.queue)) > 0:
                                receive = list(q.queue)[0]
                                if isinstance(receive, Package):
                                    if receive.json()["sentFrom"] == VOICE_ASSISTANT_THREAD_NAME:
                                        receive = q.get().json()
                                        if receive["content"]["cancel"]:
                                            print("Reset detection status")
                                            self.detectedOnce = False
                                        else:
                                            logging.debug(f"Getting {str(receive)}: {str(q.qsize())} items in queue")
                                            x1, y1, x2, y2 = left, top, right, bottom
                                            roi = frame[y1:y1 + y2, x1:x1 + x2]
                                            content = {
                                                "roi": roi,
                                                "name": receive["content"]["value"]
                                            }
                                            logging.debug(f"ROI to put {roi} to S3")
                                            packageToS3 = Package(S3_IMAGE_UPLOADER_THREAD_NAME, self.getName(), False,
                                                                  content)
                                            q.put(packageToS3)
                                            logging.debug(f"Putting {str(packageToS3)}: {str(q.qsize())} items in queue")
                                    if receive.json()["sentFrom"] == S3_IMAGE_UPLOADER_THREAD_NAME:
                                        receive = q.get().json()["content"]
                                        self.classNames.append(receive["className"])  # add new class name fo the face after successful upload
                                        self.encodedTargetFaces.append(receive["encodedFace"])  # add new encoded face values
                                        self.detectedOnce = False
                    cv2.imshow("Webcam Image", frame)
                except:
                    print("Camera is not connected --> Trying to Reconnect")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    classNames, encodedTargetFaces = None, None
    while True:
        if getClassNameAndEncodings():
            classNames, encodedTargetFaces = getClassNameAndEncodings()
            break
    fr = FaceRecognition(classNames, encodedTargetFaces)
    v = VoiceAssistantThread(name=VOICE_ASSISTANT_THREAD_NAME)
    s3 = S3ImageUploaderThread(name=S3_IMAGE_UPLOADER_THREAD_NAME)
    s3.start()
    v.start()

    fr.run()

    s3.join()
    v.join()



