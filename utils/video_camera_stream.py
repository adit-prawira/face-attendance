import cv2
from threading import Thread, Lock


class VideoCameraStream:
    def __init__(self, cameraSrc: int, width: int = 640, height: int = 480):
        self._width = width  # set width of display window
        self._height = height  # set height of the display window
        self._capture = cv2.VideoCapture(cameraSrc)  # launched the camera
        self._threadStarted = True
        self._thread = Thread(target=self._update, args=())  # launch thread that will continuously collects frame
        self._readLock = Lock()
        self._thread.daemon = True  # kills threads when the program is terminated
        self._thread.start()  # start thread during initialization

    def _update(self):
        while self._threadStarted:
            _, self._frame = self._capture.read()
            self._readLock.acquire()
            self._newFrame = cv2.resize(src=self._frame, dsize=(self._width, self._height))
            self._readLock.release()

    def getFrame(self): return self._newFrame

    def stopStreamThread(self):
        self._threadStarted = False
        self._thread.join()

    def release(self):
        self._capture.release()
