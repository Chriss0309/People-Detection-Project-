import cv2
import numpy as np
from PySide6 import QtGui
from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGridLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Signal, Slot, Qt, QThread
import sys
import math
from dvg_ringbuffer import RingBuffer
from ultralytics import YOLO
from deepface import DeepFace

model = YOLO('yolov8n-pose.pt')
oldCentroids = []
LINE = 320

class ProcessThread(QThread):
    changePixmapSignal = Signal(int, bool)

    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            if rb.is_full:
                counts = processVideoFeed(rb.pop())
                if counts:
                    if counts[0] != 0:
                        self.changePixmapSignal.emit(counts[0], True)
                    if counts[1] != 0:
                        self.changePixmapSignal.emit(counts[1], False)

class VideoThread(QThread):
    changePixmapSignal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            rb.append(frame)
            if ret:
                self.changePixmapSignal.emit(frame)
        cap.release()

class MyWidget(QWidget):
    peopleEntered = 0
    peopleLeft = 0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.width = 640
        self.height = 480

        self.logo = QLabel(self)
        pixmap = QPixmap('logo.png')
        self.logo.setPixmap(pixmap)
        self.logo.setFixedWidth(130)
        self.logo.setFixedHeight(85)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.logo.setStyleSheet(
            "background-color: black;"
        )

        self.entered = QLabel("Entered")
        self.entered.setFixedHeight(100)
        self.entered.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.entered.setStyleSheet(
            "font-family: 'Times New Roman', Times, serif;"
            "color: white;"
            "background-color: black;"
        )

        self.enteredCount = QLabel(self)
        self.enteredCount.setText("0")
        self.enteredCount.setFixedHeight(50)
        self.enteredCount.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.enteredCount.setStyleSheet(
            "font-family: 'Times New Roman', Times, serif;"
            "color: white;"
            "font-size: 30px;"
            "background-color: black;"
            "padding-top: 1px;"
        )

        self.left = QLabel("Left")
        self.left.setFixedHeight(50)
        self.left.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.left.setStyleSheet(
            "font-family: 'Times New Roman', Times, serif;"
            "color: white;"
            "background-color: black;"
        )

        self.leftCount = QLabel("0")
        self.leftCount.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.leftCount.setStyleSheet(
            "font-size: 30px;"
            "background-color: black;"
            "color: white;"
            "font-family: 'Times New Roman', Times, serif;"
            "padding-top: 1px;"
        )

        self.feed = QLabel(self)

        gridLayout = QGridLayout(self)
        gridLayout.setSpacing(0)
        gridLayout.addWidget(self.logo, 0, 0)
        gridLayout.addWidget(self.entered, 1, 0)
        gridLayout.addWidget(self.enteredCount, 2, 0)
        gridLayout.addWidget(self.left, 3, 0)
        gridLayout.addWidget(self.leftCount, 4, 0)
        gridLayout.addWidget(self.feed, 0, 1, 5, 1)

        self.thread = VideoThread()
        self.pThread = ProcessThread()
        self.thread.changePixmapSignal.connect(self.updateImage)
        self.pThread.changePixmapSignal.connect(self.updateCount)
        self.thread.start()
        self.pThread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def updateCount(self, count, enter):
        if enter:
            self.peopleEntered += count
            self.enteredCount.setText(str(self.peopleEntered))
        else:
            self.peopleLeft += count
            self.leftCount.setText(str(self.peopleLeft))

    @Slot(np.ndarray)
    def updateImage(self, frame):
        qtImg = self.cvToQt(frame)
        self.feed.setPixmap(qtImg)

    def cvToQt(self, frame):
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytes = ch * w
        qtFormat = QtGui.QImage(rgbImg.data, w, h, bytes, QtGui.QImage.Format_RGB888)
        p = qtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

def analyzeFace(frame, x1, y1, x2, y2):
    face = frame[y1:y2, x1:x2]

    try:
        if face.size > 0:
            analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
                age = analysis.get('age', None)
                gender = analysis.get('gender', None)
                print(f"Age: {age}, Gender: {gender}")
                return age, gender
            else:
                print("Analysis result is not valid")
                return None, None
        else:
            print("Face area is invalid (size 0)")
            return None, None
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None, None

def deleteLostCentroids(oldCentroids):
    i = 0
    lostList = []
    while i != len(oldCentroids):
        if oldCentroids[i]['framesLost'] == 10:
            lostList.append(i)
        i += 1
    count = 0
    for l in lostList:
        l -= count
        oldCentroids.pop(l)
        count += 1

def manageLostCentroids(oldCentroids):
    i = 0
    while i != len(oldCentroids):
        if not oldCentroids[i]['matched']:
            oldCentroids[i]['framesLost'] += 1
        i += 1

def setAllCentroidsToUnmatched(centroids):
    i = 0
    while i != len(centroids):
        centroids[i]['matched'] = False
        i += 1

def matchCentroids(oldCentroids, newCentroids):
    global LINE
    distances = []
    for i in range(len(newCentroids)):
        distance = []
        for j in range(len(oldCentroids)):
            distance.append(calculateDistance(newCentroids[i], oldCentroids[j]))
        distances.append(distance)
    
    cols = []
    rows = []
    for i in range(len(distances)):
        smallest = 1000
        matched = False
        row = 0
        col = 0
        for j in range(len(distances)):
            if j not in cols:
                for k in range(len(oldCentroids)):
                    if k not in rows:
                        if smallest > distances[j][k]:
                            smallest = distances[j][k]
                            if smallest < 350 and newCentroids[j]['walkingDirection'] == oldCentroids[k]['walkingDirection']:
                                row = k
                                col = j
                                matched = True

        if matched:
            newCentroids[col]['matched'] = True
            oldCentroids[row]['matched'] = True
            x = newCentroids[col]['x']
            y = newCentroids[col]['y']
            oldCentroids[row]['x'] = x
            oldCentroids[row]['y'] = y
            oldCentroids[row]['framesLost'] = 0
            oldCentroids[row]['walkingDirection'] = newCentroids[col]['walkingDirection']
            cols.append(col)
            rows.append(row)

    registerNewCentroids(oldCentroids, newCentroids)

    count = [0, 0]

    for i in range(len(oldCentroids)):
        if oldCentroids[i]['enter'] and oldCentroids[i]['x'] > LINE:
            if oldCentroids[i]['matched']:
                oldCentroids[i]['enter'] = False
                count[0] += 1
        elif not oldCentroids[i]['enter'] and oldCentroids[i]['x'] < LINE:
            if oldCentroids[i]['matched']:
                oldCentroids[i]['enter'] = True
                count[1] += 1

    return count

def registerNewCentroids(oldCentroids, newCentroids):
    global LINE
    for i in range(len(newCentroids)):
        if not newCentroids[i]['matched']:
            if newCentroids[i]['x'] < LINE:
                oldCentroids.append({'x': newCentroids[i]['x'], 'y': newCentroids[i]['y'], 'matched': True, 'framesLost': 0, 'walkingDirection': newCentroids[i]['walkingDirection'], 'enter': True})
            else:
                oldCentroids.append({'x': newCentroids[i]['x'], 'y': newCentroids[i]['y'], 'matched': True, 'framesLost': 0, 'walkingDirection': newCentroids[i]['walkingDirection'], 'enter': False})

def calculateDistance(a, b):
    return math.sqrt(math.pow(a['x'] - b['x'], 2) + math.pow(a['y'] - b['y'], 2))

def processVideoFeed(frame):
    global oldCentroids
    global model 

    newCentroids = []

    results = model.predict(frame, classes=0, conf=0.7, verbose=False, imgsz=(416, 256))

    if oldCentroids: 
        setAllCentroidsToUnmatched(oldCentroids)
    
    i = 0
    for box in results[0].boxes:
        boxNumpy = box.numpy()
        coords = boxNumpy.xyxy
        x1 = int(coords[0, 0])
        y1 = int(coords[0, 1])
        x2 = int(coords[0, 2])
        y2 = int(coords[0, 3])

        age, gender = analyzeFace(frame, x1, y1, x2, y2)

        walkingDirection = ''

        if results[0].keypoints.numpy().xy[i, 3][0] == 0:
            walkingDirection = 'Right' 
        if results[0].keypoints.numpy().xy[i, 4][0] == 0:
            walkingDirection = 'Left'

        newCentroids.append({'x': (x1+x2)/2, 'y': (y1+y2)/2, 'matched': False, 'walkingDirection': walkingDirection})

        i += 1

    count = []

    if newCentroids: 
        count = matchCentroids(oldCentroids, newCentroids)
    if oldCentroids: 
        manageLostCentroids(oldCentroids)
        deleteLostCentroids(oldCentroids)

    return count

if __name__=="__main__":
    rb = RingBuffer(capacity=1, dtype=np.ndarray, allow_overwrite=True)
    app = QApplication(sys.argv)
    a = MyWidget()
    a.show()
    sys.exit(app.exec())
