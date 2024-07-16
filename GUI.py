from PySide6 import QtGui
from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGridLayout
from PySide6.QtGui import QPixmap, QImage
import sys
import cv2
from PySide6.QtCore import Signal, Slot, Qt, QThread, QTimer, QMutex, QWaitCondition
import numpy as np
import math
from dvg_ringbuffer import RingBuffer
from ultralytics import YOLO
import time
import queue

# Load pre-trained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

model = YOLO('yolov8n-pose.pt')
oldCentroids = []
LINE = 320


last_detection_time = 0
last_count_time = 0
frame_skip = 3 # Process every nth frame

frame_queue = queue.Queue(maxsize=5)  # Limit the number of frames in the queue
mutex = QMutex()
condition = QWaitCondition()

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_age_gender(face_img):
    try:
        print(f"Input face_img shape: {face_img.shape}")
        if len(face_img.shape) == 2 or face_img.shape[2] == 1:  # If grayscale
            print("Converting grayscale to RGB")
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # If RGBA
            print("Converting RGBA to RGB")
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
        elif face_img.shape[2] != 3:
            print(f"Unexpected number of channels: {face_img.shape[2]}")
            return "Unknown", "Unknown"
        
        print(f"Face image shape before resize: {face_img.shape}")
        face_img = cv2.resize(face_img, (227, 227))
        print(f"Face image shape after resize: {face_img.shape}")
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        gender_preds = genderNet.forward()
        if len(gender_preds) == 0 or len(gender_preds[0]) == 0:
            print("Error: Empty gender prediction")
            return "Unknown", "Unknown"
        gender = genderList[gender_preds[0].argmax()]

        ageNet.setInput(blob)
        age_preds = ageNet.forward()
        if len(age_preds) == 0 or len(age_preds[0]) == 0:
            print("Error: Empty age prediction")
            return "Unknown", gender
        age = ageList[age_preds[0].argmax()]

        return age, gender
    except Exception as e:
        print(f"Error in detect_age_gender: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Unknown", "Unknown"
    
    
def calculateDistance(centroidOne, centroidTwo):
    heightDistance = abs(centroidOne['y'] - centroidTwo['y'])
    baseDistance = abs(centroidOne['x'] - centroidTwo['x'])
    squaredHypotenuseDistance = pow(baseDistance, 2) + pow(heightDistance, 2)
    return int(math.sqrt(squaredHypotenuseDistance))

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
        if oldCentroids[i]['matched'] == False: 
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
                      
         if matched == True:
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
        if oldCentroids[i]['enter'] is True and oldCentroids[i]['x'] > LINE: 
             if oldCentroids[i]['matched'] == True: 
                oldCentroids[i]['enter'] = False
                count[0] += 1
        elif oldCentroids[i]['enter'] is False and oldCentroids[i]['x'] < LINE: 
             if oldCentroids[i]['matched'] == True:
                 oldCentroids[i]['enter'] = True
                 count[1] += 1

    return count 

def registerNewCentroids(oldCentroids, newCentroids):
    global LINE
    for i in range(len(newCentroids)): 
         if newCentroids[i]['matched'] == False:  
             if newCentroids[i]['x'] < LINE: 
                oldCentroids.append({
                    'x': newCentroids[i]['x'], 
                    'y': newCentroids[i]['y'], 
                    'matched': True, 
                    'framesLost':0, 
                    'walkingDirection': newCentroids[i]['walkingDirection'], 
                    'enter': True,
                    'age': newCentroids[i]['age'],
                    'gender': newCentroids[i]['gender']
                })
             else: 
                oldCentroids.append({
                    'x': newCentroids[i]['x'], 
                    'y': newCentroids[i]['y'], 
                    'matched': True, 
                    'framesLost':0, 
                    'walkingDirection': newCentroids[i]['walkingDirection'],
                    'enter': False,
                    'age': newCentroids[i]['age'],
                    'gender': newCentroids[i]['gender']
                })

class ProcessThread(QThread):
    changePixmapSignal = Signal(int, bool)

    def __init__(self):
        super().__init__()

    def run(self):
        global last_count_time
        while True: 
            mutex.lock()
            if frame_queue.empty():
                condition.wait(mutex)
            frame = frame_queue.get()
            mutex.unlock()
            
            if time.time() - last_count_time > 1: 
                counts = processVideoFeed(frame)
                last_count_time = time.time()
                if counts: 
                    if counts[0] != 0: 
                        self.changePixmapSignal.emit(counts[0], True)
                    if counts[1] != 0: 
                        self.changePixmapSignal.emit(counts[1], False)


class VideoThread(QThread):
    changePixmapSignal = Signal(np.ndarray)
    updateFaceBoxesSignal = Signal(list)

    def __init__(self):
        super().__init__()

    def run(self):
        global last_detection_time
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit the frame rate to 15 FPS
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))
                resultImg, faceBoxes = highlightFace(faceNet, frame)
                if time.time() - last_detection_time > 2:
                    self.updateFaceBoxesSignal.emit(faceBoxes)
                    last_detection_time = time.time()

                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1]):min(faceBox[3], frame.shape[0] - 1), max(0, faceBox[0]):min(faceBox[2], frame.shape[1] - 1)]
                    age, gender = detect_age_gender(face)
                    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if not frame_queue.full():
                    frame_queue.put(frame)
                    mutex.lock()
                    condition.wakeAll()
                    mutex.unlock()
                    
                self.changePixmapSignal.emit(resultImg)
            else:
                if not frame_queue.full():
                    frame_queue.put(frame)
                    mutex.lock()
                    condition.wakeAll()
                    mutex.unlock()

            frame_count += 1

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
        self.logo.setStyleSheet("background-color: black;")

        self.entered = QLabel("Entered")
        self.entered.setFixedHeight(100)
        self.entered.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.entered.setStyleSheet("font-family: 'Times New Roman', Times, serif;"
                                   "color: white;"
                                   "background-color: black;")

        self.enteredCount = QLabel(self)
        self.enteredCount.setText("0")
        self.enteredCount.setFixedHeight(50)
        self.enteredCount.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.enteredCount.setStyleSheet("font-family: 'Times New Roman', Times, serif;"
                                        "color: white;"
                                        "font-size: 30px;"
                                        "background-color: black;"
                                        "padding-top: 1px;")

        self.left = QLabel("Left")
        self.left.setFixedHeight(50)
        self.left.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        self.left.setStyleSheet("font-family: 'Times New Roman', Times, serif;"
                                "color: white;"
                                "background-color: black;")

        self.leftCount = QLabel("0")
        self.leftCount.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.leftCount.setStyleSheet("font-size: 30px;"
                                     "background-color: black;"
                                     "color: white;"
                                     "font-family: 'Times New Roman', Times, serif;"
                                     "padding-top: 1px;")

        self.ageGenderInfo = QLabel(self)
        self.ageGenderInfo.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.ageGenderInfo.setStyleSheet("font-size: 20px;"
                                         "background-color: black;"
                                         "color: white;"
                                         "font-family: 'Times New Roman', Times, serif;"
                                         "padding-top: 1px;")

        self.feed = QLabel(self)

        gridLayout = QGridLayout(self)
        gridLayout.setSpacing(0)
        gridLayout.addWidget(self.logo, 0, 0)
        gridLayout.addWidget(self.entered, 1, 0)
        gridLayout.addWidget(self.enteredCount, 2, 0)
        gridLayout.addWidget(self.left, 3, 0)
        gridLayout.addWidget(self.leftCount, 4, 0)
        gridLayout.addWidget(self.ageGenderInfo, 5, 0)
        gridLayout.addWidget(self.feed, 0, 1, 6, 1)

        self.videoThread = VideoThread()
        self.processThread = ProcessThread()
        self.videoThread.changePixmapSignal.connect(self.updateImage)
        self.videoThread.updateFaceBoxesSignal.connect(self.updateFaceBoxes)
        self.processThread.changePixmapSignal.connect(self.updateCount)
        self.videoThread.start()
        self.processThread.start()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAgeGenderInfo)
        self.timer.start(5000)  # Update every 5 seconds

    def closeEvent(self, event):
        self.videoThread.terminate()
        self.processThread.terminate()
        event.accept()

    def updateCount(self, count, enter): 
        if enter: 
            self.peopleEntered += count
            self.enteredCount.setText(str(self.peopleEntered))
        else: 
            if self.peopleLeft + count <= self.peopleEntered:
                self.peopleLeft += count
                self.leftCount.setText(str(self.peopleLeft))

    @Slot(np.ndarray)
    def updateImage(self, frame):
        qtImg = self.cvToQt(frame)
        self.feed.setPixmap(qtImg)

    
    def updateAgeGenderInfo(self):
        pass
    
    @Slot(list)
    def updateFaceBoxes(self, faceBoxes):
        age_gender_text = ""
        for i, faceBox in enumerate(faceBoxes):
            try:
                print(f"Processing face {i+1}: {faceBox}")
                face = self.feed.pixmap().toImage().copy(faceBox[0], faceBox[1], faceBox[2] - faceBox[0], faceBox[3] - faceBox[1])
                face_np = self.qimageToNp(face)
                if face_np.size == 0:
                    print(f"Skipping face {i+1}: Empty array")
                    continue
                print(f"Face {i+1} shape after qimageToNp: {face_np.shape}")
                age, gender = detect_age_gender(face_np)
                age_gender_text += f"Face {i+1} - Age: {age}, Gender: {gender}\n"
            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        self.ageGenderInfo.setText(age_gender_text)

    def cvToQt(self, frame):
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytes = ch * w
        qtFormat = QtGui.QImage(rgbImg.data, w, h, bytes, QtGui.QImage.Format_RGB888)
        p = qtFormat.scaled(self.width, self.height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def qimageToNp(self, qimage):
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        
        ptr = qimage.constBits()
        if ptr is None:
            print("Error: constBits() returned None")
            return np.array([])
        
        if isinstance(ptr, memoryview):
            buffer = ptr
        else:
            try:
                buffer = memoryview(ptr)
            except TypeError:
                print(f"Error: Unexpected type for constBits(): {type(ptr)}")
                return np.array([])
        
        arr = np.array(buffer, copy=False, dtype=np.uint8).reshape(height, bytes_per_line)
        rgb_arr = arr[:, :width*3].copy()
        print(f"RGB array shape: {rgb_arr.shape}")
        return rgb_arr
    
    
def processVideoFeed(frame): 
    global oldCentroids
    global model 

    newCentroids = []

    results = model.predict(frame, classes=0, conf=0.7, verbose=False, imgsz=(416, 256))

    if oldCentroids: 
        setAllCentroidsToUnmatched(oldCentroids)
    
    i = 0

    for box in results[0].boxes :
        boxNumpy = box.numpy()
        coords = boxNumpy.xyxy
        x1 = int(coords[0, 0])
        y1 = int(coords[0, 1])
        x2 = int(coords[0, 2])
        y2 = int(coords[0, 3])

        walkingDirection = ''

        if results[0].keypoints.numpy().xy[i,3][0] == 0:
            walkingDirection = 'Right' 
        if results[0].keypoints.numpy().xy[i,4][0] == 0:
            walkingDirection = 'Left'

        newCentroids.append({
            'x': (x1 + x2) / 2, 
            'y': (y1 + y2) / 2, 
            'matched': False, 
            'walkingDirection': walkingDirection,
            'age': "N/A",
            'gender': "N/A"
        })

        i += 1

    count = []

    if newCentroids: 
        count = matchCentroids(oldCentroids, newCentroids)
    if oldCentroids: 
        manageLostCentroids(oldCentroids)
        deleteLostCentroids(oldCentroids)

    return count

if __name__ == "__main__":
    rb = RingBuffer(capacity=1, dtype=np.ndarray, allow_overwrite=True)
    app = QApplication(sys.argv)
    a = MyWidget()
    a.show()
    sys.exit(app.exec())
