import cv2
from ultralytics import YOLO
import numpy as np 
import math

#Global Count for Total People Entering/Leaving
peopleEntered = 0
peopleLeft = 0
#newCentroids Stores Centroids in the Current Frame
oldCentroids = []
newCentroids = []
#Define the Line Here 
LINE = 620

#Calculate Euclidean Distance Between Centroids
def calculateDistance(centroidOne, centroidTwo):
        heightDistance = abs(centroidOne['y'] - centroidTwo['y'])
        baseDistance = abs(centroidOne['x'] - centroidTwo['x'])
        squaredHypotenuseDistance = pow(baseDistance, 2) + pow(heightDistance, 2)
        return int(math.sqrt(squaredHypotenuseDistance))

#Deletes Centroids if Lost
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

def matchCentroids(oldCentroids, newCentroids):
    global peopleEntered
    global peopleLeft 
    global LINE
    #Find the Distance New Centroids to all the Other Old Centroids
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
         #Search for the Shortest Distance in the Whole List, and Repeat the Process and Exclude Previously Matched Centroids (Includes Old and New)
         for j in range(len(distances)):
              if j not in cols: 
                for k in range(len(oldCentroids)):  
                    if k not in rows: 
                        if smallest > distances[j][k]:
                            smallest = distances[j][k]
                            #The Conditions to Match is Set Here
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
             #Appending Indexes to the cols and rows list will Prevent the Already Matched Centroids being Matched Again 
             cols.append(col)
             rows.append(row)
             
    registerNewCentroids(oldCentroids, newCentroids)

    for i in range(len(oldCentroids)):
        if oldCentroids[i]['enter'] is True and oldCentroids[i]['x'] > LINE: 
            oldCentroids[i]['enter'] = False
            peopleEntered += 1
        elif oldCentroids[i]['enter'] is False and oldCentroids[i]['x'] < LINE: 
            oldCentroids[i]['enter'] = True
            peopleLeft += 1 

def registerNewCentroids(oldCentroids, newCentroids):
    global LINE
    for i in range(len(newCentroids)): 
         if newCentroids[i]['matched'] == False:  
             if newCentroids[i]['x'] < LINE: 
                oldCentroids.append({'x': newCentroids[i]['x'], 'y': newCentroids[i]['y'], 'matched': True, 'framesLost':0, 'walkingDirection': newCentroids[i]['walkingDirection'], 'enter': True})
             else: 
                oldCentroids.append({'x': newCentroids[i]['x'], 'y': newCentroids[i]['y'], 'matched': True, 'framesLost':0, 'walkingDirection': newCentroids[i]['walkingDirection'],'enter': False})   

def setAllCentroidsToUnmatched(centroids): 
    i = 0
    while i != len(centroids):
        centroids[i]['matched'] = False
        i += 1   
         
if __name__ == "__main__":

    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture("scenarios.mp4")

    oldCentroids = []

    while cap.isOpened():
    
            i = 0
            newCentroids = []
            success, frame = cap.read()
            results = model.predict(frame, classes=0, conf=0.7, verbose=False, imgsz=(416, 256))

            if oldCentroids: 
                setAllCentroidsToUnmatched(oldCentroids)

            for box in results[0].boxes :
                boxNumpy = box.numpy()
                coords = boxNumpy.xyxy
                x1 = int(coords[0, 0])
                y1 = int(coords[0, 1])
                x2 = int(coords[0, 2])
                y2 = int(coords[0, 3])

                walkingDirection = 'Bot'
                #[i,3] is the Coordinates for Left Ear
                if results[0].keypoints.numpy().xy[i,3][0] == 0:
                    walkingDirection = 'Right' 
                #[i,4] is the Coordinates for Right Ear
                if results[0].keypoints.numpy().xy[i,4][0] == 0:
                    walkingDirection = 'Left'
 
                newCentroids.append({'x': (x1+x2)/2, 'y': (y1+y2)/2, 'matched': False, 'walkingDirection': walkingDirection})

                i += 1
            
            if newCentroids: 
                matchCentroids(oldCentroids, newCentroids)
            if oldCentroids: 
                manageLostCentroids(oldCentroids)
                deleteLostCentroids(oldCentroids)
            print(oldCentroids)

            frame = results[0].plot()
            cv2.putText(frame, f"People Entered : {peopleEntered}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"People Left : {peopleLeft}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.resize(frame, (800, 608))
            cv2.imshow("YOLOv8 Tracking", frame)
            if cv2.waitKey(0) == ord('q') : break

    cap.release()
    cv2.destroyAllWindows()
