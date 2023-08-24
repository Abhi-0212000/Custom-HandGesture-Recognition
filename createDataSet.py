import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle



# updated mediapipe API's but they are avialable for Preview
""" 
model_path = './hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

frame = cv2.imread('./img.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_RGB = np.copy(frame)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

options = HandLandMarkerOptions(base_options = BaseOptions(model_path), running_mode = VisionRunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.5)

landmarker = HandLandMarker.create_from_options(options)
hand_landmarker_result = landmarker.detect(frame_RGB)

"""


# Current but Old version of Mediapipe API's to detect hands and visualize them

mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands

handsLandmarker = mpHands.Hands(static_image_mode = True, min_detection_confidence=0.3, max_num_hands=2)

DATA_DIR = './data'
classFolders = os.listdir(DATA_DIR)

data = []  # this list will containg a list of landmarks (x, y) coordinates for every img of all classes.  
            # [[img_1 of class_1], [img_2 of class_1],...., [img_1 of class_2], [img_2 of class_2],......, [img_1 of class_3], [img_2 of class_3], .....]
labels = [] # this list will contain label for every img for all classes. [class_1, class_1, ....., class_2, class_2, ....., class_3, class_3]

for classFolder in classFolders:
    for imgName in os.listdir(os.path.join(DATA_DIR, classFolder)):
        
        imgPath = os.path.join(DATA_DIR, classFolder, imgName)
        imgBGR = cv2.imread(imgPath)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        handsLandmarkerResults = handsLandmarker.process(imgRGB)

        if handsLandmarkerResults.multi_handedness:
            for handLandmarkerResult in handsLandmarkerResults.multi_hand_landmarks:
                imgLandmarkData = []
                for landmarkIndex in range(len(handLandmarkerResult.landmark)):
                    x = handLandmarkerResult.landmark[landmarkIndex].x
                    y = handLandmarkerResult.landmark[landmarkIndex].y

                    imgLandmarkData.append(x)
                    imgLandmarkData.append(y)
                data.append(imgLandmarkData)
                labels.append(classFolder)


print(len(data), len(labels))

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels}, f)
f.close()


