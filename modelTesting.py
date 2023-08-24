import cv2
import mediapipe as mp
import pickle
import numpy as np

# Initializing the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# initializing the mediapipe API's for drawing landmarks and for preparing the hand landmark info
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles
mpHands = mp.solutions.hands

handsLandmarker = mpHands.Hands(static_image_mode = True, min_detection_confidence=0.3, max_num_hands=2) # we initialized the hand landmark detecter to detect max of 2 hands

cap = cv2.VideoCapture(0)



while True:
    dataToBePredicted = [] # this list will store 2 lists for 2 hand landmarks  [[x1, y1, x2, y2, ....x21, y21], [x1, y1, x2, y2, ....x21, y21]]. If 1 hand is detected then 1 list is stored
    bboxData = [] # this list will contain [[min(x1 coords), min(y1 coords), max(x1 coords), max(y1 coords)], [min(x1 coords), min(y1 coords), max(x1 coords), max(y1 coords)]]
    success, frame = cap.read()
    H, W, _ = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    handsLandmarkerResults = handsLandmarker.process(frameRGB)

    if handsLandmarkerResults.multi_hand_landmarks:    # here we check if any hand is detected. if detected then we go inside if statement
        for handLandmarkerResult in handsLandmarkerResults.multi_hand_landmarks:  # this loop is to draw the visualization of all landmarks detected
            mpDrawing.draw_landmarks(
                    frame, # image to draw
                    handLandmarkerResult, # model output
                    mpHands.HAND_CONNECTIONS, # hand connections
                    mpDrawingStyles.get_default_hand_landmarks_style(),
                    mpDrawingStyles.get_default_hand_connections_style())
            
        for handLandmarkerResult in handsLandmarkerResults.multi_hand_landmarks: # if only 1 hand is detected then this loop execute for 1 time and if 2 detected then 2 times
            imgLandmarkData = []  # we initialized this to store x, y coords of each landmark detected for single hand. 
            x_ = [] # we initiated this list to store all x coords of each landmark of single hand. so that we can get min, max values and append them to bboxData
            y_ = []  # we initiated this list to store all y coords of each landmark of single hand. so that we can get min, max values and append them to bboxData
            for landmarkIndex in range(len(handLandmarkerResult.landmark)): # for 1st hand detected, we take all x, y coords of all 21 landmarks 
                x = handLandmarkerResult.landmark[landmarkIndex].x
                y = handLandmarkerResult.landmark[landmarkIndex].y
                imgLandmarkData.append(x) # we store x, y coords of all 21 landmark points to a list. this list is used for prediction using trained model
                imgLandmarkData.append(y)
                x_.append(x) 
                y_.append(y)
            dataToBePredicted.append(imgLandmarkData)
            bboxData.append([min(x_), min(y_), max(x_), max(y_)])
                
        if len(dataToBePredicted) > 1: # here if dataToBePredicted list length == 2 then we go in and draw 2 bounding boxes, predicted classes for 2 hands that are detected
        
            x1A = int(bboxData[0][0] * W) - 15
            y1A = int(bboxData[0][1] * H) - 15

            x2A = int(bboxData[0][2] * W) - 15
            y2A = int(bboxData[0][3] * H) - 15

            x1B = int(bboxData[1][0] * W) - 15
            y1B = int(bboxData[1][1] * H) - 15

            x2B = int(bboxData[1][2] * W) - 15
            y2B = int(bboxData[1][3] * H) - 15

            predictionA = model.predict([np.array(dataToBePredicted[0])])
            predictionAClass = predictionA[0]

            predictionB = model.predict([np.array(dataToBePredicted[1])])
            predictionBClass = predictionB[0]

            cv2.rectangle(frame, (x1A, y1A), (x2A, y2A), (0, 0, 0), 1)
            cv2.putText(frame, predictionAClass, (x1A, y1A - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (x1B, y1B), (x2B, y2B), (0, 0, 0), 1)
            cv2.putText(frame, predictionBClass, (x1B, y1B - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        if len(dataToBePredicted) == 1: # here if dataToBePredicted list length == 1 then we go in and draw 1 bounding boxes, predicted class for 1 hand1 that is detected
            x1A = int(bboxData[0][0] * W) - 8
            y1A = int(bboxData[0][1] * H) - 8

            x2A = int(bboxData[0][2] * W) + 8
            y2A = int(bboxData[0][3] * H) + 20

            predictionA = model.predict([np.array(dataToBePredicted[0])])
            predictionAClass = predictionA[0]

            cv2.rectangle(frame, (x1A, y1A), (x2A, y2A), (0, 0, 0), 1)
            cv2.putText(frame, predictionAClass, (x1A, y1A - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(10)
    
    if k==27:   # we break the prediction process if user hits Esc button
        break

cap.release()
cv2.destroyAllWindows()

