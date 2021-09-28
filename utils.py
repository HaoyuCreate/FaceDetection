import cv2
import dlib
import mediapipe as mp
import numpy as np
import math
import sys


# Video input useful functions (videp API)

# Image input useful functions (image API)


# Three usful funitons for head-pose estimation
def Ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def Ref2dImagePoints(landmarks):
    imagePoints = [[landmarks.part(30).x, landmarks.part(30).y],  # Nose tip 
                   [landmarks.part(8).x,  landmarks.part(8).y ],  # Chin 
                   [landmarks.part(36).x, landmarks.part(36).y],  # Left eye left corner 
                   [landmarks.part(45).x, landmarks.part(45).y],  # Right eye right corne 
                   [landmarks.part(48).x, landmarks.part(48).y],  # Left Mouth corner        
                   [landmarks.part(54).x, landmarks.part(54).y]]  # Right mouth corner       
    return np.array(imagePoints, dtype=np.float64)


def CameraMatrix(fl, center):
    cameraMatrix = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(cameraMatrix, dtype=np.float)



# Two usful funitons for ROI calculation
def ROIGenerator(img,area_rate=0.8):
    if len(img.shape) == 1:
        height, weight = img.shape
    else:
        height,weight,_ = img.shape
    
    ROI_rate = math.sqrt(area_rate)
    roi_right = weight - int((weight-ROI_rate*weight)/2) 
    roi_bottom = height - int((height-ROI_rate*height)/2) 
    roi_left = int((weight-ROI_rate*weight)/2) 
    roi_top = int((height-ROI_rate*height)/2) 
    
    ROI_bbx = [roi_bottom,roi_right,roi_top,roi_left]
    return ROI_bbx

def CalOcpyofROI(boxA, boxB): # boxA: face; boxB: ROI
    # Per requst of the assignment, the ratio is similar to IoU
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of ROI
    boxBBrea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over area of ROI
    occupy = interArea / float(boxBBrea)

    # return the intersection over union value
    return occupy

# Check the deteced face is under focus
def CalDistance(img,face):
    # calculate the distance between the detected face and image center 
    if len(img.shape) == 1:
        height, weight = img.shape
    else:
        height,weight,_ = img.shape
    
    img_center =  np.array([int(height/2), int(weight/2)])
    face_center = np.array([int((face.left()+face.right())/2),int((face.top()+face.bottom())/2)])
    return math.sqrt(sum((face_center - img_center)**2))