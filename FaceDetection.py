# -*- coding: utf-8 -*- #
# Coded by Haoyu Fang
# Contact: haoyu.fang@nyu.edu

import cv2
import argparse
import os
import os.path as ops
import dlib
import mediapipe as mp
import numpy as np
import math
import sys
import copy
from utils import CalOcpyofROI,ROIGenerator,CalDistance,CameraMatrix,Ref3DModel,Ref2dImagePoints


def CheckinROI(ROI_bbx,face,area_rate=0.8):
    face_bbx = [face.bottom(),face.right(),face.top(),face.left()]
    return (CalOcpyofROI(face_bbx,ROI_bbx) >= 0.1)


def CalDistance(img,face):
    # calculate the distance between the detected face and image center 
    if len(img.shape) == 1:
        height, weight = img.shape
    else:
        height,weight,_ = img.shape
    
    img_center =  np.array([int(height/2), int(weight/2)])
    face_center = np.array([int((face.left()+face.right())/2),int((face.top()+face.bottom())/2)])
    return math.sqrt(sum((face_center - img_center)**2))


def HeadPoseDetector(img,landmarks):
    height,width,_ = img.shape
    focal_length = 1 * width
    
    camera_matrix = CameraMatrix(focal_length,(height / 2, width / 2))
    face_3d_model = Ref3DModel()
    ref_img_pts = Ref2dImagePoints(landmarks)
    mdists = np.zeros((4, 1), dtype=np.float64) # Assuming no lens
    
    # calculate rotation and translation vector using solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(face_3d_model,
                                        ref_img_pts,
                                        camera_matrix,
                                        mdists)
    
    # calculate nose start and nose end 
    nose_end_point, _ = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]),
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        mdists)
     
    p1 = ( int(ref_img_pts[0][0]), int(ref_img_pts[0][1])) # start
    p2 = ( int(nose_end_point[0][0][0]), int(nose_end_point[0][0][1])) # end
    
    return p1,p2 #theta and phi


###  Face Detection ###
def FaceDetection_OneFrame(img,detector,predictor,Colors,args):
    color_flag = []
    distance_list = []

    origin_img = copy.copy(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)

    # generate ROI, ROI_ares = area_rate * image_area
    roi = ROIGenerator(img,area_rate=args.roi_rate)
    roi_bottom,roi_right,roi_top,roi_left = roi
    cv2.rectangle(img,(roi_left,roi_top),(roi_right,roi_bottom),(0,255,255),1)

    for face in faces:
        color_flag.append(Colors[1])
        distance_list.append(CalDistance(img,face))
    if distance_list:
        color_flag[distance_list.index(min(distance_list))] = Colors[0] # find the first nearest face

    for face,color in zip(faces,color_flag) :
        # Check if the face in ROI 
        if CheckinROI(roi,face): break
        
        # Detect the landmarks
        landmarks = predictor(gray,face)
        
        # Estimate head pose
        pose_start,pose_end = HeadPoseDetector(img,landmarks)
        
            
        # drawing the face detection
        cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),color,3)

         # drawing the landmarks 
        for n in range (68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            cv2.circle(img,(x,y),2,color,-1)
              
        # drawing the head pose 
        cv2.line(img, pose_start, pose_end, (255,0,0), 2)
        
    if not args.is_video:
        cv2.imshow('Original Image',origin_img)
        cv2.imshow('Face Detection Result',img)
        output_file_name = ops.join(args.output_dir,\
            (args.input_image_file.split('/')[-1]).split('.')[0]+'_detected.jpg')
        cv2.imwrite(output_file_name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return img

def FaceDetection_OnVideo(cap,detector,predictor,Colors,args):
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(ops.join(
        args.output_dir,(args.input_video_file.split('/')[-1]).split('.')[0]+'_detected.avi'),\
        fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while cap.isOpened():
        _,frame =cap.read()
        detected_frame = FaceDetection_OneFrame(frame,detector,predictor,Colors,args)

        writer.write(detected_frame)

        cv2.imshow('Face Detection',detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def Initial_arg():
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('is_video',type=int,default=0,\
                help='0 or 1: is the algorithm process an video or a image')
    parser.add_argument('-im','--input_image_file',type=str,default='./test/glasses.jpg')
    parser.add_argument('-iv','--input_video_file',type=str,default='./test/head-pose-face-detection-female-and-male.mp4')
    parser.add_argument('-o','--output_dir',type=str,default='./output')
    parser.add_argument('-l','--landmark_data',type=str,default='landmarks/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('-r','--roi_rate',type=int,default=0.8)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = Initial_arg()
    if not ops.exists(args.output_dir):
        os.mkdir(args.output_dir)

    detector =dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(args.landmark_data)
    Colors = [(0,255,0),(0,0,255)] #0:green = focused; 1:red = unfocused
    
    if args.is_video:
        print('Currently processing ',args.input_video_file)
        cap = cv2.VideoCapture(args.input_video_file)
        if not cap.isOpened():
            raise ValueError('Video open failed.')
        FaceDetection_OnVideo(cap,detector,predictor,Colors,args)
    else:
        print('Currently processing ',args.input_image_file)
        img = cv2.imread(args.input_image_file)
        if img is None:
            raise ValueError('Image open failed.')
        FaceDetection_OneFrame(img,detector,predictor,Colors,args)
