# Face Detection
This project, based on Python, is trying to detect human faces with classical 68-landmarks detection and head pose estimation.

### Aditional Questions:
- If the head goes out of focus and comes back in, will your algorithm start tracking again?
  > Yes, the algorithm would not stop when users go out of the view and it continues to work when users come back.
- How are you prioritizing tracking for objects in focus?
  > I calculated distance bettween the focus area (assumed as image center) and all detected faces. Since faces may move in the video, so the algorithm alter the object in focus and keep tracking the object that get nearest to the facus area. 
- Are you able to track with foreign objects like headphones, spectacles, etc?
  > Currently, the learned feature that I applied to the aglorithm is based on [shape_predictor_68_face_landmarks.dat](test/shape_predictor_68_face_landmarks.dat). It doesn't contain features of other objects. However, the predictor is actually able to use other learned features of foreign objects which I can have a further exploitation. 
- If there is occlusion (example: if someone is drinking from a bottle) will the tracking stop?
  > No, the algorithm does not stop when it can not detect human face in current frame. It might fail to find the faces temporarily and give no predition but will continue to track the faces whenever it can again. 
- Anything else that you think will be important for a real-world application but might not have been mentioned above. Explain why it is important.
  > Efficiency. In this project, it is efficiency that is of importancein real-world applications from my personal view. Before developing current algorithm, I also tried another DNN-based SSD-structure algorithm to detect human faces (***code in the develop branch under this repo***). Although it has a relative high accuracy on human face detection, it is a little bit slow in simultaneous tracking when using high-resolution web-camera. 
  
  > Conviniency. This project combines three main tasks together: 1) human face datetion; 2) landmarks detection; 3) head pose estimation. There are learned-based algorithms to couple with each of the tasks but is few to solve the combined mission. It is feasible to use three independed DNN to sequencially solve the three tasks, but this solution requires huge computation resources. Another solution is using a single DNN which are trained on a combined dataset in an end-to-end manner. But this solution needs to construct a dataset which has ground truth for all tasks.

### Results
Detection results on images:
Face_detected in the crowed
<img src="https://github.com/HaoyuCreate/FaceDetection/tree/main/output/crowd_detected.jpg" width="25%">

Face_detected with occlusion
<img src="https://github.com/HaoyuCreate/FaceDetection/tree/main/output/occlusion_detected.jpg" width="25%">


Detection results on videos:
<video src="https://github.com/HaoyuCreate/FaceDetection/tree/main/output/head-pose-face-detection-female-and-male_detected.avi" width="50%" height="50%">

### Used Python packages
+ opencv-python
+ dlib
+ mediapipe
+ numpy
+ jupyter
Users can install this 
```
$ pip install -r /path/to/requirements.txt
```

#### Face detection on images
To detect faces within a image, try to run the command
```
$ python FaceDetection.py 0 --input_image_file=/path/to/imageA.jpg
```

#### Face detection on video
To detect faces within a video, try to run the command
```
$ python FaceDetection.py 1 --input_video_file=path/to/videoA.mp4
```

#### Face detection with web-camera
To run the detection algorithm with an web-camera
Use jupyter notebook under a python enviroment with all required libs
```
$ jupyter notebook
```
Open ***demo_webcamera.ipynb*** to test its performance. 


