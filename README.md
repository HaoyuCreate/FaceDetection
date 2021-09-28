# Face Detection
This project, based on Python, is trying to detect human faces with classical 68-landmarks detection and head pose estimation.

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
Open demo.ipynb to test its performance. 


