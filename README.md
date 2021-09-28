Face Detection
This project is trying to detect human faces with classical 68-landmarks detection and head pose estimation.


$ pip install -r /path/to/requirements.txt

To detect faces within a image, try to run the command
$ python FaceDetection.py 0 --input_image_file=/path/to/imageA.jpg


To detect faces within a video, try to run the command
$ python FaceDetection.py 1 --input_video_file=path/to/videoA.mp4


To run the detection algorithm with an web-camera
Use jupyter notebook under a python enviroment with all required libs
$ jupyter notebook
Open demo.ipynb to test its performance. 


