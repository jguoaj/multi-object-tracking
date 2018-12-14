## Multi-Object-Tracking

### General Idea
A multi-object-tracking algorithm uses YOLO v3, deep_sort and optical flow based on Kanade–Lucas–Tomasi (KLT). 

### Methodology
1. YOLO v3 detection
2. deep_sort tracker update
3. optical flow tracker update

### Dependences
The code has been tested in python 3.5, ubuntu 16.04. 
1. tensorflow
2. keras
3. numpy
4. sklearn
5. scipy
6. scikit-image
7. opencv

### How to run
1. Download yolov3 model from [YOLO website](http://pjreddie.com/darknet/yolo/). Convert this model to a Keras model. For this project, we train a new yolov3 model and use Keras.save_model. 
2. Run script: python3.5 tracking.py


### Results

1. test result video 1: https://youtu.be/SKX-EcQnens
2. test result video 2: https://youtu.be/56RKbOaInYI

### Reference work
1. keras YOLO v3: https://github.com/qqwweee/keras-yolo3
2. deep_sort: https://github.com/nwojke/deep_sort
3. YOLO v3 deep_sort integration: https://github.com/Qidian213/deep_sort_yolov3
4. optical flow: https://github.com/ZheyuanXie/OpticalFlow
