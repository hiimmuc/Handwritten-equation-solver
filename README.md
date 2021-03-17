### Handwritten-equation-solver

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow+v2.4.1-yellow.svg)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/phgnam-dang/)


## :innocent: Motivation
In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. 


## What I did??
- Training 2 yolov4 models to detect equations and characters in pictures
- Training cnn model to classify those characters
- Put all of it together to and using some libraries to some and format the result as HTML form
- Using Flask to create web app

## :star: Features
Our face mask detector didn't use any morphed masked images dataset. The model is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

This system can detect and classify three types: masked, unmasked and incorrectly masked


## Preparation

## 1. Download weight and darknet
- Download weight file at https://drive.google.com/file/d/15NGlxNoybiPfOjey9ObV6O67HcaOgMlN/view?usp=sharing
- Download darknet at https://github.com/AlexeyAB/darknet.git
## 2. Config
- folow this tutorial https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/
- In file yolov4_training.cfg :
  - Change classes = 80 to classes = 1 at lines 970 1058 1146
  - change filters = 255 to filters = 18 at line 1139 1051 963
## 3. Run this program
- repository : !git clone https://github.com/khangdx1998/Equations-solver.git
- Download weight and unzip it to the same directory
- run by command:
```
$ python3 webapp.py
```
- get the https link and run it on browser
## 4. Results
![](result/1.png)

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: hiimmuc1811@gmail.com
or my facebbok:
https://www.facebook.com/phgnam1811/

## Contributors:
- Đặng Phương Nam - https://github.com/hiimmuc?tab=repositories
- Doãn Xuân Khang - https://github.com/khangdx1998
- Nguyễn Đức Thắng - https://www.facebook.com/ducthangbka8


