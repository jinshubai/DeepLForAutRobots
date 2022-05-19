import cv2
import torch

import numpy as np
from PIL import Image
import io

import html
import time
import matplotlib.pyplot as plt
#!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#!pip install --upgrade openpifpaf==0.11.2
import openpifpaf
#!pip install git+https://github.com/openpifpaf/openpifpaf
import PIL
import requests
import cv2 as cv

#!git clone https://github.com/ultralytics/yolov5
#!pip install -r C:/Users/tobia/Documents/EPFL/DeepLForAuto/project/yolov5/requirements.txt

#device = torch.device('cpu')
device = torch.device('cuda')  # if cuda is available

import logging


print(openpifpaf.__version__)
print(torch.__version__)

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('C:\\Users\\tobia\\Documents\\EPFL\\DeepLForAuto\\project\\Yolov5_DeepSort_OSNet')
sys.path.append('C:\\Users\\tobia\\Documents\\EPFL\\DeepLForAuto\\project\\Yolov5_DeepSort_OSNet\\yolov5')

from loomo.detector import Detector


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.cuda.is_available()


# start streaming video from webcam
camera=cv2. VideoCapture(0) 


dector= Detector()
while True:
    result, frame = camera.read()
    if not result:
        break
    boxes,labels,image=dector.forward(frame)
    print(boxes,labels)
    cv2.imshow("Annotated frame",image)

   
    
    #exit command
    
    c = cv2.waitKey(1) #escape key
    if c == 27:
        break        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
      #cv2_imshow(imToShow[:,:,0:3])
      #doDetectSceleton=False
camera.release()
cv2.destroyAllWindows() 