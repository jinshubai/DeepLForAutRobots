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


#git clone https://github.com/ultralytics/yolov5
#!pip install -r C:/Users/tobia/Documents/EPFL/DeepLForAuto/project/yolov5/requirements.txt

#device = torch.device('cpu')
device = torch.device('cuda')  # if cuda is available

import logging


print(openpifpaf.__version__)
print(torch.__version__)

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('/home/group12/DeepLForAutRobots')
sys.path.append("/home/group12/DeepLForAutRobots/yolov5/")
sys.path.append("/home/group12/DeepLForAutRobots/deep_sort/")
sys.path.append("/home/group12/DeepLForAutRobots/yolov5/utils")

print(sys.path)
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.plots import Annotator,colors, save_one_box
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
     
class Detector():
    """docstring for Detector"""
    def __init__(self):
        


        self.net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w')
        self.net = self.net_cpu.to(device)

        openpifpaf.decoder.CifSeeds.threshold = 0.5
        openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
        openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
        self.processor = openpifpaf.decoder.factory_decode(self.net.head_nets, basenet_stride=self.net.base_net.stride)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.model.classes=[0]

        cfg = get_config()
        cfg.merge_from_file("/home/group12/DeepLForAutRobots/deep_sort/configs/deep_sort.yaml")
        deep_sort_model = 'osnet_x1_0_MSMT17'

        self.deepsort = DeepSort(
            deep_sort_model,
            device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)

        self.verbose = False

        self.doDetectSceleton=True
        self.doTracking = False


        # initialze bounding box to empty
        self.bbox = ''
        self.count = 0 
        self.locationOfPersonToTrack=(None,None)
        self.bboxToTrack = []
        self.IdToTrack = -1
        self.lostIdCount = 0
        self.MaxIdCount = 50
        self.begin = True


    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def forward(self, img): 
        img=np.array(img)
        frame=img  
        trackingBox=[]
        trackingLabel=[]
        timeY = time.time()
        #call YOLOv5
        results = self.model(frame,augment=False)
        #print(results)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        #print(labels,cord)
        
        xloc,yloc=None,None
        #if we want to do first persone detection -> use openpifpaf until it discovers the person of interest
        timebp = time.time()
        if(self.doDetectSceleton):
            self.doTracking = False
            img = PIL.Image.fromarray(frame, 'RGB') #convert img to PIL
        #arrange data for OpenPifPaf prediction
            data = openpifpaf.datasets.PilImageList([img])
            loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True,collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        #keypoint_painter = openpifpaf.show.KeypointPainter(linewidth=6)

        #predict
            for images_batch, _, __ in loader:
                predictions = self.processor.batch(self.net, images_batch, device=device)[0]

        #get people data from img
            people = process_people(predictions)
        #label img for vizualisation
            frame,(xloc,yloc),hasDetected,personDetected = return_skeletons(frame,people)
            timeE = time.time()
            if self.verbose: print("infTime:",timeE-timebp)
        #cv2_imshow(labeled_img)
            
            
        if self.doTracking and cord is not None and len(cord): #avoid empty detections
            annotator = Annotator(frame, line_width=2, pil=not ascii)
            labels = torch.Tensor(labels)
            cord = np.array(cord)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            #xywhs = torch.Tensor(cord[:,:4]) #see below for one guy: int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape), row[4]*
            xyxys = []
            for i in range(len(labels)):
                xyxys.append([cord[i,0]*x_shape,cord[i,1]*y_shape,cord[i,2]*x_shape,cord[i,3]*y_shape])
            xyxys = np.atleast_2d(xyxys)
            #print('yolo xy',xyxys)
            #Transform fomr xyxy (top left, bottom right) to xywh where xy is the center and wh is the width and height (full width and height)
            xywhs = torch.Tensor(xyxy2xywh(xyxys)) 
            #print('yolo ws :',xywhs)
            confs = torch.Tensor(cord[:,4])
            #print(frame.shape)
            #print(np.reshape(frame,[3,480,640]))
            #print(xywhs,confs,labels,)
            outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), labels.cpu(), frame)
            # draw boxes for visualization
            #print('DeepSort',(outputs))
            IDs = []
            if len(outputs) > 0:
                for output in outputs:
                    self.bboxes = output[0:4] #given in xyxy
                    #print('Tracker BB',bboxes)
                    id = output[4]
                    IDs.append(id)
                    cls = output[5]
                    conf = output[6]
                    #just forinitialisation (first loop): link OpenPifPaf to ID to track
                    if self.begin and IsPersonOfInterest(self.bboxes,self.locationOfPersonToTrack): 
                        self.IdToTrack = id
                        c = 0 #mark red
                        self.begin = False
                    elif not self.begin and id == self.IdToTrack: #identify the id of interest
                        c = 0

                        trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                        trackingLabel=output[4]
                    else: 
                        c = 1 #blue I think
                    text = f'{id:0.0f} person {conf:.2f}'
                    #print(bboxes,id,conf)
                    #display_xywh(frame,bboxes,text)
                    annotator.box_label(self.bboxes, text, color=colors(c, True))
            
            #Id still here ?
            if self.IdToTrack not in IDs:
                self.lostIdCount += 1
            else: self.lostIdCount = 0 #id is here re initialize counter
            
            #wait clk*MaxIdCount time before saying we lost the Id: TO OPTIMIZE
            if self.lostIdCount > self.MaxIdCount:
                print('WARNING: Lost person of Interest ! Do gesture of interest again')
                self.doDetectSceleton = True
                #doTracking = False #it is finally done in the beginning of the skeleton loop -> avoid entering in the YOLo below statement
            if self.verbose: print('tracking time :', time.time()-timeY)
            
            
        #loop through YOLO detections and draw them on transparent overlay image
        if not self.doTracking:
            print("")
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            #print(x_shape)
            for i in range(n):
                row = cord[i]
                if row[4] >= 0.2:     #if confidence is above 0.2
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    colorToUse=bgr

                    isSamePerson = False
                    #avoid having personDetected empty
                    box = (x1,y1,x2,y2)
                    if hasDetected and self.doDetectSceleton: isSamePerson = isSamePersonDetector(personDetected,box) 
                    if(hasDetected and isSamePerson):
                    #locationOfPersonToTrack=((x1+x2)/2.0,(y1+y2)/2.0)
                        self.locationOfPersonToTrack=(xloc,yloc) #need to change this
                        print("Detected at(x,y):(",self.locationOfPersonToTrack[0],self.locationOfPersonToTrack[1],")")
                        colorToUse=(255,0,0)
                        self.doDetectSceleton = False
                        self.doTracking = True
                        self.begin = True
                        self.bboxToTrack = box #not used
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colorToUse, 2)
                    cv2.putText(frame, results.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorToUse, 2)
            if self.verbose: print('full time: ',time.time()-timeY)

        #plot a screen of the current moment
        #if(xloc!=None and yloc!=None):
        #    imToShow=np.zeros_like(bbox_array)
        #    imToShow[:,:,0:3]=frame
        #    imToShow[bbox_array!=0]=bbox_array[bbox_array!=0]
        #cv2.imshow("image",frame)

        
        if(len(trackingBox)==0):
           return[float("NAN"),float("NAN")],[float("NAN")]
        else:
          print([trackingBox[0][0],trackingBox[0][1]],[float(trackingLabel)])
          return [trackingBox[0][0],trackingBox[0][1]],[float(trackingLabel)]







def localize(people):
    shoulders = ['right_shoulder','left_shoulder']
    elbows = ['right_elbow','left_elbow'] 
    #a way to localize a specific gesture: elbow above shoulder for the moment
    #Edit to detect severak rising hand ? idk will see
    loc = False
    location = (0,0)
    personDetected = []
    #go through all people
    for p in people:
        right_elb = False 
        left_elb = False
        right_rise = False
        left_rise = False
        #elbow occlusion
        if p[elbows[0]][2] != 0: right_elb = True
        if p[elbows[1]][2] != 0: left_elb = True
        #rised hand
        if p[shoulders[0]][1]>p[elbows[0]][1]: right_rise = True
        if p[shoulders[1]][1]>p[elbows[1]][1]: left_rise = True
        #if see two shoulders:
        if p[shoulders[0]][2] > 0 and p[shoulders[1]][2] > 0:
            if (right_elb and right_rise) or (left_elb and left_rise):
                x1,y1 = p[shoulders[0]][0],p[shoulders[0]][1]
                x2,y2 = p[shoulders[1]][0],p[shoulders[1]][1]
                loc = True
                location = ((x1+x2)/2,(y1+y2)/2)
                personDetected = p
                break
        #case if an elbow is occluded
        elif right_rise and (right_elb and not left_elb):
                loc = True
                location = p[shoulders[0]][0],p[shoulders[0]][1]
                personDetected = p
                break
        #case if an elbow is occluded
        elif left_rise and (not right_elb and left_elb):
                loc = True
                location = p[shoulders[1]][0],p[shoulders[1]][1]
                personDetected = p
                break
    return loc, location,personDetected
def process_people(pred):
    #get data
    people = []
    for p in pred:
        body = {}
        for i,name in enumerate(p.keypoints):
            body[name] = (p.data[i][0],p.data[i][1],p.data[i][2])
        people.append(body)
    return people

#as it crashed, do custom img labelling
def return_skeletons(img,people):
    Customcolors = [(203, 192, 255),(5,20,90),(100,0,100),(0,255,0),(255,0,0),(60,40,232),(200,14,100)]
    I = np.copy(img)
    #go through every persons joint and add them to the labeled img as dots
    #one color per person
    for i,p in enumerate(people):
        for joint in p:
            unpack = p[joint]
            x,y,score = int(unpack[0]),int(unpack[1]),unpack[2]
            I=cv2.rectangle(I, (x-1, y-1), (x+1, y+1), Customcolors[i], 2)
    
    #check for specific gesture
    loc, location,personDetected = localize(people)
    (x,y)=(None,None)
    if loc:
        x,y=int(location[0]),int(location[1])
        #label specific gesture
        I=cv2.rectangle(I,(x-1, y-1), (x+1, y+1),(0,0,255),2) #red square on cible
    return I,(x,y),loc,personDetected


jointsOfInterest = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']


def isSamePersonDetector(personDetected,box):
    tolX = 45 #30 pixels location tolerance ??
    tolY = 35
    version = 3
  #locYolo is the center of the bounding box
  #we need to compare this to the keypoints detected
  #This task is quite difficult...grr
    isSamePerson = False

    (x1,y1,x2,y2) = box
    locYolo = [(x1+x2)/2.0,(y1+y2)/2.0]
#loop over keypoints
    OPENLOC = []
    for joint in personDetected:
        unpack = personDetected[joint]
    #if joint in jointsOfInterest: actually no
        x,y,score = int(unpack[0]),int(unpack[1]),unpack[2]
        if score > 0.4:
            OPENLOC.append([x,y])
    OPENLOC = np.array(OPENLOC)
  #compute openpifpaf middle
  
    if version == 1:
        Xmin = np.min(OPENLOC[:,0])
        Xmax = np.max(OPENLOC[:,0])
        Ymin = np.min(OPENLOC[:,1])
        Ymax = np.max(OPENLOC[:,1])
        Openloc = [(Xmin+Xmax)/2.,(Ymin+Ymax)/2.]

        # #print(Openloc)
        errX = np.abs(Openloc[0]-locYolo[0])
        errY = np.abs(Openloc[1]-locYolo[1])
        print(errX,errY)
        if errX < tolX and errY < tolY:
            isSamePerson = True
    elif version == 2:
    #build shape of furthest joints
    #then compute its centroid
    #I think it could still fail...
        pass
    else:
        Openloc = np.mean(OPENLOC,axis = 0)
        if Openloc[0] > x1 and Openloc[0]<x2 and Openloc[1] > y1 and Openloc[1]<y2:
            isSamePerson = True

    return isSamePerson

def display_xywh(frame,bboxes,text):
    x, y, w, h = bboxes[0],bboxes[1],bboxes[2],bboxes[3]
    colorToUse = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), colorToUse, 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorToUse, 2)

def IsPersonOfInterest(bbox,locationOfPersonToTrack):
    x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
    xt, yt = locationOfPersonToTrack[0],locationOfPersonToTrack[1]
    if x1<xt and xt < x2 and y1 < yt and yt < y2:
        return True
    return False
    
    
    
if __name__ == "__main__":
    image=cv2.imread("/home/group12/DeepLForAutRobots/testImage.jpg")
    detect=Detector()
    print(detect.forward(image))