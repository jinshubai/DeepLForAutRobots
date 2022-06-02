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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

        self.capDetector = torch.hub.load('ultralytics/yolov5','custom',path=r'/home/group12/DeepLForAutRobots/best (17) (1).pt') #load custom detector
        self.capDetector.classes=[0]

        cfg = get_config()
        cfg.merge_from_file("/home/group12/DeepLForAutRobots/deep_sort/configs/deep_sort.yaml")
        deep_sort_model = 'osnet_x1_0_MSMT17'

        self.deepsort = DeepSort(
            deep_sort_model,
            device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET)
        print("Finished init2")
        self.verbose = False

        self.doDetectSceleton=True
        self.doTracking = False
        self.isCap = False
        self.capLocation = [None,None]
        #self.findCapAgain = False
        self.doCapDetection=True

        # initialze bounding box to empty
        self.bbox = ''
        self.count = 0 
        self.locationOfPersonToTrack=(None,None)
        self.bboxToTrack = []
        self.IdToTrack = -1
        self.lostIdCount = 0
        self.MaxIdCount = 5
        self.begin = True
                ################################################################################33NATHAN
        self.Xtrack = float("NAN")
        self.Ytrack = float("NAN")
        self.LabelTrack = float("NAN")


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
        print("Forward")
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
        
        self.isCap = False
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        if self.doCapDetection:
            Capresults = self.capDetector(frame,augment=False)
            Caplabels, Capcord = Capresults.xyxyn[0][:, -1].cpu().numpy(), Capresults.xyxyn[0][:, :-1].cpu().numpy()
            nCap = len(Caplabels)
            #print(Capcord)
            if nCap > 1: print("WARNING: several cap detected")
            elif nCap == 0 and self.verbose: print("No cap detected")
            elif nCap == 1: 
                self.isCap = True
                cap = Capcord[0]
                x1, y1, x2, y2 = int(cap[0]*x_shape), int(cap[1]*y_shape), int(cap[2]*x_shape), int(cap[3]*y_shape)
                self.capLocation = [(x1+x2)/2,(y1+y2)/2]
            #plot anyway the results
            
            for i in range(nCap):
                cap = Capcord[i]
                if cap[4] >= 0.25:     #if confidence is above 0.2
                    x1, y1, x2, y2 = int(cap[0]*x_shape), int(cap[1]*y_shape), int(cap[2]*x_shape), int(cap[3]*y_shape)
                    colorToUse = (255, 0, 0)
                    conf = cap[4]
                    box = (x1,y1,x2,y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colorToUse, 2)
                    cv2.putText(frame, Capresults.names[int(Caplabels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorToUse, 2)
                    cv2.putText(frame, str(conf), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorToUse, 2) 
                    #print(path_to_img+'cap.jpg')
                    #cv2.imwrite(path_to_img+'\cap.jpg', frame)
            



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
                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), labels.cpu(), frame)
                # draw boxes for visualization
                #print('DeepSort',len(outputs))
                IDs = []
                if len(outputs) > 0:
                    for output in outputs:
                        bboxes = output[0:4] #given in xyxy
                        #print('Tracker BB',bboxes)
                        id = output[4]
                        IDs.append(id)
                        cls = output[5]
                        conf = output[6]
                        #just forinitialisation (first loop): link OpenPifPaf to ID to track
                        if self.begin and IsPersonOfInterest(bboxes,self.locationOfPersonToTrack): #just initialize tracker without taking care of the cap first
                            self.IdToTrack = id
                            c = 0 #mark red
                            self.begin = False
                            #Xtrack,Ytrack=output[0],output[1]
                            #LabelTrack = output[4]
                            #trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                            #trackingLabel=output[4]
                        elif not self.begin and id == self.IdToTrack and not self.isCap: #identify the id of interest when cap is not detected
                            c = 0 #red
                            if self.verbose: print("no cap")
                            #Xtrack,Ytrack=output[0],output[1]
                            #LabelTrack = output[4]
                            #trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                            #trackingLabel=output[4]
                        elif not self.begin and id == self.IdToTrack and self.isCap: #if the cap has been detected, check if the tracked person has the cap
                            if IsPersonOfInterest(bboxes,self.capLocation): #check if center of bbox of cap is in the bb of person tracked
                                c = 0
                                if self.verbose: print("cap confirmed")
                                #Xtrack,Ytrack=output[0],output[1]
                                #LabelTrack = output[4]
                                #trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                                #trackingLabel=output[4]
                            else: #that should mean the person of interest has changed to another person
                                #try to find back the person of interest if still here
                                print("cap not confirmed")
                                for pp in outputs:
                                    bboxesPP = pp[0:4]
                                    IdPP = pp[4]
                                    if IsPersonOfInterest(bboxesPP,self.capLocation): 
                                        #it means the cap is in another box ! just take it as the new id to track
                                        print("missmatch")
                                        self.IdToTrack = IdPP
                                        c = 7 
                                        break
                                c = 2 #orange
                                #trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                                #trackingLabel=output[4]
                        elif not self.begin and id != self.IdToTrack and self.isCap: #if cap is in another box
                            #check if cap is in another id
                            if IsPersonOfInterest(bboxes,self.capLocation): 
                                self.IdToTrack = id
                                c = 5 #green
                                print("cap re ids")
                            c = 1
                            #trackingBox=xyxy2xywh(np.expand_dims(output[0:4],axis=0))
                            #trackingLabel=output[4]
                        else: 
                            c = 1 #pink 
                            #print('pink')
                        text = f'{id:0.0f} person {conf:.2f}'
                        #print(bboxes,id,conf)
                        #display_xywh(frame,bboxes,text)
                        #annotator.box_label(bboxes, text, color=colors(c, True))
                        #cv2.imwrite(r"C:\\Users\\Nate\\Desktop\\Cours_EPFL\\Robotique\\MA2\\DL\\Project\\img\\tracker\\track"+str(yo)+".jpg", frame)
                        
                        #if begin: 
                        #    cv2.imwrite(path_to_img+'\capTracked.jpg', frame)
                        #    begin = False

              #Id still here ?
                if self.IdToTrack not in IDs:
                    self.lostIdCount += 1
                else: 
                    self.lostIdCount = 0 #id is here, re initialize counter
                    output = outputs[np.where(self.IdToTrack == IDs)].flatten()
                    Xt1,Yt1,xt2,yt2=int(output[0]),int(output[1]),int(output[2]),int(output[3])
                    if(self.isCap):
                        #Yt1,yt2=int(Capcord[0][1]),int(Capcord[0][3])
                        pass
                    #xywhTrack=xyxy2xywh(np.expand_dims([Xt1,Yt1,xt2,yt2],axis=0))
                    self.Xtrack,self.Ytrack = (xt2+Xt1)//2,(yt2+Yt1)//2
                    self.LabelTrack = output[4]
            
              #wait clk*MaxIdCount time before saying we lost the Id: TO OPTIMIZE
                if self.lostIdCount > self.MaxIdCount:
                    print('WARNING: Lost person of Interest ! Do gesture of interest again')
                    self.doDetectSceleton = True
                    self.Xtrack,self.Ytrack = float("NAN"),float("NAN")
                    self.LabelTrack = float("NAN")
                    #doTracking = False #it is finally done in the beginning of the skeleton loop -> avoid entering in the YOLo below statement
                if self.verbose: print('tracking time :', time.time()-timeY)
            
            
        #loop through YOLO detections and draw them on transparent overlay image
        if not self.doTracking:
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
                    isSamePersonFromCap = False
                    #avoid having personDetected empty
                    box = (x1,y1,x2,y2)
                    if self.isCap: isSamePersonFromCap = IsPersonOfInterest(box,self.capLocation)
                    if hasDetected and self.doDetectSceleton: isSamePerson = isSamePersonDetector(personDetected,box) 
                    if(hasDetected and isSamePerson) or isSamePersonFromCap:
                    #locationOfPersonToTrack=((x1+x2)/2.0,(y1+y2)/2.0)
                        if hasDetected:
                            self.locationOfPersonToTrack=(xloc,yloc) #need to change this
                        elif isSamePersonFromCap:
                            self.locationOfPersonToTrack = self.capLocation
                        print("Detected at(x,y):(",self.locationOfPersonToTrack[0],self.locationOfPersonToTrack[1],")")
                        colorToUse=(255,0,0)
                        self.doDetectSceleton = False
                        self.doTracking = True
                        self.begin = True
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colorToUse, 2)
                    cv2.putText(frame, results.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorToUse, 2)
                    
                    #if (hasDetected and isSamePerson) or isSamePersonFromCap:
                    #    cv2.imwrite(path_to_img+'\capDetetcted.jpg', frame)
                    
        
        cv2.imshow("Annotated frame",frame)
        return [self.Xtrack,self.Ytrack],[float(self.LabelTrack)]







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