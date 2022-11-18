import os
import numpy as np
import pickle
import cv2
import json
import re
import pandas as pd
from collections import OrderedDict
      
class vidVRD_Data_Preparation():
  def __init__(self):
    self.testvids_path = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\datasets\vid_VRD_Data\vidvrd_test_vids'
    self.images_path = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\datasets\vid_VRD_Data\vidVRD_TestData'
    self.annotation_folder = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\datasets\vidVRD\vidvrd-annotations\Annotation_Test'
    self.annotation_list = os.listdir(self.annotation_folder)
    self.output_path = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\datasets\vidVRD_Pickle.pkl'
  
    
  def Create_Pickle_Out(self):
    self.annotation_data = OrderedDict()
    for value in self.annotation_list:
      value = value.replace('.json','')

      #print(os.path.join(self.images_path, value))
      img_list_final = sorted([p for p in os.listdir(os.path.join(self.images_path, value)) if os.path.splitext(p)[1] == '.jpg'])
      
      #img_list_final = [os.path.join(self.images_path, value, img) for img in img_list_final]
      self.annotation_data[value] =  {'images': img_list_final, 'gt': self.corners}

    #print(annotation_data)
    #print(len(img_list_final))

  def Pickle_Send(self):
    
    output_dir = os.path.dirname(self.output_path)
    print(self.output_path)
    os.makedirs(output_dir, exist_ok=True)

    pickle.dump(self.annotation_data, open(self.output_path, 'wb'))
    #print(self.annotation_data)
    
    
  def Vid_to_Imgs(self, vid_id, vid_count):
    #print(os.path.join(self.testvids_path, vid_id + '.mp4' ))
    os.makedirs(os.path.join(self.images_path, vid_id), exist_ok=True)
    vidcap = cv2.VideoCapture(os.path.join(self.testvids_path, vid_id + '.mp4'))
    success, image = vidcap.read()
    if success:
      print('Video Number: ', vid_count)
    frame_count = 1

    while success:
      #print(os.path.join(self.images_path, vid_id, str(frame_count) + '.jpg'))
      cv2.imwrite(os.path.join(self.images_path, vid_id, str(frame_count) + '.jpg'), image)
      success, image = vidcap.read()
      if success:
        #print('Saved Image: ', frame_count)
        frame_count+=1
  
        
  def Create_Annotations(self):
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    tid = []

    vid_count = 0

    for value in self.annotation_list:
      vid_count +=1

      with open(os.path.join(self.annotation_folder, value)) as f:
        data = json.load(f)
        vid_id = data['video_id']
        print (vid_id)
        self.Vid_to_Imgs(vid_id, vid_count)

        for i in range(len(data['trajectories'])):
          
          try:          
            #print(data['trajectories'][i][0]['bbox']['xmin'])
            tid.append( data['trajectories'][i][0]['tid'])
            

            if tid[i] == 0:
              xmin.append( data['trajectories'][i][0]['bbox']['xmin'])
              xmax.append( data['trajectories'][i][0]['bbox']['xmax'])
              ymin.append( data['trajectories'][i][0]['bbox']['ymin'])
              ymax.append( data['trajectories'][i][0]['bbox']['ymax'])
            
          except:
            xmin.append(0)
            xmax.append(0)
            ymin.append(0)
            ymax.append(0)
        

        
        
        #print(os.path.join(self.images_path, vid_id))
      
    
      img_list = sorted([p for p in os.listdir(os.path.join(self.images_path, vid_id)) if os.path.splitext(p)[1] == '.jpg'])
      img_list = [os.path.join(self.images_path, vid_id, img) for img in img_list]

      if len(img_list) != len(data['trajectories']):
        for i in range(len(img_list) - len(data['trajectories'])):
          os.remove(os.path.join(self.images_path, vid_id, str(len(img_list) - i) + '.jpg'))

      coords = np.array([xmin, xmax, ymin, ymax])
      self.corners = np.transpose(np.array([coords[0], coords[2], coords[1]-coords[0], coords[3]-coords[2]]))

    self.Create_Pickle_Out()
    


test = vidVRD_Data_Preparation()
test.Create_Annotations()
test.Pickle_Send()


