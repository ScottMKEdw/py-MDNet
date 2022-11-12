import os
import numpy as np
import pickle
import json
import re
import pandas as pd

id=[]

annotation_folder = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\datasets\vidVRD\vidvrd-annotations\train'
output_path = r'C:\Users\scott\Documents\School\GitHub\MDNet\py-MDNet\pretrain/data/vidVRD.pkl'


#def train_or_test(input):
#  if input == 'train': 
#    annotation_folder = os.path.join(annotaton_folder, 'train')

#  elif input == 'test':
#    annotation_folder = os.path.join(annotation_folder, 'test')

annotation_list = os.listdir(annotation_folder)

for value in annotation_list:
  split = re.split(r'[_.]', value)
  id.append(split[2])
  
xmin = []
xmax = []
ymin = []
ymax = []
tid = []

with open(os.path.join(annotation_folder,'ILSVRC2015_train_00005003.json')) as f:
  data = json.load(f)

  for i in range(len(data['trajectories'])):
    for j in range(len(data['subject/objects'])):
      tid.append( data['trajectories'][i][j]['tid'])
      
      xmin.append( data['trajectories'][i][j]['bbox']['xmin'])
      xmax.append( data['trajectories'][i][j]['bbox']['xmax'])
      ymin.append( data['trajectories'][i][j]['bbox']['ymin'])
      ymax.append( data['trajectories'][i][j]['bbox']['ymax'])

coords = np.array([xmin, xmax, ymin, ymax])
corners = np.transpose(np.array([coords[0], coords[2], coords[1]-coords[0], coords[3]-coords[2]]))

data = {'images': id, 'corners': corners}

output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp)




