import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
current_file_path = os.path.dirname(__file__)


def get_spectrum(delta):
    # shape (H,W,3):
    s = np.zeros(np.minimum(delta.shape[0],delta.shape[1]))
    for c in range(delta.shape[2]):
        s+= np.linalg.svd(delta[:,:,c],full_matrices=False,compute_uv=False)
    return s/3


data = []
SV_Data = []
L1 = 0.1
for V in ["Video"+str(i) for i in range(1,16)]:
    root_dir =  './experiments/VTM-Data/'+V+'/AO-Exp/'
    print(root_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        # Remove list fields
                        if content['l1'] == L1 and content['l2']==0.0002 and content['nit']==2 and content['k']==1.0:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary CW4C ########')

print(df.describe())

print('###### End ########\n ')


np.save(current_file_path+"/Plot_deltas/mean.npy",np.median(np.vstack(SV_Data),axis=0))



data = []
SV_Data = []
L1 = 0.1
for V in ["Video"+str(i) for i in range(1,16)]:
    root_dir =  './experiments/VTM-Data/'+V+'/AO-Exp'

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        # Remove list fields
                        if content['l1'] == L1 and content['l2']==0.001 and content['nit']==50 and content['k']=='top1':

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary CW4C top1 ########')

print(df.describe())

print('###### End top1 ########\n ')


np.save(current_file_path+"/Plot_deltas/mean_top1_VTM.npy",np.median(np.vstack(SV_Data),axis=0))



data = []
SV_Data = []
L1 = 0.1
frac = 10

for V in ["View_00"+str(i) for i in range(1,8)]:
    root_dir =  './experiments/PETS09/'+V+'/AO-Exp/'
    #print(root_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        # Remove list fields
                        if content['l1'] == L1 and content['l2']==0.01 and content['nit']==100 and content['k']==1.0:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary PETS ########')

print(df.describe())

print('###### End ########\n ')


#np.save(current_file_path+"/Plot_deltas/mean_PETS.npy",np.vstack(SV_Data).mean(axis=0))




data = []
#SV_Data = []
L1 = 0.1
frac = 10

for V in ["View_00"+str(i) for i in range(1,8)]:
    root_dir =  './experiments/PETS09/'+V+'/AO-Exp/'
    #print(root_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        # Remove list fields
                        if content['l1'] == L1 and content['l2']==0.002 and content['nit']==50 and content['k']=='top1':

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            #delta = np.load(os.path.join(folder_path,'delta.npy'))

                            #SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary PETS top1 ########')

print(df.describe())

print('###### End ########\n ')



data = []
SV_Data = []
L1 = 0.75

for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/AO-Exp/'
    #print(root_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        # Remove list fields
                        if content['l1'] == L1 and content['l2']==0.005 and content['nit']==50 and content['k']==1.0:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary EPFL ########')

print(df.describe())

print('###### End ########\n ')


data = []
SV_Data = []
L1 = 0.75


for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/AO-Exp/'
    #print(root_dir)

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Find JSON file in folder
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    json_path = os.path.join(folder_path, file)
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        #print(content)
                        if content['l1'] == L1 and content['l2']==0.005 and content['nit']==50 and content['k']=='top1':

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary EPFL top 1 ########')

print(df.describe())

print('###### End ########\n ')
