import os
import json
import pandas as pd
import numpy as np


def get_spectrum(delta):
    # shape (H,W,3):
    s = np.zeros(np.minimum(delta.shape[0],delta.shape[1]))
    for c in range(delta.shape[2]):
        s+= np.linalg.svd(delta[:,:,c],full_matrices=False,compute_uv=False)
    return s/3

data = []
SV_Data = []
eps = 40
nit= 30
for V in ["Video"+str(i) for i in range(1,16)]:
    root_dir =  './experiments/VTM-Data/'+V+'/FWNucl/'
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
                        if content['eps'] == eps and content['nit']==nit:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary ########')

print(df.describe())

print('###### End ########\n ')


np.save("mean",np.vstack(SV_Data).mean(axis=0))

np.save("sd",np.vstack(SV_Data).std(axis=0))


data = []
SV_Data = []


for V in ["View_00"+str(i) for i in range(1,8)]:
    root_dir =  './experiments/PETS09/'+V+'/FWNucl/'
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
                        if content['nit'] == 30 and content['eps']==40:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                            #delta = np.load(os.path.join(folder_path,'delta.npy'))

                           # SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary ########')

print(df.describe())

print('###### End ########\n ')



for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/FWNucl/'
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
                        if content['nit'] == 30 and content['eps']==40:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            #load delta.npy
                           # delta = np.load(os.path.join(folder_path,'delta.npy'))

                            #SV_Data.append(get_spectrum(delta))

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary ########')

print(df.describe())

print('###### End ########\n ')




