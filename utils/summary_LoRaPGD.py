import os
import json
import pandas as pd
import numpy as np


current_file_path = os.path.dirname(__file__)


def get_spectrum(delta):
    # shape (H,W,3):
    s = np.zeros(np.minimum(delta.shape[0],delta.shape[1]))
    for c in range(delta.shape[2]):
        s+= np.linalg.svd(delta[:,:,c],full_matrices=False,compute_uv=False)
    return s/3

data_01 = []
data_05 = []
data_1 = []
SV_Data_01 = []
SV_Data_05 = []
SV_Data_1 = []
rank=0.1
eps = 10.0

for V in ["Video"+str(i) for i in range(1,16)]:
    root_dir =  './experiments/VTM-Data/'+V+'/LoRa-PGD/'

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
                        if content['rank'] == 0.1 and content['eps']==eps and content['nit']==100:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_01.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data_01.append(get_spectrum(delta))
                        

                        elif content['rank'] == 0.5 and content['eps']==eps and content['nit']==100:
                            print(folder_path)
                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_05.append(content)
                            #load delta.npy
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data_05.append(get_spectrum(delta))

                            
                        elif content['rank'] == 1.0 and content['eps']==eps and content['nit']==100:
                            print(folder_path)
                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_1.append(content)
                            delta = np.load(os.path.join(folder_path,'delta.npy'))

                            SV_Data_1.append(get_spectrum(delta))
                            #load delta.npy


# Create DataFrame
df = pd.DataFrame(data_01)
print('###### Summary r=0.1 ########')

print(df.describe())

print('###### End ########\n ')

np.save(current_file_path+"/Plot_deltas/mean_PGD01.npy",np.median(np.vstack(SV_Data_01),axis=0))

np.save("sd-PGD01",np.vstack(SV_Data_01).std(axis=0))

df = pd.DataFrame(data_05)
print('###### Summary r=0.5 ########')
np.save(current_file_path+"/Plot_deltas/mean_PGD05.npy",np.median(np.vstack(SV_Data_05),axis=0))

print(df.describe())

print('###### End ########\n ')


df = pd.DataFrame(data_1)
print('###### Summary r=1.0 ########')
np.save(current_file_path+"/Plot_deltas/mean_PGD1.npy",np.median(np.vstack(SV_Data_1),axis=0))

print(df.describe())

print('###### End ########\n ')



data_01 = []
data_05 = []
data_1 = []
SV_Data = []
eps = 10.0

for V in ["View_00"+str(i) for i in range(1,8)]:
    root_dir =  './experiments/PETS09/'+V+'/LoRaPGD/'
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
                        if content['rank'] == 0.1 and content['eps']==eps and content['nit']==100:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_01.append(content)
                            #load delta.npy
                            #delta = np.load(os.path.join(folder_path,'delta.npy'))

                            #SV_Data.append(get_spectrum(delta))
                        elif content['rank'] == 0.5 and content['eps']==eps and content['nit']==100:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_05.append(content)

                        elif content['rank'] == 1.0 and content['eps']==eps and content['nit']==100:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data_1.append(content)
# Create DataFrame
df = pd.DataFrame(data_01)
print('###### Summary ########')

print(df.describe())


df = pd.DataFrame(data_05)
print('###### Summary ########')

print(df.describe())

df = pd.DataFrame(data_1)
print('###### Summary ########')

print(df.describe())


data = []
eps = 15.0
rank = 1.0

for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/LoRaPGD/'
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
                        if content['eps'] == eps and content['rank'] == rank and content['nit']==50:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary top r=1.0 ########')

print(df.describe())

print('###### End ########\n ')


data = []
data = []
#SV_Data = []
eps = 15.0
rank = 0.5

for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/LoRaPGD/'
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
                        if content['eps'] == eps and  content['rank'] == rank and  content['nit']==50:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary top r=0.5 ########')

print(df.describe())

print('###### End ########\n ')

eps = 15.0
rank = 0.1

for V in ["cam0","cam1","cam2"]:
    root_dir =  './experiments/EPFL/'+V+'/LoRaPGD/'
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
                        if content['eps'] == eps and content['rank']==rank and content['nit']==50:

                            content.pop('IOU_run', None)
                            content.pop('nuc_norm_run', None)
                            data.append(content)
                            

# Create DataFrame
df = pd.DataFrame(data)
print('###### Summary top r=0.1  ########')

print(df.describe())

print('###### End ########\n ')

