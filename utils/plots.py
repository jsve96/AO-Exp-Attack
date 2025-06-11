import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from data_config import *



####### Fig 1 
current_file_path = os.path.dirname(__file__)

print(current_file_path)

DATA = load_data(name='VTM-Data',source=DATA_PATH[0],instance='Video1')

from torchvision.utils import draw_bounding_boxes

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to('cuda:0')

#Perturbations
delta_L1 = np.load(current_file_path+"/Example_delta/L1_example/delta.npy")
delta_aoexp1 = np.load(current_file_path+"/Example_delta/AO-Exp/delta.npy") 
delta_aoexp = np.load(current_file_path+"/Example_delta/AO-Exp-top1/delta.npy") 

def filter(delta):
    return np.abs(delta).sum(axis=2)/3

from torchvision.utils import draw_bounding_boxes



fig,ax = plt.subplots(3,4,figsize=(18,7))

fig.colorbar(ax[0,-1].imshow(filter(delta_L1),cmap='binary',aspect='auto'),ax=ax[0,-1])
ax[0,-1].set_title('L1 Perturbation',fontsize=15)

fig.colorbar(ax[1,-1].imshow(filter(delta_aoexp1),cmap='binary',aspect='auto',vmin=filter(delta_L1).min(),vmax=filter(delta_L1).max()),ax=ax[1,-1])

ax[1,-1].set_title('Structured Perturbation',fontsize=15)

fig.colorbar(ax[2,-1].imshow(filter(delta_aoexp),cmap='binary',aspect='auto',vmin=filter(delta_L1).min(),vmax=filter(delta_L1).max()),ax=ax[2,-1])
ax[2,-1].set_title('Structured Perturbation (rank 1 update)',fontsize=15)



ax[0,1].set_title('Perturbed Frames',fontsize=15)


PLOTS = []

for i,idx in enumerate([0,150,210]):
    input = torch.tensor(DATA[idx],device='cuda').permute(2,0,1)
    input = input/255.0
    adv_e1 = (input + torch.tensor(delta_L1).to('cuda').permute(2,0,1)).clamp(0,1).to(torch.float32)
    adv_e2 = (input + torch.tensor(delta_aoexp1).to('cuda').permute(2,0,1)).clamp(0,1).to(torch.float32)
    adv_e3 = (input + torch.tensor(delta_aoexp).to('cuda').permute(2,0,1)).clamp(0,1).to(torch.float32)
    list_adve = [adv_e1,adv_e2,adv_e3]
    res = model([adv_e1,adv_e2,adv_e3])
    # boxes = res.boxes
    # scores = res.scores
    # labels = res
    for c in range(3):
        boxes = res[c]['boxes']
        scores = res[c]['scores']
        labels = res[c]['labels']
        print(list_adve[c].permute(1,2,0).shape)
        plot_adv = draw_bounding_boxes(list_adve[c].detach().cpu(),boxes=torch.tensor(boxes)[(scores > 0.8) & (labels==3)],colors=(0,255,0),width=8)
        PLOTS.append(plot_adv)
        ax[c,i].imshow(plot_adv.permute(1,2,0).numpy())

        
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])

fig.tight_layout()

fig.savefig(current_file_path+'/fig1.png')





####### Fig5 #############
#Fig 5a)
delta_aoexp = np.load(current_file_path+'Plot_deltas/mean.npy')
delta_aoexp_top1 = np.load(current_file_path+'Plot_deltas/mean_top1_VTM.npy')
delta_PGD01 = np.load(current_file_path+'Plot_deltas/mean_PGD01.npy')
delta_PGD05 = np.load(current_file_path+'Plot_deltas/mean_PGD05.npy')
delta_PGD1 = np.load(current_file_path+'Plot_deltas/mean_PGD1.npy')


fig, ax = plt.subplots(figsize=(10,5))
N = len(delta_aoexp)
marker_on = np.arange(0,N,25)
ax.plot(delta_PGD01,label='LoRa-PGD-10',color ='black',ls='-',markevery=marker_on,marker='*',alpha=0.5)
ax.plot(delta_PGD05,label='LoRa-PGD-50',color='black',ls='-',markevery=marker_on,marker='v',alpha=0.5)
ax.plot(delta_PGD1,label='LoRa-PGD-100',color='black',ls='-',markevery=marker_on,marker='^',alpha=0.5)
ax.plot(delta_aoexp,label='AO-Exp',color='blue',markevery=marker_on,marker='v')
ax.plot(delta_aoexp_top1,label='AO-Exp (LoRa)',color='red',markevery=marker_on,marker='v')

ax.set_yscale('log')
ax.set_ylabel('Value',size=20)
ax.set_xlabel('Order of Singular Values',size=20)
ax.legend()
ax.tick_params(axis='both', which='major', labelsize=15)
fig.tight_layout()

fig.savefig(current_file_path+'/SVD_Plot.pdf')

##### Fig 5b)

import os,json

SV_Data = []


Methods = ['AO-Exp10/','AO-Exp30/','AO-Exp50/','AO-Exp75/','AO-Exp/']
k = [0.1,0.3,0.5,0.75,1.0]
data = {}

for m in Methods:
    data[m] = []

for V in ["cam0","cam1","cam2"]:
    for m,k in zip(Methods,k):
        root_dir =  './experiments/EPFL/'+V+'/AO-Exp/'
        print(os.listdir(root_dir))
        counter = 0
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                # Find JSON file in folder
                for file in os.listdir(folder_path):
                    if file.endswith('.json'):
                        json_path = os.path.join(folder_path, file)
                        if counter == 0:
                            with open(json_path, 'r') as f:
                                content = json.load(f)
                                #print(content)
                                # Remove list fields
                                if content['l1'] == 0.75 and content['l2']==0.005 and content['nit']==50 and content['k']==k:

                                    content.pop('IOU_run', None)
                                    content.pop('nuc_norm_run', None)
                                    data[m].append(content['BoxRatio'])
                                    #load delta.npy
                                    delta = np.load(os.path.join(folder_path,'delta.npy'))
                                    counter=+1

                                #SV_Data.append(get_spectrum(delta))

PLOT_DATA = []
LABEL = []
for k,item in data.items():
    print(k, np.mean(item),np.std(item))
    LABEL.append(k)
    PLOT_DATA.append(item)

import matplotlib.cm as cm
import matplotlib.colors as mcolors

cmap = cm.get_cmap('winter',5)  # 'Blues' is a built-in colormap

blue_palette = [mcolors.to_hex(cmap(i)) for i in range(5)]
print(blue_palette)
#blue_palette[2] = '#808080'
blue_palette = ['blue','red','gray','darkred','black']

Labels = ['AO-Exp10','AO-Exp30','AO-Exp50','AO-Exp75','AO-Exp (full)']
alphas= [0.6,0.7,0.8,0.9,1.0]
fig,ax = plt.subplots(figsize=(6,5))
for i,m in enumerate(Labels):
    ax.plot(PLOT_DATA[i], marker='v',label=m,color=blue_palette[i],alpha=0.7)

ax.legend()
ax.set_xticks(ticks=[0,1,2],labels=['cam0','cam1','cam2'])
ax.locator_params(axis='y', nbins=4)
ax.set_ylabel('advBR',size=20)
ax.set_xlabel('Camera Views',size=20)
fig.tight_layout()
fig.savefig(current_file_path+'/EPFL_BoxRatios.pdf')





