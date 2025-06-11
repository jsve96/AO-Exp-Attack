import matplotlib.pyplot as plt
import numpy as np
import zipfile
#from PIL import Image
import cv2
import os
import torchvision
import torch
import torchvision.transforms as transforms
from utils import *
import matplotlib.image as mpimg
import time
import argparse
from data_config import *


class AoEXP_Attack_L1:
    def __init__(self,THRESHOLD,device,model,data):
        self.THRESHOLD = THRESHOLD
        self.DEVICE = device
        self.model = model
        self.data = data
        self.IOU_Loss = 0
        self.Attack_rate = 0
        self.IOU_RUN =[]
        self.Nuc_NORM_RUN = []
        self.BoxRatio =[]



    def wrapper_func_p(self,x):
        self.THRESHOLD = 0.8
        loss = 0
        #SIZE = (3,240, 320)
        x = x.copy()
        x = torch.tensor(x.copy(),device=self.DEVICE,dtype=torch.float32).permute(2,0,1).requires_grad_(True)
        counter = 0
        g = torch.zeros_like(x)
        num_batches=0
        self.iou_loss = 0
        ngt_boxes = 0
        nad_boxes= 0


        for BATCH in batch(self.data,30):
            
            x.requires_grad_(True)
            loss = 0
            num_batches+=1
            for i,frame in enumerate(BATCH):
                input = torch.tensor(frame,device=self.DEVICE).permute(2,0,1)
                input = input/255.0
                #print("####",i)
                adv_e = (input + x).clamp(0,1)

                predictions = self.model([input])
                CleanBoxes =  torch.sum((predictions[0]['scores']>=self.THRESHOLD)&(predictions[0]['labels']==3))
                ngt_boxes += CleanBoxes
               # print('clean:', CleanBoxes)

                masks = predictions[0]['masks'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==3)]
                gt_boxes =  predictions[0]['boxes'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==3)].detach().cpu()

                if masks.size(0)==0:
                    print('skip')
                    continue

                masks = masks.sum(dim=0).squeeze(0).clamp(0,1)

                adv_pred = self.model([adv_e])
                Adv_Boxes = torch.sum((adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3))
                nad_boxes+=Adv_Boxes
               # print('adv:',Adv_Boxes)
            
                if adv_pred[0]['scores'].size(0)==0 or adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].size(0)==0:
                    loss_fg,loss_bg, confl = 0,0,0
                    self.IOU_Loss +=0
                    counter+=1

                   
                
                    
                else:
                    masks_adv = adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].sum(dim=0).clamp(0,1)[0,:,:]
                    adv_boxes = adv_pred[0]['boxes'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].detach().cpu()
                    self.iou_loss+= IOU_Loss(gt_boxes,adv_boxes)
                    loss_fg,loss_bg = get_CE_Losses(masks_adv,masks) 
                    #loss_dice = 0*(dice_loss(masks_adv,masks))
                    confl = conf_loss(adv_pred[0]['scores'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)])
                loss += -1*loss_bg+ 1*loss_fg+0.001*confl #+ 0.1*(nuclear_norm(adv_e)-nuclear_norm(input))**2#20000*loss_fg  #+ 5*conf_loss(adv_pred[0]['scores'][adv_pred[0]['scores']>= THRESHOLD]) #- 0.4*(nuclear_norm(adv_e)-nuclear_norm(input))**2
           
            
            if isinstance(loss,int) or isinstance(loss,float):
                continue
            else:
                loss.backward()
                g += x.grad


        print(self.iou_loss/len(self.data))
        self.IOU_RUN.append((self.iou_loss/len(self.data)).detach().cpu().numpy())
        self.Attack_rate = 1-nad_boxes/ngt_boxes
        print(self.Attack_rate)
        self.Nuc_NORM_RUN.append(nuclear_norm(x).detach().cpu().numpy())
        self.BoxRatio.append((nad_boxes/ngt_boxes).detach().cpu().numpy())
        print('BoxRatio:', self.BoxRatio[-1])
        grad = (g/num_batches).permute(1,2,0).detach().cpu().numpy()
        print(self.Nuc_NORM_RUN[-1])
        return grad




from ao_exp_grad_tensor import AOExpGrad,fmin
from ao_exp_grad_L1 import fmin as fmin_L1
import uuid


def main(args):

    data = load_data(name=args.dataset,source=args.path_dataset,instance=args.instance)
    print(torch.cuda.is_available())
    print('Devices')
    print(torch.cuda.device_count())
    DEVICE = torch.device('cuda:2')
    print(DEVICE)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(DEVICE)


    THRESHOLD  =0.8

    wrapper_func = None

    Aoexp_attack = AoEXP_Attack_L1(THRESHOLD,DEVICE,model,data)


    x_init = np.zeros(shape=(data[0].shape))
    start =time.time()
    Result = fmin_L1(wrapper_func,Aoexp_attack.wrapper_func_p,x_init,lower=-1,upper=1,eta=0.55,maxfev=args.nit,l1=args.l1,l2=args.l2,callback=None)
    end = time.time()
    print(end-start)
    print(Aoexp_attack.iou_loss)
    #Result = fmin_L1(wrapper_func,wrapper_func_p,x_init,lower=-1,upper=1,eta=0.55,maxfev=150,l1=0.0001,l2=0.00005,callback=None,epoch_size=10)


    def filter_perturbation(P,eps):
        P = np.clip(np.abs(P),0,1)#/np.clip(P,0,1).max()
        P[P<eps]=0
        return P.mean(axis=2)


    try:
        l=Aoexp_attack.iou_loss.detach().cpu().numpy()
    except:
        l = Aoexp_attack.iou_loss
    

    Metrics = {'dataset':args.dataset,'l1':args.l1,'l2':args.l2,'nit':args.nit,'IOU':l,'IOU_run':Aoexp_attack.IOU_RUN,'nuc_norm_run':Aoexp_attack.Nuc_NORM_RUN}
    Metrics['MAP'] = np.sum(np.mean(np.abs(Result),axis=2))/(Result.shape[0]*Result.shape[1])
    Metrics['AttackRate'] = Aoexp_attack.Attack_rate.detach().cpu().numpy()
    Metrics['BoxRatios'] = Aoexp_attack.BoxRatio
    Metrics['BoxRatio'] = Aoexp_attack.BoxRatio[-1]

    perturbation_tensor = torch.tensor(Result)

    print("nuc_norm: {}".format(nuc_nor(perturbation_tensor)))
    nuc_norm = nuc_nor(perturbation_tensor)

    Metrics['nuc_norm'] = nuc_norm


    #### make new directory for run 

    run_name = uuid.uuid4().hex
    # if not os.path.exists(args.save):
    #     os.mkdirs(args.save)
    if not os.path.exists(args.save+"/"+run_name):
        print('save')
        os.makedirs(args.save+"/"+run_name)
    
    np.save(args.save+"/"+run_name+"/delta.npy",Result)

    #torch.save(perturbation_tensor.detach().cpu(), "Perturbation_aoexp_VTM.pt")

    #plt.imshow(np.clip(Result.x,0,1)/Result.x.max())
    plt.imshow(filter_perturbation(Result,0.01),cmap='binary')
    plt.savefig('Test.png')



    json_path = args.save+"/"+run_name+"/Metrics.json"
    

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(Metrics, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)



if __name__ == "__main__":
    from itertools import product
   

    Data = ['VTM-Data']
    instance = ['Video'+ str(i) for i in range(1,2)]#['Video10'] #['VTM/Sample1-30']
    #Save = ['/home/jacob/Repos/python/VideoAttacks/experiments/VTM-Data/'+instance[0]+"/AO-Exp"]
    Path_Data = ["/home/jacob/datasets"]
    Nit = [20]
    #L1 = [0.005,0.01,0.075,0.1,0.001]
    L1 = [0.008,0.009,0.01]#[0.05,0.1,0.08,0.09] #[0.075,0.01,0.5]
    #L2 = [0.0001,0.0005,0.005,0.0075]
    frac = [500]#,100,500]

    for dataset,path_dataset,nit,l1,frac,instance in product(Data,Path_Data,Nit,L1,frac,instance):
        args = argparse.Namespace(
            dataset=dataset,
            path_dataset = path_dataset,
            nit = nit,
            l1=l1,
            l2=0,#0.1,#0.001,
            save='/home/jacob/Repos/python/VideoAttacks/experiments/VTM-Data/'+instance+"/L1_example",
            instance = instance
        )
        print(args.dataset)
        main(args) 
