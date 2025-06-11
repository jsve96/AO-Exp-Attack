import numpy as np
import os
import torchvision
import torch
import time
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from utils.data_config import *

class AoEXP_Attack:
    def __init__(self,THRESHOLD,device,model,data):
        self.THRESHOLD = THRESHOLD
        self.DEVICE = device
        self.model = model
        self.data = data
        self.IOU_Loss = 0
        self.Attack_rate = 0
        self.IOU_RUN =[]
        self.Nuc_NORM_RUN = []
        self.BoxRatio = []



    def wrapper_func_p(self,x):
        self.THRESHOLD = 0.8
        loss = 0
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
                adv_e = (input + x).clamp(0,1)

                predictions = self.model([input])
                CleanBoxes =  torch.sum((predictions[0]['scores']>=self.THRESHOLD)&(predictions[0]['labels']==1))
                ngt_boxes += CleanBoxes
                print('clean:', CleanBoxes)

                masks = predictions[0]['masks'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==1)]
                gt_boxes =  predictions[0]['boxes'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==1)].detach().cpu()

                if masks.size(0)==0:
                    print('skip')
                    continue

                masks = masks.sum(dim=0).squeeze(0).clamp(0,1)

                adv_pred = self.model([adv_e])
                Adv_Boxes = torch.sum((adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 1))
                nad_boxes+=Adv_Boxes
                print('adv:',Adv_Boxes)
            
                if adv_pred[0]['scores'].size(0)==0 or adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 1)].size(0)==0:
                    loss_fg,loss_bg, confl = 0,0,0
                    self.IOU_Loss +=0
                    counter+=1

                   
                
                    
                else:
                    masks_adv = adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 1)].sum(dim=0).clamp(0,1)[0,:,:]
                    adv_boxes = adv_pred[0]['boxes'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 1)].detach().cpu()
                    self.iou_loss+= IOU_Loss(gt_boxes,adv_boxes)
                    loss_fg,loss_bg = get_CE_Losses(masks_adv,masks) 
                    confl = conf_loss(adv_pred[0]['scores'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 1)])
                loss += -1*loss_bg+ 1*loss_fg+0.001*confl 
           
            
            if isinstance(loss,int) or isinstance(loss,float):
                continue
            else:
                loss.backward()
                g += x.grad
        print(self.iou_loss/len(self.data))
        self.IOU_RUN.append((self.iou_loss/len(self.data)).detach().cpu().numpy())
        self.Attack_rate = 1-nad_boxes/ngt_boxes
        print(self.Attack_rate)
        self.BoxRatio.append((nad_boxes/ngt_boxes).detach().cpu().numpy())
        self.Nuc_NORM_RUN.append(nuclear_norm(x).detach().cpu().numpy())
        print(self.Nuc_NORM_RUN[-1])
        grad = (g/num_batches).permute(1,2,0).detach().cpu().numpy()
        self.iou_loss = self.iou_loss/len(self.data)

        return grad




from ao_exp_grad_tensor import AOExpGrad,fmin
import uuid


def main(args):

    data = load_data(name=args.dataset,source=args.path_dataset,instance=args.instance)

    DEVICE = args.DEVICE
    print(DEVICE)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(DEVICE)


    THRESHOLD  =0.8

    wrapper_func = None

    Aoexp_attack = AoEXP_Attack(THRESHOLD,DEVICE,model,data)


    x_init = np.zeros(shape=(data[0].shape))
    start =time.time()
    Result = fmin(wrapper_func,Aoexp_attack.wrapper_func_p,x_init,lower=-1,upper=1,eta=0.55,maxfev=args.nit,l1=args.l1,l2=args.l2,callback=None,k=args.k)
    end = time.time()
    print(end-start)
    print(Aoexp_attack.iou_loss)


   

    Metrics = {'dataset':args.dataset,'l1':args.l1,'l2':args.l2,'k':args.k,'nit':args.nit,'IOU':Aoexp_attack.iou_loss.detach().cpu().numpy(),'IOU_run':Aoexp_attack.IOU_RUN,'nuc_norm_run':Aoexp_attack.Nuc_NORM_RUN}
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

    if not os.path.exists(args.save+"/"+run_name):
        os.makedirs(args.save+"/"+run_name)
    
    np.save(args.save+"/"+run_name+"/delta.npy",Result)


    json_path = args.save+"/"+run_name+"/Metrics.json"
    

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(Metrics, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)



if __name__ == "__main__":
    from itertools import product

    Data = ['PETS09']
    instance = ['View_00'+ str(i) for i in range(1,8)]
    Path_Data = DATA_PATH
    L1 = [0.1]
    frac = [10]
    k = ['top1',1.0]
    DEVICE = torch.device('cuda:0')


    for dataset,path_dataset,l1,frac,instance,k in product(Data,Path_Data,L1,frac,instance,k):
        args = argparse.Namespace(
            dataset=dataset,
            path_dataset = path_dataset,
            nit = 50 if k =='top1' else 100,
            l1=l1,
            l2=l1/frac,
            save=REPO_PATH+'/experiments/PETS09'+"/"+instance+"/AO-Exp",
            instance = instance,
            k = k,
            DEVICE = DEVICE
        )
        print(args.dataset)
        main(args) 
