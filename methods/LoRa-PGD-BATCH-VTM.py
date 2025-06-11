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
from data_config import *
import argparse


# ### Path to zip folder
# zip_path = "/home/jacob/datasets/Video_006.zip"
# #zip_path = "/home/sven/Downloads/Video_006.zip"

# img_folder = "Video_006/Video_006/"

# frames = read_data(zip_path)

# indices = list(np.arange(0,len(frames)))
# DATA = [frames[i] for i in indices]

# print('CUDA')
# print(torch.cuda.is_available())
# print('Devices')
# print(torch.cuda.device_count())
# DEVICE = torch.device('cuda:1')
# print(DEVICE)


# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()
# model.to(DEVICE)



class LoRaPGD:
    def __init__(self,data,device,model):
        self.data = data
        self.device = device
        self.eps_for_div = 1e-10
        self.model= model
        self.iou_loss = 0
        self.THRESHOLD = 0.8
        self.IOU_RUN =[]
        self.Nuc_NORM_RUN = []
        self.BoxRatio = []
    
    def Attack(self,nit=10,rank=5,method='lora',batch_size=30,eps=1.2):
       # assign shapes

        #c,h, w  = self.data[0].shape
        h, w,c  = self.data[0].shape

        #print("c",c,"h",h,'w',w)
        #first initialize u and v
        if method == 'lora':

            #u = torch.randn([c, h, rank], device=self.device)
            u = torch.randn([h,rank,c], device=self.device)
            norm_u = torch.norm(u.view(-1), p=2)
            u = (u / norm_u).detach()

            #v = torch.zeros([c, rank, w], device=self.device).detach()
            v = torch.zeros([rank, w,c], device=self.device).detach()


        elif method == 'fgsm':
            u, v = None
            u = u[:, :, :rank].detach().to(self.device)
            v = v[:, :rank, :].detach().to(self.device)

        for _ in range(nit):
            print(_)
            self.iou_loss = 0
            #u.requires_grad = True
            #v.requires_grad = True
            #update perturbation
           # delta = torch.einsum('cik,ckj ->cij',u,v)
            delta = torch.einsum('hrc,rwc ->hwc',u,v)

            print(delta.shape)

            delta_norm = torch.norm(delta.reshape(-1), p=2) + self.eps_for_div
            num_batches=0
            #empty_grad
            g_u = torch.zeros_like(u)
            g_v = torch.zeros_like(v)
            #iterate over batches
            ngt_boxes = 0
            nad_boxes= 0



            
                #delta = torch.einsum('cik,ckj ->cij',u,v)
            delta = torch.einsum('hrc,rwc ->hwc',u,v)
            delta_norm = torch.norm(delta.reshape(-1), p=2) + self.eps_for_div
            delta = eps * delta / delta_norm

            self.Nuc_NORM_RUN.append(np.float64(nuc_nor(delta.detach().cpu())))
            print(self.Nuc_NORM_RUN[-1])


            for BATCH in batch(self.data,batch_size):
                #zero_grads
                u.requires_grad = True
                v.requires_grad = True

                #delta = torch.einsum('cik,ckj ->cij',u,v)
                delta = torch.einsum('hrc,rwc ->hwc',u,v)
                delta_norm = torch.norm(delta.reshape(-1), p=2) + self.eps_for_div
                delta = eps * delta / delta_norm

                #empty loss
                loss = 0
                #print(loss)
                num_batches +=1
                #loop over batches to accumulate gradient
                for i,frame in enumerate(BATCH):
                    input = torch.tensor(frame,device=self.device).permute(2,0,1)
                    input = input/255.0
                    #print("####",i)
                    adv_e = (input + delta.permute(2,0,1)).clamp(0,1)

                    predictions = self.model([input])
                    #print('clean:', torch.sum((predictions[0]['scores']>=self.THRESHOLD)&(predictions[0]['labels']==3)))
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
                    #print('adv:', torch.sum((adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)))
                
                    if adv_pred[0]['scores'].size(0)==0 or adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].size(0)==0:
                        loss_fg,loss_bg= 0,0
                        confl = 0
                        self.iou_loss +=0
                        #counter+=1
                        
                    else:
                        masks_adv = adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].sum(dim=0).clamp(0,1)[0,:,:]
                        adv_boxes = adv_pred[0]['boxes'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].detach().cpu()
                        self.iou_loss+= IOU_Loss(gt_boxes,adv_boxes)
                        loss_fg,loss_bg = get_CE_Losses(masks_adv,masks) 
                        confl = conf_loss(adv_pred[0]['scores'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)])
                       
                    loss += -0.1*loss_bg+ 0.1*loss_fg + 0*confl
                #print(loss)
                #loss.backward()
                if isinstance(loss,int) or isinstance(loss,float):
                    continue
                else:
                    loss.backward()
                   
            
                    g_u+=u.grad#.detach()
                    g_v+=v.grad#grad_v.detach()

            g_u = g_u/num_batches
            g_v = g_v/num_batches

            norm_u = torch.norm(g_u.view(-1), p=2) + self.eps_for_div
            norm_v = torch.norm(g_v.view(-1), p=2) + self.eps_for_div

            u = (u + g_u / norm_u).detach()
            v = (v + g_v / norm_v).detach()


            self.iou_loss = self.iou_loss/len(self.data)
            print(self.iou_loss)
            self.Attack_rate = 1-nad_boxes/ngt_boxes
            print(self.Attack_rate)
            print('BoxRatio:',nad_boxes/ngt_boxes)
            self.BoxRatio.append((nad_boxes/ngt_boxes).detach().cpu().numpy())
            #print(delta)
            if isinstance(self.iou_loss, float)== True:
                self.IOU_RUN.append(self.iou_loss)
            else:
            #self.Nuc_NORM_RUN.append(np.float64(nuc_nor(delta.detach().cpu())))
                self.IOU_RUN.append(np.float64((self.iou_loss).detach().cpu().numpy()))
            
        
        delta = torch.einsum('hrc,rwc->hwc', u, v)
        delta_norm = torch.norm(delta.reshape(-1), p=2) + self.eps_for_div
        delta = eps * delta / delta_norm

        self.Nuc_NORM_RUN.append(np.float64(nuc_nor(delta.detach().cpu())))

        return delta.detach().cpu().numpy()
    
import uuid


def main(args):

    data = load_data(name=args.dataset,source=args.path_dataset,instance=args.instance)

    print('CUDA')
    print(torch.cuda.is_available())
    print('Devices')
    print(torch.cuda.device_count())
    DEVICE = torch.device('cuda:2')
    print(DEVICE)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(DEVICE)
    ##### #run attack
    loraPGD = LoRaPGD(data=data,device=DEVICE,model=model)

    rank = int(args.rank*np.minimum(data[0].shape[0],data[0].shape[1]))
    print(args.rank,rank,data[0].shape)
    Result = loraPGD.Attack(rank=rank,nit=args.nit,eps=args.eps)

    print(Result.shape)

    if isinstance(loraPGD.iou_loss,float):
        l = loraPGD.iou_loss
    else:
        l = np.float64(loraPGD.iou_loss.detach().cpu().numpy())

    Metrics = {'dataset':args.dataset,'rank':args.rank,'eps':args.eps,'nit':args.nit,'IOU':l,'IOU_run':loraPGD.IOU_RUN,'nuc_norm_run':loraPGD.Nuc_NORM_RUN}
    Metrics['MAP'] = np.float64(np.sum(np.mean(np.abs(Result),axis=2))/(Result.shape[0]*Result.shape[1]))
    Metrics['AttackRate'] = np.float64(loraPGD.Attack_rate.detach().cpu().numpy())
    Metrics['BoxRatios'] = loraPGD.BoxRatio
    Metrics['BoxRatio'] = loraPGD.BoxRatio[-1]

    perturbation_tensor = torch.tensor(Result)

    print("nuc_norm: {}".format(nuc_nor(perturbation_tensor)))
    nuc_norm = nuc_nor(perturbation_tensor)

    Metrics['nuc_norm'] = np.float64(nuc_norm)


    #### make new directory for run 

    run_name = uuid.uuid4().hex

    if not os.path.exists(args.save+"/"+run_name):
        os.makedirs(args.save+"/"+run_name)
    
    np.save(args.save+"/"+run_name+"/delta.npy",Result)

    json_path = args.save+"/"+run_name+"/Metrics_new.json"
    print(Metrics)


    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(Metrics, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)


from itertools import product
if __name__ == "__main__":
    
    Data = ['VTM-Data']
    instance = ['Video'+ str(i) for i in range(1,16)]#['Video10'] #['VTM/Sample1-30']
    #Save = ['/home/jacob/Repos/python/VideoAttacks/experiments/VTM-Data/'+instance[0]+"/AO-Exp"]
    Path_Data = ["/home/jacob/datasets"]
    Nit = [100]
    rank = [1.0]#,1.0] #[0.1,0.5,1.0]#,0.25,0.5]#[0.1,0.5,1.0]
    #eps = [0.5,1.0,3.0]
    eps=[10.0]


    for dataset,path_dataset,nit,eps,rank,instance in product(Data,Path_Data,Nit,eps,rank,instance):
        args = argparse.Namespace(
            dataset=dataset,
            path_dataset = path_dataset,
            nit = nit,
            eps=eps,
            rank=rank,
            save='/home/jacob/Repos/python/VideoAttacks/experiments/VTM-Data/'+instance+"/LoRa-PGD",
            instance = instance
        )
        print(args.dataset)
        main(args) 

    #main(args) 

