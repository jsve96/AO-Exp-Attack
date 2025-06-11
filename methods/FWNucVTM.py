import torch 
import math
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
import os
from utils import *
import matplotlib.image as mpimg
from data_config import *
import argparse
from utils import *




class FWnucl:
    def __init__(self, data,model,device, iters=10, img_range=(-1, 1),
                  eps=5):
        '''
        Implementation of the nuclear group norm attack.

        args:
        model:         Callable, PyTorch classifier.
        ver:           Bool, print progress if True.
        img_range:     Tuple of ints/floats, lower and upper bound of image
                       entries.
        targeted:      Bool, given label is used as a target label if True.
        eps:           Float, radius of the nuclear group norm ball.
        '''

        self.iters = iters
        self.eps = eps
        self.gr = (math.sqrt(5) + 1) / 2
        self.data =data 
        self.device = device
        self.model= model
        self.iou_loss = 0
        self.THRESHOLD = 0.8
        self.IOU_RUN =[]
        self.Nuc_NORM_RUN = []
        self.BoxRatio = []
    

    def Attack(self,batch_size=30):
    

        h,w,c  = self.data[0].shape

        delta = torch.zeros((h,w,c),device =self.device, requires_grad = True)


        for t in range(self.iters):
             
            grad_acc = torch.zeros_like(delta)
            num_batches = 0
            ngt_boxes = 0
            nad_boxes = 0
            self.iou_loss = 0
            for BATCH in batch(self.data,batch_size):

                loss = 0
                num_batches +=1

                for i,frame in enumerate(BATCH):

                    input = torch.tensor(frame,device=self.device).permute(2,0,1)
                    input = input/255.0

                    adv_e = (input + delta.permute(2,0,1)).clamp(0,1)

                    predictions = self.model([input])
                    CleanBoxes =  torch.sum((predictions[0]['scores']>=self.THRESHOLD)&(predictions[0]['labels']==3))
                    ngt_boxes += CleanBoxes
                    masks = predictions[0]['masks'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==3)]
                    gt_boxes =  predictions[0]['boxes'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==3)].detach().cpu()


                    if masks.size(0)==0:
                        print('skip')
                        continue

                    masks = masks.sum(dim=0).squeeze(0).clamp(0,1)

                    adv_pred = self.model([adv_e])
                    Adv_Boxes = torch.sum((adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3))
                    nad_boxes+=Adv_Boxes
                    
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
                       
                    loss += -1*loss_bg+ 1*loss_fg + 0.001*confl

                
                if isinstance(loss,int) or isinstance(loss,float):
                    continue
                else:
                    loss.backward()

                    grad_acc+=delta.grad.detach()


                
            grad_acc/=num_batches

            ### Apply Nuclear norm LMO 
            #s --> H x W x C
            s = self.groupNuclearLMO_HWC(grad_acc, eps=self.eps)
            print("Nuc Group Update")


            ### Line Search 
            with torch.no_grad():
                print('start Line Search')
                gamma = self.__lineSearchUniversal(delta,s,batch_size)

            #FW Update speed up use fix gamma
            #gamma=0.1
            delta = ((1-gamma)*delta + gamma * s).detach()
            delta.requires_grad = True


            print(self.iou_loss/len(self.data))
            self.IOU_RUN.append((self.iou_loss/len(self.data)).detach().cpu().numpy())
            self.Attack_rate = 1-nad_boxes/ngt_boxes
            print(self.Attack_rate)
            self.Nuc_NORM_RUN.append(nuclear_norm(delta).detach().cpu().numpy())
            self.BoxRatio.append((nad_boxes/ngt_boxes).detach().cpu().numpy())



        return delta.detach().cpu().numpy()



    def groupNuclearLMO_HWC(self, x,eps):
        '''
        LMO for the nuclear group norm ball, adapted for input in (H, W, C) format.
        '''

        # Convert HWC → BCHW
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        B, C, H, W = x.shape

        size = 32 if H > 64 else 4

        # Split per channel and extract patches
        xrgb = [x[:, c, :, :] for c in range(C)]
        xrgb = [xc.unfold(1, size, size).unfold(2, size, size) for xc in xrgb]
        xrgb = [xc.reshape(-1, size, size) for xc in xrgb]

        # Compute nuclear norms
        norms = torch.linalg.svdvals(xrgb[0])
        for xc in xrgb[1:]:
            norms += torch.linalg.svdvals(xc)
        norms = norms.sum(-1).reshape(B, -1)

        # Keep patch with max nuclear norm
        idxs = norms.argmax(dim=1).view(-1, 1)
        xrgb = [xc.reshape(B, -1, size, size) for xc in xrgb]
        xrgb = [xc[torch.arange(B).view(-1, 1), idxs].view(B, size, size) for xc in xrgb]

        # Index computation
        off = (idxs % (W // size)).long() * size
        off += (idxs // (W // size)).long() * W * size
        idxs = torch.arange(0, size**2, device=x.device).view(1, -1).repeat(B, 1) + off

        off = torch.arange(0, size, device=x.device).view(-1, 1).repeat(1, size)
        off = off * W - off * size
        idxs += off.view(1, -1)

        # Singular vector product
        pert = torch.zeros_like(x)
        for i, xc in enumerate(xrgb):
            U, _, V = torch.linalg.svd(xc)
            U = U[:, :, 0].view(B, size, 1)
            V = V.transpose(-2, -1)[:, :, 0].view(B, size, 1)
            pert_gr = torch.bmm(U, V.transpose(-2, -1)).reshape(B, size * size)
            idx = torch.arange(B).view(-1, 1)
            pert_tmp = pert[:, i, :, :].view(B, -1)
            pert_tmp[idx, idxs] = pert_gr *eps
            pert[:, i, :, :] = pert_tmp.view(B, H, W)

        # Convert back to HWC
        pert = pert.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        return pert
    

    def __lineSearchUniversal(self, delta,s,batch_size,steps=5):
        '''
        Line search for universal perturbation.
        Applies to multiple batches internally.
        '''
        a = torch.tensor(0.0, device=self.device)
        b = torch.tensor(1.0, device=self.device)
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr

        sx = s - delta

        
    

        def eval_loss(gamma,batch_size):
            loss = 0.0
            count =0

            for BATCH in batch(self.data,batch_size):
                for i, frame in enumerate(BATCH):

                    input = torch.tensor(frame,device=self.device).permute(2,0,1)
                    input = input/255.0
                    adv_e = (input + (delta+gamma*sx).permute(2,0,1)).clamp(0,1)
                    predictions = self.model([input])
                    masks = predictions[0]['masks'][(predictions[0]['scores']>= self.THRESHOLD) & (predictions[0]['labels']==3)]


                    if masks.size(0)==0:
                        print('skip')
                        continue

                    masks = masks.sum(dim=0).squeeze(0).clamp(0,1)
                    adv_pred = self.model([adv_e])
                   
                
                    if adv_pred[0]['scores'].size(0)==0 or adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].size(0)==0:
                        loss_fg,loss_bg= 0,0
                        confl = 0
                       
                        
                    else:
                        masks_adv = adv_pred[0]['masks'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)].sum(dim=0).clamp(0,1)[0,:,:]
                        loss_fg,loss_bg = get_CE_Losses(masks_adv,masks) 
                        confl = conf_loss(adv_pred[0]['scores'][(adv_pred[0]['scores']>= self.THRESHOLD) & (adv_pred[0]['labels'] == 3)])
                       
                    loss += -1*loss_bg+ 1*loss_fg + 0.001*confl
                count+=1

            return loss/count
                    

        for _ in range(steps):
            print(_)
            loss1 = eval_loss(c,batch_size)
            loss2 = eval_loss(d,batch_size)
            if loss1 > loss2:
                a = c
            else:
                b = d
            c = b - (b - a) / self.gr
            d = a + (b - a) / self.gr

        return (a + b) / 2