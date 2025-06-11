# msk = torch.stack((1-mask,mask),2)
# msk.view(-1,2)
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def CE(p,y,idx):
    p_ = torch.index_select(p,0,idx)
    y_ = torch.index_select(y,0,idx)
    return F.cross_entropy(p_,y_.long())
    #return F.nll_loss(p_,y_.long())

def get_CE_Losses(adv_mask,mask):
    #mask for pixels in bb in (0,1) --> probability that pixel is class 0 or 1
    adv_msk = torch.stack((1-adv_mask,adv_mask),2).view(-1,2)
    #background/foreground labels (0 or 1)
    target = ((mask>0)*1).squeeze(0).flatten()

    fg = target.data.eq(1).nonzero().squeeze()
    bg = target.data.eq(0).nonzero().squeeze()

    loss_fg = CE(adv_msk,target,fg)
    loss_bg = CE(adv_msk,target,bg)

    return loss_fg, loss_bg


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() 

def conf_loss(conf):
    return torch.sum(conf*100)

def nuclear_norm(x):
    return torch.mean(torch.norm(x.squeeze(0), p='nuc', dim=(1, 2)))


def frobenius_norm(x):
    return torch.mean(torch.norm(x.squeeze(0), p='fro', dim=(1, 2))**2)

def l1_norm(x):
    return torch.mean(torch.norm(x.squeeze(0), p=1, dim=(1, 2)))


import zipfile 
import cv2
import numpy as np

def read_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_names = zip_file.namelist()
        print(file_names)
        png_files = [f for f in file_names if f.startswith("Video_006/Video_006/Img_") and f.lower().endswith(".bmp")]
        png_files = np.sort(png_files)
        #print(png_files)
        frames = []
        for png in png_files:
            with zip_file.open(png) as file:
                image_bytes = file.read()
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR_RGB)
                frames.append(image)
    return frames



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def nuc_nor(X):
    #shape (h,w,3)
    s = 0
    for c in range(3):
        singular_vals = np.linalg.svd(X[:,:,c].detach().cpu().numpy(),full_matrices=False,compute_uv=False)
        s += singular_vals.sum()
    return s/3



def IOU_Loss(gt_b,adv_b):

    if adv_b.shape[0] == 0:
        return 0
    
    else:
        n = gt_b.shape[0]
        m = adv_b.shape[0]
        ### [n,4], [m,4] ---> [n,m] matrix of iou scores
        IOU_Matrix = box_iou(gt_b,adv_b)

        return IOU_Matrix.sum()#/(n*m)
    
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
    