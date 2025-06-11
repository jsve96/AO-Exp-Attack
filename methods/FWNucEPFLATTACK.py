import uuid
from utils import *
from data_config import *
from FWNucPETS import FWnucl
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



def main(args):

    data = load_data(name=args.dataset,source=args.path_dataset,instance=args.instance)

    print('CUDA')
    print(torch.cuda.is_available())
    print('Devices')
    print(torch.cuda.device_count())
    DEVICE = torch.device('cuda:3')
    print(DEVICE)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(DEVICE)
    ##### #run attack
    
    fwnucl = FWnucl(data=data,device=DEVICE,model=model,img_range=(0,1),iters=args.nit,eps=args.eps)

    
    Result = fwnucl.Attack(batch_size=10)

    print(Result.shape)

    try: 
        l = np.float64(fwnucl.iou_loss.detach().cpu().numpy())
    except:
        l = fwnucl.iou_loss


    Metrics = {'dataset':args.dataset,'eps':args.eps,'nit':args.nit,'IOU':l,'IOU_run':fwnucl.IOU_RUN,'nuc_norm_run':fwnucl.Nuc_NORM_RUN}
    Metrics['MAP'] = np.float64(np.sum(np.mean(np.abs(Result),axis=2))/(Result.shape[0]*Result.shape[1]))
    Metrics['AttackRate'] = np.float64(fwnucl.Attack_rate.detach().cpu().numpy())
    Metrics['BoxRatios'] = fwnucl.BoxRatio
    Metrics['BoxRatio'] = fwnucl.BoxRatio[-1]

    perturbation_tensor = torch.tensor(Result)

    print("nuc_norm: {}".format(nuc_nor(perturbation_tensor)))
    nuc_norm = nuc_nor(perturbation_tensor)

    Metrics['nuc_norm'] = np.float64(nuc_norm)


    # #### make new directory for run 

    run_name = uuid.uuid4().hex

    if not os.path.exists(args.save+"/"+run_name):
        os.makedirs(args.save+"/"+run_name)
    
    np.save(args.save+"/"+run_name+"/delta.npy",Result)

    json_path = args.save+"/"+run_name+"/Metrics.json"
    print(Metrics)


    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(Metrics, f, ensure_ascii=False, indent=4,cls=NumpyEncoder)


from itertools import product
if __name__ == "__main__":
    
    Data = ['EPFL_RLC']
    instance = ['cam1','cam2','cam3']
    Path_Data = ["/home/jacob/datasets"]
    Nit = [30]
    eps=[40]


    for dataset,path_dataset,nit,eps,instance in product(Data,Path_Data,Nit,eps,instance):
        args = argparse.Namespace(
            dataset=dataset,
            path_dataset = path_dataset,
            nit = nit,
            eps=eps,
            save='/home/jacob/Repos/python/VideoAttacks/experiments/EPFL/'+instance+"/FWNucl",
            instance = instance
        )
        print(args.dataset)
        main(args) 
