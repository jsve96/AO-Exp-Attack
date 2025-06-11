# AO-Exp-Attack
## Framework
Video-based object detection is essential in safety-critical systems such as autonomous driving and surveillance. While modern deep learning-based object detectors have achieved remarkable accuracy, they remain vulnerable to adversarial attacks — particularly universal perturbations that generalize across multiple inputs.

This project presents a novel, minimally distorted **universal adversarial attack** designed specifically for video object detection.
<p align="center">
  <img width="700" src="UAP_Framework.svg">
</p>

### Universal adversarial perturbations (UAP)

<p align="center">
  <img width="700" src="Attack.gif">
</p>

### Getting Started

Tested on Linux Ubuntu Ubuntu 22.04.5 LTS with python 3.10.12. 

First clone repository:
```bash
git clone https://github.com/jsve96/AO-Exp-Attack.git
```
Set up virtual environment:
```bash
python -m venv UAT
source ~/path/to/AO-Exp/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download dataset


### Usage
If you want to run AO-Exp Attack on the VTM dataset then navigate to Repository and run:
```bash
python3 methods/AOEXPAttackVTMBATCH.py
```

### Hyperparameter
| Parameter       | Description                                                                 | Value / Condition            |
|----------------|-----------------------------------------------------------------------------|------------------------------|
| `dataset`       | Name of the dataset                                                        | `"VTM-Data"`                 |
| `path_dataset`  | Path to the dataset (from `DATA_PATH`, a list or iterable)                 | e.g. `DATA_PATH[0]`          |
| `instance`      | Video instance name                                                        | `"Video1"` to `"Video15"`    |
| `nit`           | Number of iterations                                                       | `50` if `k == 'top1'`, else `100` |
| `l1`            | Nuclear norm regularization                                               | `0.1`                        |
| `l2`            | Frobenius regularization                                               | `0.001` if `k == 'top1'`, else `0.0002` |
| `save`          | Path to save experiment outputs                                             | `REPO_PATH/experiments/VTM-Data/<instance>/AO-Exp` |
| `k`             | Attack mode: top-1 targeting or scalar multiplier                          | `'top1'` or `1.0`            |
| `DEVICE`        | Device used for computation                                                 | `torch.device('cuda:0')`     |

You can change all hyperparameter in /methods/AOEXPAttackVTMBATCH.py by modifying 

```python3
if __name__ == "__main__":
    from itertools import product
   
    Data = ['VTM-Data']
    instance = ['Video'+ str(i) for i in range(1,16)]
    Path_Data = DATA_PATH
    k = [1.0, 'top1']
    L1 = [0.1]
    DEVICE = torch.device('cuda:0')

    

    for dataset,path_dataset,l1,instance,k in product(Data,Path_Data,L1,instance,k):
        args = argparse.Namespace(
            dataset=dataset,
            path_dataset = path_dataset,
            nit = 50 if k == 'top1' else 100,
            l1=l1,
            l2=0.001 if k =='top1' else 0.0002,
            save=REPO_PATH+'/experiments/VTM-Data/'+instance+"/AO-Exp",
            instance = instance,
            k=k,
            DEVICE = DEVICE
        )
        print(args.dataset)
        main(args) 

```
