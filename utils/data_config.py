import zipfile
import os
from utils import *
import pathlib
import matplotlib.image as mpimg

print('cwd: {}'.format(os.getcwd()))
current_file_path = os.path.dirname(__file__)

REPO_PATH = os.path.dirname(current_file_path)

print('repo: {}'.format(REPO_PATH))

DATA_PATH = ["/home/jacob/datasets"]

def load_data(name='BMC',source="",instance=""):

    path_datasets = pathlib.Path(source)


    if name == 'VTM-Data':
        vtm_path = path_datasets.joinpath(name)
        exp_path = vtm_path.joinpath(instance)
        names = os.listdir(exp_path)
        names = sorted(names, key=lambda x: int(x[:-4]))
        DATA = []
        for n in names:
            DATA.append(mpimg.imread(exp_path.joinpath(n)))


        data = [DATA[i][0::2,0::2] for i in range(0,len(DATA))]
        print(data[0].shape)

    elif name == "PETS09":
        #instance ---> View_00x
        pet_path = path_datasets.joinpath(name)
        exp_path = pet_path.joinpath(instance)
        names = os.listdir(exp_path)
        data = []
        for n in names[::3]:
            data.append(mpimg.imread(exp_path.joinpath(n)))

    elif name == 'EPFL_RLC':

        epfl_path = path_datasets.joinpath(name)
        exp_path = epfl_path.joinpath(instance)
        names = os.listdir(exp_path)
        data = []
        if instance =="cam0":
            for n in names[:200]:
                data.append(mpimg.imread(exp_path.joinpath(n)))
        else:
            for n in names[:200]:
                data.append(mpimg.imread(exp_path.joinpath(n))[30:,50:-50,:])

    
    return data