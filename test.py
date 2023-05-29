#%%
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
from numpy import asarray
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import torch
import torchvision.transforms as T


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            # data=asarray(img)
            # data = np.rollaxis(data,1,4)
            # plt.imshow(data.squeeze(0))
            # plt.show()
            print(img.shape)
            # data=img.squeeze(0)
            
            # trans=T.ToPILImage()
            # data=trans(data)
            # print(data.shape)
            img=img.to(device)
            
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            plt.show()
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root=r"C:\Users\CVPR\source\repos\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\images"
    gt_dmap_root=r"C:\Users\CVPR\source\repos\ShanghaiTech_Crowd_Counting_Dataset\part_B_final\test_data\ground_truth"
    model_param_path=r'D:\Soumi\Soumi DI\DmapVehicleDetection\checkpoints\CSRNet-Epochs-10_BatchSize-64_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
    cal_mae(img_root,gt_dmap_root,model_param_path)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,1) 
    