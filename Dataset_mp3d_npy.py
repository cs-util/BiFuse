from __future__ import division

import os
import h5py
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset as TorchDataset
from PIL import Image
from tensorboardX import SummaryWriter
import cv2
from skimage import img_as_ubyte

# Computed from random subset of ImageNet training images. Used since
# sparse-to-dense model use resnet-50 pretrained on ImageNet.
MEANSTD_RGB = {
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225)
}
# Raw output size
OUTPUT_SIZE = (960, 1920) #(512,1024)

def readData(path, mode, shape):
    assert mode in ['color', 'depth']
    if mode == 'color' :
        img = Image.open(path).convert("RGB") #to RGB
        img = np.array(img,np.float32) / 255
        [h, w, c] = img.shape
        if h != shape[0] or w != shape[1]:
            img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_AREA)
    elif mode == 'depth' :
        img = Image.open(path)
        img = np.array(img,np.float32) / 4000
        [h, w] = img.shape
        if h != shape[0] or w != shape[1]:
            img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_NEAREST)
        img = img[:,:,np.newaxis]

    return img

def readDepth_npy(path, shape):
    depth = np.load(path)
    [h, w] = depth.shape
    if h != shape[0] or w != shape[1]:
        depth = cv2.resize(depth, (shape[1], shape[0]), cv2.INTER_NEAREST)
    return depth[:, :, np.newaxis]

class Dataset(TorchDataset): #shape = (960, 1920)
    def __init__(self, data_dir, mode, shape = (512, 1024)):
        #assert mode in ['train', 'val'], 'Invalid dataset mode {}'.format(mode)
        self.data_dir = data_dir
        self.mode = mode
        self.shape = shape
        
        with open(os.path.join(data_dir, '%s.txt'%mode)) as f:
            lst1 = []
            tmp = {}

            for line in  f:
                line = line[:-1]
                aa = line.split(' ')
                if aa[0] not in lst1:
                    lst1.append(aa[0])
                if aa[0] not in tmp:
                    tmp[aa[0]] = []
                tmp[aa[0]].append(aa[1])

        # Obtain all image filename (.h5)
        self.rgb_paths = []
        self.depth_paths = []
        #lst1 = os.listdir(data_dir) # lst1 contain 89 folder


        for i in lst1:
            #lst2 = os.listdir(os.path.join(data_dir,i))
            lst2 = tmp[i]
            for j in lst2:
                lst3 = os.listdir(os.path.join(os.path.join(data_dir,i),j)) #contain depth.png and color.jpg ['depth.png', 'color.jpg']
                for k in lst3:
                    assert 'color.jpg' in lst3 and 'depth.npy' in lst3
                    #print(os.path.join(os.path.join(os.path.join(data_dir,i),j),k))
                    if k == 'color.jpg' :
                        self.rgb_paths.append(os.path.join(os.path.join(os.path.join(data_dir,i),j),k))
                    if k == 'depth.npy' :
                        self.depth_paths.append(os.path.join(os.path.join(os.path.join(data_dir,i),j),k))

        # Define transforms
        self.transforms = dict()
        self.transforms['raw_rgb'] = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transforms['rgb'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEANSTD_RGB['mean'], MEANSTD_RGB['std'])
        ])
        self.transforms['depth'] = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # Load raw data
        #print (len(self.rgb_paths))
        #print (len(self.depth_paths))
        rgb = readData(self.rgb_paths[index], 'color', self.shape) #(960, 1920, 3)
        #depth = readData(self.depth_paths[index], 'depth', self.shape) #(960, 1920, 1)
        depth = readDepth_npy(self.depth_paths[index], self.shape)

        #rgb = img_as_ubyte(rgb) #float32 to uint8
        #depth = depth.astype('<f4') #float32 to float32 QQ
        '''
        print(rgb.shape)
        print(type(rgb))
        print(rgb.dtype)
        print(depth.shape)
        print(type(depth))
        print(depth.dtype)        
        print(self.transforms)
        '''
        # Perform data transform and sample sparse depth
        raw_rgb_tensor = self.transforms['raw_rgb'](rgb) #torch.Size([3, 240, 480])
        rgb_tensor = self.transforms['rgb'](rgb.copy()) #torch.Size([3, 240, 480])
        depth_tensor = self.transforms['depth'](depth) #torch.Size([1, 240, 480])
        
        #depth_tensor = torch.from_numpy(depth)
        
        #raw_rgb_tensor = torch.from_numpy(rgb)
        #rgb_tensor = torch.from_numpy(rgb)

        # Pack up data
        return {
            'raw_rgb': raw_rgb_tensor,
            'rgb': rgb_tensor,
            'depth': depth_tensor
        }

    def __len__(self):
        return len(self.rgb_paths)


def test_basic():
    """ Examine basic utility of this dataset code """
    import pdb
    import matplotlib.pyplot as plt
    #from sampler import UniformSamplerByNumber

    # Setup dataset
    mode = ['train', 'test'][0]
    #sampler = UniformSamplerByNumber(n_samples=100, max_depth=30)
    dataset = Dataset(data_dir='/media/yuhsuan/Elements/panorama_train',
                      mode=mode)

    # Check data
    print('Dataset size: {}'.format(len(dataset)))
    for i, data in enumerate(dataset):
        # unpack data and convert to numpy array
        data_np = {
            'raw_rgb': data['raw_rgb'].numpy().transpose(1, 2, 0), #torch.Size([960, 1920, 3]) torch.Size([512, 1024, 3])
            'rgb': data['rgb'].numpy().transpose(1, 2, 0), #torch.Size([960, 1920, 3])
            'sdepth': data['sdepth'][0].numpy(), #torch.Size([960, 1920, 1])
            'depth': data['depth'][0].numpy() #torch.Size([960, 1920, 1])
        }
        import pdb;pdb.set_trace()
        # visualization
        fig, axes = plt.subplots(2, 2)
        axes = axes.ravel()
        for j, (key, val) in enumerate(data_np.items()):
            axes[j].imshow(val)
            axes[j].set_title(key)
        plt.show()
        plt.cla; plt.clf; plt.close()


if __name__ == '__main__':
    test_basic()
