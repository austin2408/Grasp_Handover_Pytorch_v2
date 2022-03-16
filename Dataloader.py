import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
import numpy as np


image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])

class parallel_jaw_based_grasping_dataset(Dataset):
    name = []
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        if self.mode == 'train':
            f = open(self.data_dir+"/train.txt", "r")
        else:
            f = open(self.data_dir+"/test.txt", "r")
        for i, line in enumerate(f):
              self.name.append(line.replace("\n", ""))

        print(len(self.name))
    def __len__(self):
        return len(self.name)
    def __getitem__(self, idx):
        idx_name = self.name[idx]
        color_img = cv2.imread(self.data_dir+"/color/color_"+idx_name+'.jpg')
        color_img = color_img[:,:,[2,1,0]]
        depth_img = np.load(self.data_dir+"/depth/depth_"+idx_name+'.npy')

        label_img = np.load(self.data_dir+"/label/label_"+idx_name+'.npy')

        f = open(self.data_dir+'/idx/id_'+idx_name+'.txt', "r")

        IDX = int(f.readlines()[0])

        # uint8 -> float
        color = (color_img/255.).astype(float)
        # BGR -> RGB and normalize
        color_rgb = np.zeros(color.shape)
        for i in range(3):
            color_rgb[:, :, i] = (color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]
        depth = (depth_img/1000.).astype(float) # to meters
        # D435 depth range
        depth = np.clip(depth, 0.0, 1.2)
        # Duplicate channel and normalize
        depth_3c = np.zeros(color.shape)
        for i in range(3):
            depth_3c[:, :, i] = (depth[:, :]-image_net_mean[i])/image_net_std[i]
        # Unlabeled -> 2; unsuctionable -> 0; suctionable -> 1
        label_tmp = np.round(label_img[IDX]/255.*2.).astype(float)
        label = np.zeros((4,28,28))
        # Set label pixel
        label[IDX] = cv2.resize(label_tmp, (int(28), int(28)))
        label[IDX][label[IDX] != 0.0] = 1.0

        transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        color_tensor = transform(color_rgb).float()
        depth_tensor = transform(depth_3c).float()
        label_tensor = transform(label).float()
        sample = {"color": color_tensor, "depth": depth_tensor, "label": label_tensor, "id": IDX, "origin_color": color_img}
        return sample