import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import os


class CMRDataset(Dataset):
    def __init__(self, dir, mode='train',):
        self.mode = mode
        self.img2_slice_list = []
        self.spacing_list = []
        self.pat_list = []
        self.weight_list = []
        pat_list = []
        self.img1_list = torch.zeros(1, 3, 256, 256)
        self.lab_list = torch.zeros(1, 1, 256, 256)
        train_path = dir + 'train_set'
        test_path = dir + 'val_set'
        if self.mode == 'train':
            path = train_path
        else:
            path = test_path
        for pat in os.listdir(path):
            pat_list.append(pat)
            print(pat)
            CT = sitk.ReadImage(os.path.join(path, pat) + '/ct.nii.gz')
            oar = sitk.ReadImage(os.path.join(path, pat) + '/oars.nii.gz')
            ptv = sitk.ReadImage(os.path.join(path, pat) + '/ptvs.nii.gz')
            Dose = sitk.ReadImage(os.path.join(path, pat) + '/dose.nii.gz')

            CT = sitk.GetArrayFromImage(CT)
            oar = sitk.GetArrayFromImage(oar)
            ptv = sitk.GetArrayFromImage(ptv)
            Dose = sitk.GetArrayFromImage(Dose) * 72.6 / 80

            tensor_CT = torch.from_numpy(CT).float().unsqueeze(0).permute(1, 0, 2, 3)
            tensor_oar1 = torch.from_numpy(oar).float().unsqueeze(0).permute(1, 0, 2, 3)
            tensor_ptv = torch.from_numpy(ptv).float().unsqueeze(0).permute(1, 0, 2, 3)
            tensor_Dose = torch.from_numpy(Dose).float().unsqueeze(0).permute(1, 0, 2, 3)
            img1 = torch.cat((tensor_CT, tensor_oar1, tensor_ptv), 1)
            self.img1_list = torch.cat((self.img1_list, img1), 0)
            self.lab_list = torch.cat((self.lab_list, tensor_Dose), 0)
        self.img1_list = self.img1_list[1:, :, :, :]
        self.lab_list = self.lab_list[1:, :, :, :]
        self.pat_list = pat_list
        print('load done, length of dataset:', len(self.lab_list))

    def __len__(self):
        return self.img1_list.shape[0]

    def __getitem__(self, idx):
        tensor_image1 = self.img1_list[idx]
        tensor_label = self.lab_list[idx]
        if self.mode == 'train':
            return tensor_image1, tensor_label
        elif self.mode == 'valid':
            return tensor_image1, tensor_label
        else:
            return tensor_image1, tensor_label


if __name__ == '__main__':
    trainset = CMRDataset(dir='./data/',  mode='train')
    print(trainset[10][0].shape)
    print(len(trainset))
