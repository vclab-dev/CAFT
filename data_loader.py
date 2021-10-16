from torchvision import datasets, transforms
import torch
import os
from data_list import ImageList_idx
from torch.utils.data import DataLoader, Subset



def data_load_target_idx(batch_size, transform,shfl = True):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = batch_size
    root = 'dataset/visda-2017'
    tar_dict = {'V': 'validation.txt', 'T': 'train.txt'}
    num_classes = 12

    txt_pth_tgt = os.path.join(root, tar_dict['V'])
    txt_tar = open(txt_pth_tgt).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=transform)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=shfl, drop_last=True)
    dset_loaders["test"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=shfl, drop_last=False)


    txt_pth_src = os.path.join(root, tar_dict['T'])
    txt_src = open(txt_pth_src).readlines()

    dsets["source"] = ImageList_idx(txt_src, transform=transform)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, drop_last=True)
    
    source_size, target_size = len(dsets["source"]), len(dsets["target"])
    print(f'Source Data Size:', source_size)
    print(f'Target Data Size:', target_size)

    return dset_loaders, dsets


def visda_load_training_idx(batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    
    # data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loaders, dset_loaders = data_load_target_idx(batch_size, transform)

    # train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loaders, dset_loaders

def visda_load_testing_idx(batch_size, shfl=False):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    
    # data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loaders, dset_loaders = data_load_target_idx(batch_size, transform)

    # train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loaders, dset_loaders



def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
def load_testing_without_shuffle(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader