import os
import glob
import random
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
    # for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet10_hdf5_2048', '%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def add_noisedata(pointcloud, num_noise=10, sigma=0.2):
    N, _ = pointcloud.shape
    noise = np.clip(sigma * np.random.randn(num_noise, 3), -1, 1)
    idx = np.random.randint(0, N, num_noise)
    pointcloud[idx, :] = pointcloud[idx, :] + noise
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        # self.random = range(2048)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        # if self.partition == 'test':
        #     pointcloud = add_noisedata(pointcloud, num_noise=10)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def add_noisedata(pointcloud, num_noise=10, sigma=0.2):
    N, _ = pointcloud.shape
    noise = np.clip(sigma * np.random.randn(num_noise, 3), -1, 1)
    idx = np.random.randint(0, N, num_noise)
    pointcloud[idx, :] = pointcloud[idx, :] + noise
    return pointcloud

class ModelNet_O(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_points):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.root_dir = root_dir
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))
            self.filepaths.extend(all_files)
        self.num_points = num_points

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3] # ModelNet_O/car/test/car_0285_003.xyz
        class_id = np.array(self.classnames.index(class_name)).astype('int64')
        point_set = np.loadtxt(self.filepaths[idx]).astype('float32')
        point_set = point_set[:self.num_points]
        partition = path.split('/')[-2] # ModelNet_O/car/test/car_0285_003.xyz
        if partition == 'train':
            np.random.shuffle(point_set)
        # if self.partition == 'test':
        #     pointcloud = add_noisedata(pointcloud, num_noise=10)
        return point_set, class_id