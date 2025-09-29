import os
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AMOSTrainDataset(Dataset):
    def __init__(self, data_root, modality, n_slices, image_transform=None):
        self.modality = modality.lower()  # ct图像或mri图像
        # assert self.modality in ('ct', 'mr')
        self.n_slices = n_slices  # 一次要取的切片数量
        self.image_transform = image_transform

        self.data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality), 'r')
        self.patient_images = self.data_file[self.modality].shape[0]  # 病人数量
        self.patient_slices = self.data_file[self.modality].shape[1]  # 每个病人的切片数量
        self.slice_nums = self.patient_slices - n_slices + 1  # n_slices取法下每个病人的切片份数
        self.length = self.patient_images * self.slice_nums  # 训练集长度

    def __getitem__(self, index):
        image_index = index // self.slice_nums  # 第几个病人
        slice_index = index % self.slice_nums  # 第几份切片

        image = self.data_file[self.modality][image_index, slice_index:slice_index + self.n_slices]
        if self.image_transform is not None:
            image = self.image_transform(image)
        data_item = {self.modality: image}

        label_name = '%s_seg' % self.modality
        if label_name in self.data_file:
            label = self.data_file[label_name][image_index, slice_index:slice_index + self.n_slices]
            data_item[label_name] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length


class AMOSValidDataset(Dataset):
    def __init__(self, data_root, modality, image_transform=None):
        self.modality = modality.lower()  # ct图像或mri图像
        # assert self.modality in ('ct', 'mr')
        self.image_transform = image_transform

        self.data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality), 'r')
        self.patient_images = self.data_file[self.modality].shape[0]  # 病人数量
        self.patient_slices = self.data_file[self.modality].shape[1]  # 每个病人的切片数量
        self.length = self.patient_images  # 验证集或测试集长度(病人数量)

    def __getitem__(self, index):
        image = self.data_file[self.modality][index]
        if self.image_transform is not None:
            image = self.image_transform(image)
        data_item = {self.modality: image}

        label_name = '%s_seg' % self.modality
        if label_name in self.data_file:
            label = self.data_file[label_name][index]
            data_item[label_name] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length


class AMOSTestDataset(Dataset):
    def __init__(self, data_root, modality, image_transform=None):
        self.modality = modality.lower()  # ct图像或mri图像
        # assert self.modality in ('ct', 'mri')
        self.image_transform = image_transform

        self.data_file = h5py.File(os.path.join(data_root, "unpaired_%s.h5" % self.modality), 'r')
        self.patient_images = self.data_file[self.modality].shape[0]  # 病人数量
        self.patient_slices = self.data_file[self.modality].shape[1]  # 每个病人的切片数量
        self.length = self.patient_images  # 验证集或测试集长度(病人数量)

    def __getitem__(self, index):
        image = self.data_file[self.modality][index]
        if self.image_transform is not None:
            image = self.image_transform(image)
        data_item = {self.modality: image}

        label_name = '%s_seg' % self.modality
        if label_name in self.data_file:
            label = self.data_file[label_name][index]
            data_item[label_name] = label.astype(np.int64)

        return data_item

    def __len__(self):
        return self.length

# # 加载数据集
# dataset = AMOSTrainDataset(
#     data_root="../experiment_datas/test_amos/train", modality='ct', n_slices=3, sample_nums=3, start_channels=32
# )
# print('dataset:', dataset)
# print('dataset.modality:', dataset.modality)
# print('dataset.n_slices:', dataset.n_slices)
# print('dataset.image_transform:', dataset.image_transform)
# print('dataset.scale:', dataset.scale)
# print('dataset.embed_channels:', dataset.embed_channels)
# print('dataset.data_file:', dataset.data_file)
# print('dataset.patient_images:', dataset.patient_images)
# print('dataset.patient_slices:', dataset.patient_slices)
# print('dataset.slice_nums:', dataset.slice_nums)
# print('dataset.length:', dataset.length)
# print('--------------------')
# iter_datas = iter(dataset)
# print('iter_datas:', iter_datas)
# datas = next(iter_datas)
# # print('datas:', datas)
# print('datas.keys():', datas.keys())
# # print('datas.values():', datas.values())
# print('--------------------')
# image = datas[dataset.modality]
# # print('image:', image)
# print('image.shape:', image.shape)
# print('image.dtype:', image.dtype)
# print('image.min():', image.min())
# print('image.max():', image.max())
# # print('np.unique(image):', np.unique(image))
# print('len(np.unique(image)):', len(np.unique(image)))
# print('--------------------')
# label = datas['%s_seg' % dataset.modality]
# # print('label:', label)
# print('label.shape:', label.shape)
# print('label.dtype:', label.dtype)
# print('label.min():', label.min())
# print('label.max():', label.max())
# print('np.unique(label):', np.unique(label))
# print('len(np.unique(label)):', len(np.unique(label)))
# print('--------------------')
# position = datas['%s_position' % dataset.modality]
# # print('position:', position)
# print('position.shape:', position.shape)
# print('position.dtype:', position.dtype)
# print('position.min():', position.min())
# print('position.max():', position.max())
# print('np.unique(position):', np.unique(position))
# print('len(np.unique(position)):', len(np.unique(position)))
# print('--------------------')
# print("slice1:", position[:, 0].min(), position[:, 0].max())
# print("slice2:", position[:, 1].min(), position[:, 1].max())
# print("slice3:", position[:, 2].min(), position[:, 2].max())
# print('--------------------')
# # 加载数据加载器
# dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=True)
# print('dataloader:', dataloader)
# print('--------------------')
# iter_dataloaders = iter(dataloader)
# print('iter_dataloaders:', iter_dataloaders)
# dataloaders = next(iter_dataloaders)
# # print('dataloaders:', dataloaders)
# print('dataloaders.keys():', dataloaders.keys())
# # print('dataloaders.values():', dataloaders.values())
# print('--------------------')
# images = dataloaders[dataset.modality]
# # print('images:', images)
# print('images.shape:', images.shape)
# print('images.dtype:', images.dtype)
# print('images.min():', images.min())
# print('images.max():', images.max())
# # print('np.unique(images):', np.unique(images))
# print('len(np.unique(images)):', len(np.unique(images)))
# print('--------------------')
# labels = dataloaders['%s_seg' % dataset.modality]
# # print('labels:', labels)
# print('labels.shape:', labels.shape)
# print('labels.dtype:', labels.dtype)
# print('labels.min():', labels.min())
# print('labels.max():', labels.max())
# print('np.unique(labels):', np.unique(labels))
# print('len(np.unique(labels)):', len(np.unique(labels)))
# print('--------------------')
# position = dataloaders['%s_position' % dataset.modality]
# # print('position:', position)
# print('position.shape:', position.shape)
# print('position.dtype:', position.dtype)
# print('position.min():', position.min())
# print('position.max():', position.max())
# print('np.unique(position):', np.unique(position))
# print('len(np.unique(position)):', len(np.unique(position)))
# print('--------------------')
# print("batch1-slice1:", position[0, :, 0].min(), position[0, :, 0].max())
# print("batch1-slice2:", position[0, :, 1].min(), position[0, :, 1].max())
# print("batch1-slice3:", position[0, :, 2].min(), position[0, :, 2].max())
# print("batch2-slice1:", position[1, :, 0].min(), position[1, :, 0].max())
# print("batch2-slice2:", position[1, :, 1].min(), position[1, :, 1].max())
# print("batch2-slice3:", position[1, :, 2].min(), position[1, :, 2].max())
# print('--------------------')

# # 加载数据集
# dataset = AMOSTestDataset(
#     data_root="../experiment_datas/test_amos/test", modality='ct', sample_nums=3, start_channels=32
# )
# print('dataset:', dataset)
# print('dataset.modality:', dataset.modality)
# print('dataset.image_transform:', dataset.image_transform)
# print('dataset.scale:', dataset.scale)
# print('dataset.embed_channels:', dataset.embed_channels)
# print('dataset.data_file:', dataset.data_file)
# print('dataset.patient_images:', dataset.patient_images)
# print('dataset.patient_slices:', dataset.patient_slices)
# print('dataset.length:', dataset.length)
# print('--------------------')
# iter_datas = iter(dataset)
# print('iter_datas:', iter_datas)
# datas = next(iter_datas)
# # print('datas:', datas)
# print('datas.keys():', datas.keys())
# # print('datas.values():', datas.values())
# print('--------------------')
# image = datas[dataset.modality]
# # print('image:', image)
# print('image.shape:', image.shape)
# print('image.dtype:', image.dtype)
# print('image.min():', image.min())
# print('image.max():', image.max())
# # print('np.unique(image):', np.unique(image))
# print('len(np.unique(image)):', len(np.unique(image)))
# print('--------------------')
# label = datas['%s_seg' % dataset.modality]
# # print('label:', label)
# print('label.shape:', label.shape)
# print('label.dtype:', label.dtype)
# print('label.min():', label.min())
# print('label.max():', label.max())
# print('np.unique(label):', np.unique(label))
# print('len(np.unique(label)):', len(np.unique(label)))
# print('--------------------')
# position = datas['%s_position' % dataset.modality]
# # print('position:', position)
# print('position.shape:', position.shape)
# print('position.dtype:', position.dtype)
# print('position.min():', position.min())
# print('position.max():', position.max())
# print('np.unique(position):', np.unique(position))
# print('len(np.unique(position)):', len(np.unique(position)))
# print('--------------------')
# print("slice-3:", position[:, -3].min(), position[:, -3].max())
# print("slice-2:", position[:, -2].min(), position[:, -2].max())
# print("slice-1:", position[:, -1].min(), position[:, -1].max())
# print('--------------------')
