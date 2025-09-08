import os
import torch
import random
import scipy.ndimage
import numpy as np
from os.path import join
from pathlib import Path
from scipy.stats import multivariate_normal
import torch as th
import nibabel as nib
from scipy.stats import ortho_group
from torch.utils.data import Dataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, apply_transform
from scipy.ndimage import zoom
import torchio as tio
from artefacts.factory import load_class


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

def get_data_loader(mod, path, file_txt, batch_size, num_workers):

    default_kwargs = {"drop_last": False, "batch_size": batch_size, "pin_memory": False, "num_workers": num_workers,
                      "prefetch_factor": 8, "worker_init_fn": seed_worker, }

    default_kwargs["shuffle"] = False
    default_kwargs["num_workers"] = num_workers
    train_transforms1 = get_brats2021_train_transform_abnormalty_train1()
    train_transforms2 = get_brats2021_train_transform_abnormalty_train2()

    'MMCCD only input single modality during testing'
    dataset = Dataset3D(
        data_root=path,
        mode='train',
        input_mod=mod,
        data_list_file=file_txt,
        transforms1=train_transforms1,
        transforms2=train_transforms2)

    print(f"dataset lenght: {len(dataset)}")
    return th.utils.data.DataLoader(dataset, **default_kwargs)

def get_data_loader_test(mod, path, file_txt, batch_size, num_workers, test_label=False, synthetic_anomaly=False, synthetic_artifact_type=None):

    default_kwargs = {"drop_last": False, "batch_size": batch_size, "pin_memory": False, "num_workers": num_workers,
                      "prefetch_factor": 8, "worker_init_fn": seed_worker, }

    default_kwargs["shuffle"] = False
    default_kwargs["num_workers"] = num_workers
    test_transforms = get_brats2021_train_transform_abnormalty_train1()
    'MMCCD only input single modality during testing'
    dataset = Dataset3D(
        data_root=path,
        mode='test',
        input_mod=mod,
        data_list_file=file_txt,
        transforms1=test_transforms,
        test_label=test_label,
        synthetic_anomaly=synthetic_anomaly,
        synthetic_artifact_type=synthetic_artifact_type
    )

    print(f"dataset lenght: {len(dataset)}")
    return th.utils.data.DataLoader(dataset, **default_kwargs)


def get_brats2021_train_transform_abnormalty_train1():
    train_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
        ]
    )
    return train_imtrans

def get_brats2021_train_transform_abnormalty_train2():
    train_imtrans = Compose(
        [
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )
    return train_imtrans


class Dataset3D(Dataset):
    def __init__(self, data_root: str, mode: str, input_mod, data_list_file, transforms1, transforms2=None, transforms3=None, test_label=None, synthetic_anomaly=None, synthetic_artifact_type=None):
        super(Dataset3D, self).__init__()

        assert mode in ['train', 'test'], 'Unknown mode'
        self.mode = mode
        self.input_mod = input_mod
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.transforms3 = transforms3
        self.synthetic_anomaly = synthetic_anomaly
        self.synthetic_artifact_type = synthetic_artifact_type
        if mode == 'train':
            self.case_names_input = []
            self.case_names_input_highfreq = []
            self.case_names_brainmask = []

            if data_list_file is not None:

                with open(data_list_file) as file:
                    for path in file:
                        self.case_names_input.append(os.path.join(data_root, input_mod, path.split('\n')[0]))
                        self.case_names_brainmask.append(
                            os.path.join(data_root, 'brainmask', path.split('\n')[0]))
                        self.case_names_input_highfreq.append(
                            os.path.join(data_root, input_mod + '_highfreq', path.split('\n')[0]))
            else:
                self.case_names_brainmask = []
                self.case_names_input_highfreq = []
                self.case_names_input = self.case_names_input + sorted(
                    list(Path(os.path.join(data_root, input_mod)).iterdir()))
                for case in self.case_names_input:
                    case_name_highfreq = Path(
                        os.path.join(str(case).split('/' + input_mod + '/')[0], input_mod + '_highfreq',
                                     str(case).split('/' + input_mod + '/')[1]))
                    self.case_names_input_highfreq.append(case_name_highfreq)
                    case_names_brainmask = Path(
                        os.path.join(str(case).split('/' + input_mod + '/')[0], 'brainmask',
                                     str(case).split('/' + input_mod + '/')[1]))
                    self.case_names_brainmask.append(case_names_brainmask)


        elif mode == 'test':
            self.case_names_input = []
            self.case_names_input_highfreq = []
            self.case_names_brainmask = []
            self.case_names_seg = []
            self.test_label = test_label

            if data_list_file is not None:
                with open(data_list_file) as file:
                    for path in file:
                        self.case_names_input.append(os.path.join(data_root, input_mod, path.split('\n')[0]))
                        self.case_names_brainmask.append(
                            os.path.join(data_root, 'brainmask', path.split('\n')[0]))
                        self.case_names_input_highfreq.append(
                            os.path.join(data_root, input_mod + '_highfreq', path.split('\n')[0]))
                        if self.test_label is True:
                            self.case_names_seg.append(
                                os.path.join(data_root, 'label', path.split('\n')[0]))

            else:
                self.case_names_brainmask = []
                self.case_names_input_highfreq = []
                self.case_names_input = self.case_names_input + sorted(
                    list(Path(os.path.join(data_root, input_mod)).iterdir()))
                for case in self.case_names_input:
                    case_name_highfreq = Path(
                        os.path.join(str(case).split('/' + input_mod + '/')[0], input_mod + '_highfreq',
                                     str(case).split('/' + input_mod + '/')[1]))
                    self.case_names_input_highfreq.append(case_name_highfreq)
                    case_names_brainmask = Path(
                        os.path.join(str(case).split('/' + input_mod + '/')[0], 'brainmask',
                                     str(case).split('/' + input_mod + '/')[1]))
                    self.case_names_brainmask.append(case_names_brainmask)
                    if self.test_label is True:
                        case_name_seg = Path(
                            os.path.join(str(case).split('/' + input_mod + '/')[0], 'label',
                                         str(case).split('/' + input_mod + '/')[1]))
                        self.case_names_seg.append(case_name_seg)

    def __getitem__(self, index: int) -> tuple:
        name_input = self.case_names_input[index]
        name_input_highfreq = self.case_names_input_highfreq[index]
        name_input_brainmask = self.case_names_brainmask[index]

        img_nib = nib.load(name_input)
        input = (img_nib.get_fdata())
        input = input.astype(np.float32)

        img_nib_highfreq = nib.load(name_input_highfreq)
        input_highfreq = (img_nib_highfreq.get_fdata())
        input_highfreq = input_highfreq.astype(np.float32)
        img_firsttransform = apply_transform(self.transforms1, input, map_items=False)

        brainmask_nib = nib.load(name_input_brainmask)
        brainmask = (brainmask_nib.get_fdata())
        brainmask = brainmask.astype(np.float32)

        if self.mode == 'test' and self.synthetic_anomaly == True:
            if self.synthetic_artifact_type == 'top_chunk':
                non_zero = np.any(brainmask, axis=(0, 2))
                start = np.argmax(non_zero)
                end = start + 30
                input[:, start:end, :] = input.min()
            elif self.synthetic_artifact_type == 'middle_chunk':
                start = 80
                end = 110
                input[:, start:end, :] = input.min()
            elif self.synthetic_artifact_type == 'bias_field':
                input_tensor = torch.from_numpy(input).unsqueeze(0).float()
                input_tensor = input_tensor - input.min()
                input_tensor = tio.ScalarImage(tensor=input_tensor)
                add_bias = tio.RandomBiasField(coefficients=(0.1,0.1))
                mni_bias = add_bias(input_tensor)
                output_tensor = mni_bias.data
                output_array = output_tensor.squeeze(0).numpy()
                input = output_array + input.min()
            
            elif self.synthetic_artifact_type == 'spike':
                input_tensor = torch.from_numpy(input).unsqueeze(0).float()
                input_tensor = tio.ScalarImage(tensor=input_tensor)
                add_spike = tio.Spike(spikes_positions = [[0.45, 0.45, 0.45]], intensity = 4)
                with_spike = add_spike(input_tensor)                                                                       
                output_tensor = with_spike.data
                input = output_tensor.squeeze(0).numpy()
                
            elif self.synthetic_artifact_type == 'ghosting':
                input_tensor = torch.from_numpy(input).unsqueeze(0).float()
                input_tensor = tio.ScalarImage(tensor=input_tensor)
                add_ghosts = tio.Ghosting(
                    num_ghosts=4,
                    axis=1,
                    intensity=2.0,
                    restore=0.01,
                )                      
                with_ghosts = add_ghosts(input_tensor)
                output_tensor = with_ghosts.data
                input = output_tensor.squeeze(0).numpy()

            elif self.synthetic_artifact_type == 'gaussian_noise':
                artefact = load_class('noise')
                corrupted_volume = input
                axis=2

                for slice_idx in range(input.shape[axis]):
                    if axis == 0:
                        slice_2d = corrupted_volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_2d = corrupted_volume[:, slice_idx, :]
                    else:
                        slice_2d = corrupted_volume[:, :, slice_idx]

                    corrupted_slice = artefact.transform(slice_2d, severity=5)

                    # Put back in volume
                    if axis == 0:
                        corrupted_volume[slice_idx, :, :] = corrupted_slice
                    elif axis == 1:
                        corrupted_volume[:, slice_idx, :] = corrupted_slice
                    else:
                        corrupted_volume[:, :, slice_idx] = corrupted_slice

                input = corrupted_volume
            
            elif self.synthetic_artifact_type == 'zipper':
                artefact = load_class('zipper')
                corrupted_volume = input
                axis=2

                for slice_idx in range(input.shape[axis]):
                    if axis == 0:
                        slice_2d = corrupted_volume[slice_idx, :, :]
                    elif axis == 1:
                        slice_2d = corrupted_volume[:, slice_idx, :]
                    else:
                        slice_2d = corrupted_volume[:, :, slice_idx]

                    corrupted_slice = artefact.transform(slice_2d, severity=5)

                    # Put back in volume
                    if axis == 0:
                        corrupted_volume[slice_idx, :, :] = corrupted_slice
                    elif axis == 1:
                        corrupted_volume[:, slice_idx, :] = corrupted_slice
                    else:
                        corrupted_volume[:, :, slice_idx] = corrupted_slice
                input = corrupted_volume

            img_firsttransform = apply_transform(self.transforms1, input, map_items=False)


                

            img = input
            y_input = np.fft.fftshift(np.fft.fftn(img))
            center = (img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2)
            X, Y, Z = np.ogrid[:img.shape[0], :img.shape[1], :img.shape[2]]
            radius = 15
            dist_from_center1 = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
            mask = dist_from_center1 >= radius
            y_masked = mask * y_input

            abs_masked = np.abs(y_masked)
            abs = np.abs(y_input)
            angle = np.angle(y_input)
            abs_ones = np.ones(abs.shape)
            abs_mask_zerotot1 = abs_masked * mask + abs_ones * ~mask
            fft_ = abs_mask_zerotot1 * np.exp((1j) * angle)
            img = np.fft.ifftn(np.fft.ifftshift(fft_))
            input_highfreq = np.real(img)


        img_firsttransform_highfreq = apply_transform(self.transforms1, input_highfreq, map_items=False)
        brainmask_firsttransform = apply_transform(self.transforms1, brainmask, map_items=False)

        if self.mode == 'train':
            ### training mask generation ###
            num = random.randint(0, 3)
            orth_mat = ortho_group.rvs(3)
            lambda1 = random.uniform(0.3, 10)
            lambda2 = random.uniform(0.3, 10)
            lambda3 = random.uniform(0.3, 10)
            diag = [lambda1, lambda2, lambda3]
            cov_mat = np.dot(np.dot(orth_mat, np.diag(diag)), orth_mat.T)

            if num == 0 or num == 1:
                resize_brain = zoom(brainmask, (0.25, 0.25, 0.25), order=0)
            elif num == 2:
                resize_brain = zoom(brainmask, (0.5, 0.5, 0.5), order=0)
            elif num == 3:
                resize_brain = zoom(brainmask, (0.125, 0.125, 0.125), order=0)

            target_area = np.where(resize_brain > 0)
            rand = np.random.randint(target_area[0].shape)

            mean1 = target_area[0][rand][0]
            mean2 = target_area[1][rand][0]
            mean3 = target_area[2][rand][0]
            rv = multivariate_normal([0,0,0], cov_mat)
            mask_small = np.ones((resize_brain.shape[0], resize_brain.shape[1], resize_brain.shape[2]))
            point = rv.rvs(100000)
            point_normalize = point * 1 + [mean1, mean2, mean3]
            x = np.where(point_normalize[:,0] >= resize_brain.shape[0], resize_brain.shape[0]-1, point_normalize[:,0]).astype(int)
            y = np.where(point_normalize[:,1] >= resize_brain.shape[1], resize_brain.shape[1]-1, point_normalize[:,1]).astype(int)
            z = np.where(point_normalize[:,2] >= resize_brain.shape[2], resize_brain.shape[2]-1, point_normalize[:,2]).astype(int)
            mask_small[x,y,z]=0
            if num == 0 or num == 1:
                mask = mask_small.repeat(4, axis=0).repeat(4, axis=1).repeat(4, axis=2)
            elif num == 2:
                mask = mask_small.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
            elif num == 3:
                mask = mask_small.repeat(8, axis=0).repeat(8, axis=1).repeat(8, axis=2)

            if mask.shape[0]<input.shape[0] or mask.shape[1]<input.shape[1] or mask.shape[2]<input.shape[2]:
                pad_sizes = [(abs(input.shape[i] - mask.shape[i]) // 2,
                              abs(input.shape[i] - mask.shape[i]) - abs(input.shape[i] - mask.shape[i]) // 2) for i in
                             range(len(input.shape))]
                mask = np.pad(mask, pad_sizes, 'constant')
            if mask.shape[0]>input.shape[0] or mask.shape[1]>input.shape[1] or mask.shape[2]>input.shape[2]:
                mask = mask[:input.shape[0], :input.shape[1], :input.shape[2]]
            mask_firsttransform = apply_transform(self.transforms1, mask, map_items=False)
            img = np.concatenate(
                (img_firsttransform, img_firsttransform_highfreq, brainmask_firsttransform, mask_firsttransform), axis=0)

            img = apply_transform(self.transforms2, img, map_items=False)

        elif self.mode == 'test' and self.test_label is True:
            name_input_seg = self.case_names_seg[index]
            seg_nib = nib.load(name_input_seg)
            seg = (seg_nib.get_fdata())
            seg = np.where(seg > 0, 1, 0)
            seg = seg.astype(np.float32)
            seg_firsttransform = apply_transform(self.transforms1, seg, map_items=False)
            img = np.concatenate((img_firsttransform, img_firsttransform_highfreq, brainmask_firsttransform, seg_firsttransform), axis=0)
        elif self.mode == 'test' and self.test_label is False:
            img = np.concatenate((img_firsttransform, img_firsttransform_highfreq, brainmask_firsttransform), axis=0)
        return img

    def __len__(self):
        return len(self.case_names_input)

