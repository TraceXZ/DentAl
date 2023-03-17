import torch
import numpy as np
import torch.nn.functional as F
import nibabel
import os
import pydicom
import SimpleITK as sitk
from pathlib import Path


def skew(vector):
    """
    skew-symmetric operator for rotation matrix generation
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def get_rotation_mat(ori1, ori2):
    """
    generating pythonic style rotation matrix
    :param ori1: your current orientation
    :param ori2: orientation to be rotated
    :return: pythonic rotation matrix.
    """
    ori1 = ori1.squeeze().numpy()
    ori2 = ori2.squeeze().numpy()
    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)
    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)
    return torch.from_numpy(np.flip(mat).copy()).float()


def implant_augmentation(implant, device):

    implant = implant.unsqueeze(0)
    orientation = F.normalize(torch.randn([3]), p=2, dim=0)
    rot_mat = get_rotation_mat(orientation, torch.tensor([0, 0, 1]))
    affine_matrix = torch.cat([rot_mat.to(device), torch.zeros([3, 1]).to(device)], dim=1).unsqueeze(0)
    grid = F.affine_grid(affine_matrix, implant.shape, align_corners=False)
    return F.grid_sample(input=implant, grid=grid, mode='bilinear'), orientation.to(device)


def convertNsave(arr, file_dir, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    """

    dicom_file = pydicom.dcmread('GUI/events/registration/method_demo/utils/dcmimage.dcm')
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))


def nii2dcm_1file(nifti, out_dir):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    number_slices = nifti.shape[2]

    for slice_ in range(number_slices):
        convertNsave(nifti[:,:,slice_], out_dir, slice_)


def save_array_as_nii(arr, name):
    return nibabel.save(nibabel.Nifti1Image(arr, np.eye(4)), name + '.nii')


def read_img(in_path):

    in_path = str(Path(in_path[0]).parent)
    file_name = os.listdir(in_path)[0]
    reader = sitk.ImageFileReader()
    reader.SetFileName(in_path + '/' + file_name)
    reader.ReadImageInformation()
    series_ID = reader.GetMetaData('0020|000e')
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(in_path, series_ID)
    dcm_obj = sitk.ReadImage(sorted_file_names)
    voxel_size = dcm_obj.GetSpacing()
    return np.array(sitk.GetArrayFromImage(dcm_obj)).squeeze(), voxel_size


def read_dcm(dcm_path):

    ct_volume, vox_size = read_img(dcm_path)

    return np.array(ct_volume)
