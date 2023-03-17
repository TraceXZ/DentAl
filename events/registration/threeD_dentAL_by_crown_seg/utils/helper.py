import torch
import numpy as np
import torch.nn.functional as F
import nibabel
import os
import pydicom
import SimpleITK as sitk
from pathlib import Path
from stl import mesh
from skimage import measure
from typing import List, Union
from math import floor, ceil


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
        convertNsave(nifti[:, :, slice_], out_dir, slice_)


def save_array_as_nii(arr, name, vox=(1, 1, 1)):
    return nibabel.save(nibabel.Nifti1Image(arr, np.diag([*vox, 1])), name + '.nii')


def read_dcm(dcm_path):

    parent_path = str(Path(dcm_path).parent)
    reader = sitk.ImageFileReader()
    reader.SetFileName(dcm_path)
    reader.ReadImageInformation()
    series_ID = reader.GetMetaData('0020|000e')
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(parent_path, series_ID)
    # adds-on

    img = []
    for idx, slice_id in enumerate(sorted_file_names):

        slice = pydicom.dcmread(slice_id)

        if idx == 0:
            vox = slice.PixelSpacing[0]
            position = slice.ImagePositionPatient
        img.append(slice.pixel_array[:, :, np.newaxis])

    return img, vox, position


def nii2stl(nii):

    verts, faces, normals, values = measure.marching_cubes(nii, 0)

    crown = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):

        crown.vectors[i] = verts[f]

    return crown


def cubic_padding(func):

    def wrapper(*args, **kwargs):
        img = args[0]
        b, _, w, h, d = img.shape
        max_dim = max(w, h, d)

        # cubic padding to avoid unreasonably stretching
        padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                   ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]
        img = F.pad(img, padding)

        return func(*(img, *args[1:]), **kwargs)[:, :, padding[-2]: w + padding[-2], padding[-4]: h + padding[-4],
               padding[-6]: d + padding[-6]]

    return wrapper


@cubic_padding
def affine_transformation(img, affine: Union[List[torch.Tensor], torch.Tensor], pure_translation=None, mode='bilinear'):
    """
    :param img:
    :param affine:
    :param pure_translation:
    :return:
    """
    device = img.device
    b = img.shape[0]

    # add no pure translation
    if pure_translation is None:
        pure_translation = torch.zeros(b, 3, 1).to(device)

    # calculate affine matrices
    affine_mat = torch.eye(3, 3)
    if not isinstance(affine, list):

        affine_mat = affine.squeeze()

    elif len(affine) == 1:

        affine_mat = affine[0]

    else:
        for index in range(len(affine) - 1):
            affine_mat = torch.matmul(affine[index].squeeze().to(device), affine[index + 1].squeeze().to(device))
            affine[index + 1] = affine_mat

    # apply one-step affine transform
    affine_mat = affine_mat.repeat([b, 1, 1]) if len(affine_mat.shape) == 2 else affine_mat
    affine_matrix = torch.cat([affine_mat.to(device), pure_translation], dim=2)
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)

    rot_img = F.grid_sample(input=img, grid=grid, mode=mode)
    _, _, w, h, d = rot_img.shape
    return rot_img


def patching_with_centroid(centroid, patch_size, volume):

    patch_border = list(range(3))

    for dim in range(3):

        start = centroid[dim] - patch_size // 2 if patch_size // 2 < centroid[dim] else 0

        end = start + patch_size

        end = min(end, volume.shape[dim])

        start = end - patch_size

        patch_border[dim] = (start, end)

    return volume[patch_border[0][0]: patch_border[0][1],
           patch_border[1][0]: patch_border[1][1], patch_border[2][0]: patch_border[2][1]]