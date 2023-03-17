import numpy as np
from scipy.ndimage import zoom
from skimage.measure import regionprops


class RegionProposal:

    def __init__(self, arr, scaling_factor=2):

        arr = arr.astype(np.int16)
        arr_downsample = zoom(arr, [1/scaling_factor, 1/scaling_factor, 1/scaling_factor])
        region = regionprops(arr_downsample)[0]

        self.bbx = tuple((ele * scaling_factor for ele in region.bbox))
        self.centroid = tuple((int(ele * scaling_factor) for ele in region.centroid))
        self.area = region.area * scaling_factor
        self.label = region.label


def get_region_bbx(arr):

    return RegionProposal(arr).bbx


def get_region_centroid(arr):

    return RegionProposal(arr).centroid


def patching_with_centroid(centroid, patch_size, volume, scale=1):

    patch_border = list(range(3))

    volume = zoom(volume, (1/scale, 1/scale, 1/scale))

    for dim in range(3):

        start = centroid[dim] - patch_size // 2 if patch_size // 2 < centroid[dim] else 0

        end = start + patch_size

        end = min(end, volume.shape[dim])

        start = end - patch_size

        patch_border[dim] = (start, end)

    return volume[patch_border[0][0]: patch_border[0][1],
           patch_border[1][0]: patch_border[1][1], patch_border[2][0]: patch_border[2][1]]
