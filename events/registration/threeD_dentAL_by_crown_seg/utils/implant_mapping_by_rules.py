import numpy as np
import cv2
import math
from skimage.measure import regionprops
import torch
from math import ceil, floor
import torch.nn.functional as F
import copy


def get_max_planes(crown):

    maxXoY, z_idx_max = -1, -1
    maxXoZ, y_idx_max = -1, -1
    maxYoZ, x_idx_max = -1, -1

    for z in range(crown.shape[2]):

        new_maxXoY = max(maxXoY, crown[:, :, z].sum())
        if new_maxXoY > maxXoY:
            z_idx_max = z
            maxXoY = new_maxXoY

    for y in range(crown.shape[1]):

        new_maxXoZ = max(maxXoZ, crown[:, y, :].sum())
        if new_maxXoZ > maxXoZ:
            y_idx_max = y
            maxXoZ = new_maxXoZ

    for x in range(crown.shape[0]):

        new_maxYoZ = max(maxYoZ, crown[x, :, :].sum())
        if new_maxYoZ > maxYoZ:
            x_idx_max = x
            maxYoZ = new_maxYoZ

    XoY = crown[:, :, z_idx_max]
    XoZ = crown[:, y_idx_max, :]
    YoZ = crown[x_idx_max, :, :]

    return XoY, XoZ, YoZ


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on the line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def line_calculation(bbox):
    ps1 = (np.array(bbox[0]) + np.array(bbox[1])) / 2, (np.array(bbox[2]) + np.array(bbox[3])) / 2
    ps2 = (np.array(bbox[0]) + np.array(bbox[3])) / 2, (np.array(bbox[1]) + np.array(bbox[2])) / 2

    k1 = abs((ps1[1][1] - ps1[0][1]) / (ps1[1][0] - ps1[0][0]))
    k2 = abs((ps2[1][1] - ps2[0][1]) / (ps2[1][0] - ps2[0][0]))

    return max(k1, k2)


def plane_crossing(XoZ, YoZ, XoY):
    p1 = XoZ.astype(np.int)
    p2 = YoZ.astype(np.int)
    p3 = XoY.astype(np.int)

    pp1 = tuple(zip(np.nonzero(p1)[0].tolist(), np.nonzero(p1)[1].tolist()))
    pp2 = tuple(zip(np.nonzero(p2)[0].tolist(), np.nonzero(p2)[1].tolist()))
    pp3 = tuple(zip(np.nonzero(p3)[0].tolist(), np.nonzero(p3)[1].tolist()))

    bbox1 = cv2.boxPoints(cv2.minAreaRect(np.array(pp1)))
    bbox2 = cv2.boxPoints(cv2.minAreaRect(np.array(pp2)))
    bbox3 = cv2.boxPoints(cv2.minAreaRect(np.array(pp3)))

    k1 = line_calculation(bbox1)
    k2 = line_calculation(bbox2)
    k3 = line_calculation(bbox3)

    if k1 == float(np.array('inf')):

        l_xoz = np.array([0, 0, 1])

    else:
        l_xoz = np.array([math.sqrt(1 / (k1 ** 2 + 1)), 0, math.sqrt(1 / (k1 ** 2 + 1)) * k1])

    if k2 == float(np.array('inf')):

        l_yoz = np.array([0, 0, 1])

    else:
        l_yoz = np.array([0, math.sqrt(1 / (k2 ** 2 + 1)), math.sqrt(1 / (k2 ** 2 + 1)) * k2])

    if k3 == float(np.array('inf')):

        l_xoy = np.array([0, 1, 0])

    else:
        l_xoy = [math.sqrt(1 / (k3 ** 2 + 1)), math.sqrt(1 / (k3 ** 2 + 1)) * k3, 0]

    norm = math.sqrt(l_xoz[0] ** 2 + l_yoz[1] ** 2)

    return np.array([l_xoy[0] * norm, l_xoy[1] * norm, l_xoz[2] + l_yoz[2]]) / (l_xoz[2] + l_yoz[2])


# def point_shift_along_vector(vector, start, shift, vox, side):
#
#         if side == 'up':
#
#             shift = -shift
#
#     x_shift = shift * math.sqrt(vector[0]) / np.linalg.norm(vector)
#     y_shift = shift * math.sqrt(vector[1]) / np.linalg.norm(vector)
#     z_shift = shift / np.linalg.norm(vector)
#
#     x_vox_num = x_shift / vox[0]
#     y_vox_num = y_shift / vox[1]
#     z_vox_num = z_shift / vox[2]
#
#     return start + np.array([x_vox_num, y_vox_num, z_vox_num])


def point_shift_along_vector(vector, start, shift, vox, side):

    sgns = np.sign(vector)
    vector = np.abs(vector)
    x_shift = sgns[0] * vector[0] / np.linalg.norm(vector)
    y_shift = sgns[1] * vector[1] / np.linalg.norm(vector)
    z_shift = sgns[2] / np.linalg.norm(vector)

    return start + np.array([x_shift, y_shift, z_shift]) * shift / vox


def implant_centre_filling(crown, vertex1, vertex2, radius, length, vox, side, matrix_size):

    diff = np.array(vertex2) - np.array(vertex1)
    diff = diff / np.abs(diff[-1])

    implant = np.zeros(matrix_size)

    bound = np.copy(vertex1).astype(np.float64)

    while crown[round(bound[0]), round(bound[1]), round(bound[2])] == 1:

        bound += diff

        if any(bound + diff) < 0 or any(bound + diff) > matrix_size[2]:

            break

    radius = int(radius)

    start = point_shift_along_vector(diff, bound, 3, vox, side)
    end = point_shift_along_vector(diff, start, 11, vox, side)

    num = abs(round(end[-1] - start[-1]))

    pointer = start

    for i in range(num):

        pointer += diff

        for x in range(round(pointer[0]) - radius, round(pointer[0]) + radius):

            for y in range(round(pointer[1]) - radius, round(pointer[1]) + radius):

                if abs(x - pointer[0]) ** 2 + abs(y - pointer[1]) ** 2 <= radius ** 2:

                    implant[x, y, round(pointer[2])] = 1

    return implant


def implant_by_rotation(crown, vertex1, vertex2, radius, length, matrix_size, vox, side):
    """

    :param crown:
    :param vertex1: centroid of crown.
    :param vertex2: centroid of predicted implanting area.
    :param radius: mapped radius according to Xu's implanting rule.
    :param length: mapped radius according to Xu's implanting rule.
    :param matrix_size:
    :param vox:
    :param side:
    :return:
    """
    bound = vertex1

    implant = np.zeros(matrix_size)

    length /= vox
    radius /= vox

    sgn = 1 if side == 'down' else -1

    for z in range(round(length)):

        for x in range(round(bound[0] - radius), round(bound[0] + radius)):

            for y in range(round(bound[1] - radius), round(bound[1] + radius)):

                if abs(x - bound[0]) ** 2 + abs(y - bound[1]) ** 2 <= radius ** 2:

                    implant[x, y, round(bound[2] + sgn * z)] = 1

    origin = np.array(bound)

    vector = np.array(vertex2) - np.array(bound)
    vector /= vector[-1]

    while crown[round(bound[0]), round(bound[1]), round(bound[2])] == 1:

        bound += vector

        if any(bound + vector) < 0 or any(bound + vector) > matrix_size[2]:

            break

    # vertex = point_shift_along_vector(vector, bound, 3, vox, side)

    ori = np.array(vertex2) - np.array(matrix_size) // 2
    ori = ori / np.linalg.norm(ori)

    # debug rotation 后的植体平移
    translate = vertex1 - origin

    affine_implant = affine_transformation(torch.from_numpy(implant[np.newaxis, np.newaxis]), get_rotation_mat(ori, [0, 0, 1]), pure_translation=translate[::-1] / implant.shape[-1] * 2)

    return affine_implant.squeeze().cpu().numpy()


def implant_identification(crown, vox, side, length=8, radius=3):

    XoY, XoZ, YoZ = get_max_planes(crown)
    centroid = regionprops(crown.astype(np.int))[0].centroid
    vector = plane_crossing(XoZ, YoZ, XoY)
    patch_size = crown.shape[0]

    bound = np.array(centroid)

    symbol = -1 if side == 'up' else 1

    while crown[round(bound[0]), round(bound[1]), round(bound[2])] == 1:

        bound += symbol * vector

        if any(bound + vector) < 0 or any(bound + vector) > patch_size:

            break

    vertex = point_shift_along_vector(vector, bound, length, vox, side)
    num = abs(round(vertex[-1] - bound[-1]))

    # direction = np.zeros_like(crown)
    implant = np.zeros_like(crown)
    pointer = bound
    radius = round(radius / math.sqrt(vox[0]))  # voxel size on xoy plane should be the same.

    for i in range(num):

        pointer += symbol * vector

        # pointer = pointer.tolist()
        implant[round(pointer[0]) - radius: round(pointer[0] + radius), round(pointer[1]) - radius: round(pointer[1] + radius), round(pointer[2])] = 1

        # if direction[round(pointer[0]), round(pointer[1]), round(pointer[2])] == 1:
        #
        #     continue
        #
        # else:
        #
        #     direction[round(pointer[0]), round(pointer[1]), round(pointer[2])] = 1

    return vertex, implant


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
def affine_transformation(img, affine, pure_translation=None, mode='bilinear'):
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
    pure_translation = torch.from_numpy(pure_translation[np.newaxis, :, np.newaxis])

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
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=True)

    rot_img = F.grid_sample(input=img, grid=grid, mode=mode)
    _, _, w, h, d = rot_img.shape
    return rot_img


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
    if type(ori1) is list:
        ori1 = np.array(ori1)
    elif type(ori2) is torch.Tensor:
        ori2 = ori2.squeeze().numpy()

    if type(ori2) is list:
        ori2 = np.array(ori2)
    elif type(ori2) is torch.Tensor:
        ori2 = ori2.squeeze().numpy()

    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)

    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)

    return torch.from_numpy(np.flip(mat).copy()).float()