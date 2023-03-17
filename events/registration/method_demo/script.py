import numpy as np
import torch
from utils.data import torch_from_nib_path, save_tensor_as_nii
import cv2
import math

no = 4
crown = torch_from_nib_path('/Volumes/Samsung_T5/projects/dentAL/episodes/episode7/crown_' + str(no) + '.nii',
                            torch.device('cpu')).squeeze()

crown[crown > 1] = 1
crown[(1 - crown) > 0.5] = 0

mask = crown != 0
crown[crown != 0] = 1

from skimage.measure import regionprops

centroid = regionprops(crown.numpy().astype(np.int))[0].centroid

maxXoY, z_idx_max = -1, -1
maxXoZ, y_idx_max = -1, -1
maxYoZ, x_idx_max = -1, -1

for z in range(crown.shape[2]):

    new_maxXoY = max(maxXoY, crown[:, :, z].sum())
    if new_maxXoY > maxXoY:
        z_idx_max = z
        maxXoY = new_maxXoY
        new_maxXoY = -1

for y in range(crown.shape[1]):

    new_maxXoZ = max(maxXoZ, crown[:, y, :].sum())
    if new_maxXoZ > maxXoZ:
        y_idx_max = y
        maxXoZ = new_maxXoZ
        new_maxXoZ = -1

for x in range(crown.shape[0]):

    new_maxYoZ = max(maxYoZ, crown[x, :, :].sum())
    if new_maxYoZ > maxYoZ:
        x_idx_max = x
        maxYoZ = new_maxYoZ
        new_maxYoZ = -1

XoY = crown[:, :, z_idx_max]
XoZ = crown[:, y_idx_max, :]
YoZ = crown[x_idx_max, :, :]


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
    
    p1 = XoZ.numpy().astype(np.int)
    p2 = YoZ.numpy().astype(np.int)
    p3 = XoY.numpy().astype(np.int)

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

    return [l_xoy[0] * norm, l_xoy[1] * norm, l_xoz[2] + l_yoz[2]]


vector = plane_crossing(XoZ, YoZ, XoY)


def point_shift_along_vector(vector, centroid, shift, vox=0.4, side='down'):

    delta_z = vector[2] / math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    if side == 'up':

        shift = -shift

    if vector[0] == 0:

        delta_x = 0
    else:
        delta_x = math.sqrt(1 - delta_z ** 2) * vector[0] / math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    if vector[1] == 0:

        delta_y = 0
    else:
        delta_y = math.sqrt(1 - delta_z ** 2) * vector[1] / math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    shift /= vox
    cx, cy, cz = centroid

    return cx + delta_x * shift, cy + delta_y * shift, cz + delta_z * shift


vertex1 = point_shift_along_vector(vector, centroid, 5, vox=0.15, side='up')
vertex2 = point_shift_along_vector(vector, centroid, 12, vox=0.15, side='up')

seg = np.array(vertex2) - np.array(vertex1)
num = np.linalg.norm(vertex2 - vertex1) / vox
points = []
direction = torch.zeros_like(crown)
implant = torch.zeros_like(crown)

for i in range(num):

    point = np.array(vertex1) + seg / num * i
    point = point.tolist()
    points.append(point)
    implant[round(point[0]) - 3: round(point[0] + 3), round(point[1]) - 3: round(point[1] + 3), round(point[2])] = 1
    if direction[round(point[0]), round(point[1]), round(point[2])] == 1:
        continue
    else:
        direction[round(point[0]), round(point[1]), round(point[2])] = 1

save_tensor_as_nii(implant, 'im_' + str(no))
