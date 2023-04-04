import nibabel
import torch
import numpy as np
from stl import mesh
from skimage import measure, morphology
from skimage.measure import regionprops
from scipy.ndimage import zoom, binary_erosion
from utils import read_dcm
from math import ceil, floor
import torch.nn.functional as F
import warnings
import torch.nn.functional as F
import SimpleITK as sitk


class Entity:

    def __init__(self, data, vox):

        self.data = data
        self.vox = vox
        self.to_ndarray()

    def to_tensor(self):

        if type(self.data) is torch.Tensor:

            return

        elif type(self.data) is np.ndarray:

            self.data = torch.from_numpy(self.data[np.newaxis, np.newaxis])

    def to_ndarray(self):

        if type(self.data) is np.ndarray:

            return

        elif type(self.data) is torch.Tensor:

            self.data = self.data.squeeze().cpu().numpy()

    def save_as_nii(self, name):

        if type(self.data) is torch.Tensor:
            self.data = self.data.squeeze().cpu().numpy()


        data = self.data

        nibabel.save(nibabel.Nifti1Image(data, np.eye(4)), name)

    def save_as_stl(self, name, vox, pos):

        data = self.data

        data = data.transpose(1, 0, 2)

        verts, faces, normals, values = measure.marching_cubes(data, 0)

        data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        for i, f in enumerate(faces):

            data.vectors[i] = verts[f] * np.array([vox, vox, vox]) + np.array(pos)

        data.save(name)

        # smooth

        import trimesh

        stl = trimesh.load_mesh(name)
        smooth = trimesh.smoothing.filter_laplacian(stl, lamb=1, iterations=20, volume_constraint=False)
        # smooth.apply_translation(np.array([4, 10, -9]) * np.array([vox, vox, vox]))
        smooth.export(name)


class SegEntity(Entity):

    def __init__(self, data, vox):
        super(SegEntity, self).__init__(data, vox)

        self.centroid = regionprops(self.data.astype(int))[0].centroid
        self.sum = self.data.sum()
        self.log_book = None


class Implant(SegEntity):

    def __init__(self, data, vox, radius=-1, length=-1):

        super(Implant, self).__init__(data, vox)

        self.radius = radius

        self.length = length

        self.side = None

        self.cls = None

        self.implant = None

    def save_as_stl(self, name, vox, pos):

        super(Implant, self).save_as_stl(name + '.stl', vox, pos)

        self.implant.save_as_stl(name + '_implant.stl', vox, pos)

    def save_as_nii(self, name):

        super(Implant, self).save_as_nii(name)

        self.implant.save_as_nii(name + '_implant')


class Teeth(SegEntity):

    def __init__(self, crown, vox, max_implant_num=1):
        super(Teeth, self).__init__(crown, vox)

        self.implants = []  # implants calculated based on physics.
        self.max_implant_num = max_implant_num

    def teeth_classification(self, idx, cbct, ts=0.2):
        """
        catergory the predicted missing teeth from upper and down, front and back.
        :param idx: the index of implant to be classified.
        :param cbct: oral CBCT.
        :param ts: threshold HU value of the cbct.
        :return:
        """
        cx, cy, cz = self.centroid
        x, y, z = cbct.shape

        implant = self.implants[idx]

        _, _, czz = implant.centroid

        side = 'upper' if czz < cz else 'down'
        implant.side = side

        start = 0

        for slice_no in range(x):

            coronal = cbct[slice_no, :, :]

            if coronal.max() > ts:
                start = slice_no
                break

        if start == 0:
            warnings.warn('牙冠定位失败，请检查输入CBCT文件是否已经预标准化，或含有大量伪影。')

        cx = round(cx)

        classifier = {0: 'front', 1: 'middle'}

        idx = (cx - start) % 20

        cls = 'back' if idx > 1 else classifier[idx]
        implant.cls = cls

        if cls == 'front':

            radius, length = 1.5, 11.5

            radius = radius + 0.15 if side == 'upper' else radius - 0.15

        elif cls == 'middle':

            radius, length = 2.1, 10.0

        elif cls == 'back':

            radius, length = 2.25, 8.0

        else:

            warnings.warn('牙冠定位失败，请检查代码实现')

            radius, length = 2.0, 11.0

        implant.radius = radius
        implant.length = length

    def save_as_nii(self, name):

        super(Teeth, self).save_as_nii(name)

        for idx, implant in enumerate(self.implants):

            implant.save_as_nii(name + '_map' + str(idx))

    def save_as_stl(self, name, vox, pos):

        super(Teeth, self).save_as_stl(name + '.stl', vox, pos)

        for idx, implant in enumerate(self.implants):

            implant.save_as_stl(name + '_map' + str(idx), vox, pos)


class Oral:

    def __init__(self, cbct_path, device, max_crown_num, env_vox=0.4):

        if cbct_path.lower().endswith('nii') or cbct_path.lower().endswith('nii.gz'):

            cbct = nibabel.load(cbct_path).get_fdata()
            data = cbct.get_fdata()
            affine = cbct.affine
            self.oral_vox = affine.diagonal()[0]
            self.oral_position = [affine[i, 3] for i in range(4)]

        else:
            data, vox, pos = read_dcm(cbct_path)
            self.oral_vox = vox
            self.oral_position = pos

        self.device = device

        self.max_crown_num = max_crown_num

        self.env_vox = env_vox

        self.cbct = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device, torch.float)
        self.cbct = self.cbct.permute([0, 1, 3, 4, 2])

        self.cbct = F.interpolate(self.cbct, scale_factor=self.oral_vox / env_vox)

        _, _, w, h, d = self.cbct.shape

        max_dim = max(w, h, d)

        self.padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                        ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]

        self.tooth = []

        self.crown_num = -1

        self.abnormal = 0

        self.PATCH_SZ = 72

        # self.cbct = F.pad(self.cbct, self.padding)

        self.full_size = None

        self.log_book = None

    def pred_teeth(self, model):

        with torch.no_grad():

            pred_crowns = model(self.cbct)

        # padding afterwards to save memory.
        self.cbct = F.pad(self.cbct, self.padding)

        pred_crowns = F.pad(pred_crowns, self.padding)

        self.full_size = self.cbct.shape[2:]

        pred_crowns = (pred_crowns - pred_crowns.min()) / (pred_crowns.max() - pred_crowns.min())

        pred_crowns[pred_crowns < 0.6] = 0
        pred_crowns[pred_crowns > 0.6] = 1

        pred_crowns = pred_crowns.squeeze().cpu().numpy()

        labeled_crowns, rank = Oral.label_and_sort(pred_crowns)

        self.crown_num = min(labeled_crowns.max(), self.max_crown_num)

        for idx in range(self.crown_num):

            crown = np.zeros_like(pred_crowns)

            crown[labeled_crowns == 1 + rank[idx][1]] = 1

            # abnormal detection
            if crown.sum() < 200:
                self.abnormal += 1

                continue
            teeth = Teeth(crown, self.env_vox)
            self.tooth.append(teeth)

        self.log_book = """\n算法共检测到{}个潜在的牙冠摆放位置，排除不可能项后保存了前{}个最有可能的位置。""".format(labeled_crowns.max(), len(self.tooth))

    def pred_implant(self, model):

        cbct = self.cbct.squeeze().cpu().numpy()

        for teeth_no, tooth in enumerate(self.tooth):

            centroid = tooth.centroid
            cropped_cbct = Oral.patching_with_centroid(centroid, self.PATCH_SZ, cbct)

            with torch.no_grad():
                pred_implants = model(torch.from_numpy(cropped_cbct[np.newaxis, np.newaxis]).to(self.device, torch.float))

            pred_implants = (pred_implants - pred_implants.min()) / (pred_implants.max() - pred_implants.min())

            pred_implants[pred_implants < 0.5] = 0
            pred_implants[pred_implants > 0.5] = 1

            padding = self.calc_padding(centroid)

            pred_implants = torch.nn.functional.pad(pred_implants, padding)

            pred_implants = pred_implants.squeeze().cpu().numpy()

            # image erosion to discontinue predicted implanting areas.

            labeled_implants, rank = Oral.label_and_sort(pred_implants, erosion=True)

            for idx in range(tooth.max_implant_num):

                implant_area = np.zeros_like(pred_implants)
                implant_area[labeled_implants == rank[idx][1] + 1] = 1

                implant = Implant(implant_area, self.env_vox)
                tooth.implants.append(implant)
                tooth.teeth_classification(idx, cbct, ts=0.2)
                Oral.implant_centre_filling(idx, tooth)

            side = {'upper': '上', 'down': '下'}
            position = {'front': '前牙', 'middle': '尖牙', 'back': '后牙'}
            temp = tooth.implants[0]
            self.log_book = """\n算法判定缺牙编号{}位于{}方，是一颗{}，根据种植规则，从距离牙冠表面2mm起，选取长度为{}mm, 直径{}mm的三维圆柱形区域作为的种植推荐位置."""\
                .format(teeth_no, side[temp.side], position[temp.cls], temp.length, temp.radius * 2)

    @staticmethod
    def label_and_sort(pred, erosion=False):

        if erosion:
            # image erosion to discontinue predicted implanting areas.
            eroded_mask = binary_erosion(pred, iterations=1)
            pred[~eroded_mask] = 0

        labeled_implants = morphology.label(pred)

        rank = list(zip([labeled_implants[labeled_implants == (i + 1)].sum() for i in range(labeled_implants.max())],
                        [i for i in range(labeled_implants.max())]))

        rank.sort(key=lambda x: x[0], reverse=True)

        return labeled_implants, rank

    @staticmethod
    def implant_centre_filling(idx, tooth):

        crown = tooth.data
        implant = tooth.implants[idx]

        cent_crown = tooth.centroid
        cent_implant = implant.centroid
        matrix_size = crown.shape

        diff = np.array(cent_implant) - np.array(cent_crown)
        diff = diff / np.abs(diff[-1])

        res = np.zeros(matrix_size)

        bound = np.copy(cent_crown).astype(np.float64)

        while crown[round(bound[0]), round(bound[1]), round(bound[2])] == 1:

            bound += diff

            if any(bound + diff) < 0 or any(bound + diff) > matrix_size[2]:
                break

        radius = implant.radius / implant.vox
        length = round(implant.length / implant.vox)

        start = Oral.point_shift_along_vector(diff, bound, 2 / implant.vox)
        end = Oral.point_shift_along_vector(diff, start, length)

        num = abs(round(end[-1] - start[-1]))

        pointer = start

        # generate stl file here.

        for i in range(num):

            pointer += diff

            for x in range(round(pointer[0] - radius), round(pointer[0] + radius)):

                for y in range(round(pointer[1] - radius), round(pointer[1] + radius)):

                    if abs(x - pointer[0]) ** 2 + abs(y - pointer[1]) ** 2 <= radius ** 2:
                        res[x, y, round(pointer[2])] = 1

        implant.implant = Entity(res, implant.vox)

    @staticmethod
    def point_shift_along_vector(vector, start, shift):

        sgns = np.sign(vector)
        vector = np.abs(vector)
        x_shift = sgns[0] * vector[0] / np.linalg.norm(vector)
        y_shift = sgns[1] * vector[1] / np.linalg.norm(vector)
        z_shift = sgns[2] / np.linalg.norm(vector)

        return start + np.array([x_shift, y_shift, z_shift]) * shift

    @staticmethod
    def patching_with_centroid(centroid, patch_size, volume, scale=1):

        patch_border = list(range(3))

        volume = zoom(volume, (1 / scale, 1 / scale, 1 / scale))

        for dim in range(3):
            start = centroid[dim] - patch_size // 2 if patch_size // 2 < centroid[dim] else 0

            end = start + patch_size

            end = min(end, volume.shape[dim])

            start = end - patch_size

            patch_border[dim] = (round(start), round(end))

        return volume[patch_border[0][0]: patch_border[0][1],
               patch_border[1][0]: patch_border[1][1], patch_border[2][0]: patch_border[2][1]]

    def calc_padding(self, centroid):

        padding = []
        for dim in range(3):

            fp = max(centroid[dim] - self.PATCH_SZ // 2, 0)
            if fp != 0:
                delta = fp + self.PATCH_SZ - self.full_size[dim]
                fp = fp if delta < 0 else (fp - delta)

            bp = max(self.full_size[dim] - centroid[dim] - self.PATCH_SZ // 2, 0)
            if bp != 0:
                delta = bp + self.PATCH_SZ // 2 + centroid[dim] - self.full_size[dim]
                bp = bp if delta < 0 else (bp - delta)

            bp, fp = round(bp), round(fp)

            if fp + bp - self.full_size[dim] + self.PATCH_SZ > 1:
                warnings.warn('存在一个植体补零时未对齐原图尺寸，请检查代码实现是否有BUG')

            padding.extend([round(bp), round(fp)])

        padding.reverse()

        return padding