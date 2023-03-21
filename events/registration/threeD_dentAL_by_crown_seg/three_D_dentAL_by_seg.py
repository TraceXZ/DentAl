import numpy as np
from registration.methods import AbstractRegistrationMethod
import sys
sys.path.append('GUI/events/registration/threeD_dentAL_by_crown_seg')
from scipy.ndimage import binary_erosion
from utils import nii2dcm_1file, save_array_as_nii, get_region_bbx, nii2stl, patching_with_centroid, \
patching_with_centroid, implant_augmentation, Unet, read_dcm, implant_identification, \
affine_transformation, implant_centre_filling, implant_by_rotation
import warnings
import torch
import nibabel
import skimage
from math import ceil, floor
import torch.nn.functional as F


PATCH_SIZE = 72

"""
**** 1.4.2 deprecated, use method.py instead. ****
"""


class ThreeDDentALByCrownSeg(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):

        super(ThreeDDentALByCrownSeg, self).__init__(cbct_path, save_path, args)

        self.cbct = self.cbct.split(' ')
        self.sample_num = 5
        self.vox = args.vox
        self.side = args.side

        self.length = 6  # pre-defined
        self.radius = float(self.args.radius)  # pre-defined
        self.device = torch.device(self.device)


    def missing_tooth_localization(self):

        # cbct_patch = patching_with_centroid(self.centroid, self.patch_size, self.cbct)
        # implant_patch = patching_with_centroid(self.centroid, self.patch_size, self.implant)
        # self.log_book = """缺牙识别定位中"""
        # return implant_patch, cbct_patch
        pass

    def load_cbct(self):

        if len(self.cbct) == 1:
            self.cbct = nibabel.load(self.cbct[0]).get_fdata()
            # self.orignal_vox = cbct.affine.diagonal()[:-1]
            # self.cbct = zoom(cbct.get_fdata(), self.orignal_vox / np.array(0.4))

        else:
            self.cbct = read_dcm(self.cbct)

        w, h, d = self.cbct.shape

        self.cbct = torch.from_numpy(self.cbct).unsqueeze(0).unsqueeze(0).to(self.device, torch.float)

        max_dim = max(w, h, d)

        self.padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                   ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]

        self.cbct = F.pad(self.cbct, self.padding)

    def registration(self):

        # prepare
        full_size = self.cbct.shape[2:]
        model = Unet(4, 16).to(self.device)
        model.load_state_dict(torch.load('H:\\dentAL\\GUI\\events\\registration\\threeD_dentAL_by_crown_seg\\utils\\model.pkl'))

        # preprocessing
        # low_idx = int(self.args.top_low * self.full_size ** 3)
        # high_low = int(self.args.top_high * self.full_size ** 3)

        # sorted_cbct = torch.sort(self.cbct.reshape(-1))
        # low_value = sorted_cbct[0][low_idx]
        # high_value = sorted_cbct[0][high_low]

        # self.cbct[self.cbct < low_value] = low_value
        # self.cbct[self.cbct > high_value] = high_value

        # crown segmentation
        with torch.no_grad():
            pred_crowns = model(self.cbct)
        pred_crowns[pred_crowns < 0.6] = 0
        pred_crowns[pred_crowns > 0.6] = 1

        pred_crowns = pred_crowns.squeeze().cpu().numpy()

        # post-processing
        labeled_crowns = skimage.morphology.label(pred_crowns)

        if labeled_crowns.max() >= self.sample_num:
            self.log_book = """算法共检测到超过{}个潜在的牙冠摆放位置，但只保存了前{}个最有可能的位置。""".format(labeled_crowns.max(), self.sample_num)

        pred_crown_num = min(self.sample_num, labeled_crowns.max())

        crowns = []
        implants_ai = []
        implants_dc = []
        cropped_cbcts = []

        for i in range(pred_crown_num):

            crown = np.zeros_like(pred_crowns)
            crown[labeled_crowns == 1 + i] = 1

            # abnormal detection
            if crown.sum() < 200:
                continue

            from skimage.measure import regionprops
            centroid = regionprops(crown.astype(np.int))[0].centroid
            centroid = [round(c) for c in centroid]

            cropped_cbct = patching_with_centroid(centroid, PATCH_SIZE, self.cbct.squeeze().cpu().numpy())

            model = Unet(4, 16).to(self.device)
            model.load_state_dict(
                torch.load('H:\\dentAL\\GUI\events\\registration\\threeD_dental_by_implant_seg\\model.pkl'))
                # torch.load('H:\\dentAL\\GUI\\events\\registration\\threeD_dentAL_by_crown_seg\\utils\\implant_by_segmentation_180.pkl'))

            # implant from AI
            with torch.no_grad():
                pred_implants = model(torch.from_numpy(cropped_cbct).unsqueeze(0).unsqueeze(0).to(self.device, torch.float))

            # accuracy verification breakpoint
            # save_array_as_nii(pred.squeeze().cpu().numpy(), 'pred')

            # implant from theory
            dc = implant_identification(crown, self.vox, self.side, length=8, radius=3)[1]

            pred_implants = (pred_implants - pred_implants.min()) / (pred_implants.max() - pred_implants.min())

            pred_implants[pred_implants < 0.5] = 0
            pred_implants[pred_implants > 0.5] = 1

            padding = []
            for dim in range(3):

                fp = max(centroid[dim] - PATCH_SIZE // 2, 0)
                if fp != 0:
                    delta = fp + PATCH_SIZE - full_size[dim]
                    fp = fp if delta < 0 else (fp - delta)

                bp = max(full_size[dim] - centroid[dim] - PATCH_SIZE // 2, 0)
                if bp != 0:
                    delta = bp + PATCH_SIZE // 2 + centroid[dim] - full_size[dim]
                    bp = bp if delta < 0 else (bp - delta)

                if fp + bp != full_size[dim] - PATCH_SIZE:
                    warnings.warn('存在一个植体补零时未对齐原图尺寸，请检查代码实现是否有BUG')

                padding.extend([bp, fp])
            padding.reverse()

            pred_implants = torch.nn.functional.pad(pred_implants, padding)

            pred_implants = pred_implants.squeeze().cpu().numpy()

            # image erosion to discontinue predicted implanting areas.
            eroded_mask = binary_erosion(pred_implants, iterations=1)
            pred_implants[~eroded_mask] = 0

            labeled_implants = skimage.morphology.label(pred_implants)

            rank = list(zip([labeled_implants[labeled_implants == (i + 1)].sum() for i in range(labeled_implants.max())],
                            [i for i in range(labeled_implants.max())]))
            rank.sort(key=lambda x: x[0], reverse=True)

            implant_ai = np.zeros_like(pred_implants)

            # 1. set a hyperparameter to determine the number of predictions for each of missing teeth;
            # 2. Do a simple check to remove any irrational predictions,
            # e.g., any prediction overlapped with the predicted crown, or too far from crown, or remove predictions that are too small.
            # 3. change the way of cropping, instead of using the centre of predicted crown, considering any shift along
            # axial plane depending on the crown being upper or down side (code implementation need).
            # 4. Double check why the result didnt achieves as good as expected when the primary result achieved a good result. (Villa_778 test)
            implant_ai[labeled_implants == rank[0][1] + 1] = 1
            implant = implant_centre_filling(crown, centroid, regionprops(implant_ai.astype(np.int))[0].centroid, self.radius, self.length, self.vox, self.side, crown.shape)
            # implant = implant_by_rotation(crown, centroid, regionprops(implant_ai.astype(np.int))[0].centroid, self.radius, self.length, full_size, self.vox[0], self.side)
            # implant_ai = zoom(implant_ai, self.vox / self.orignal_vox)
            # dc = zoom(dc, self.vox / self.orignal_vox)
            # crown = zoom(crown, self.vox / self.orignal_vox)

            implants_ai.append(implant[np.newaxis])
            implants_dc.append(dc[np.newaxis])

            crowns.append(crown[np.newaxis])
            cropped_cbcts.append(cropped_cbct[np.newaxis])

        self.log_book = """算法选取长度为{}mm, 半径{}mm的三维矩形区域作为植体种植的推荐位置.""".format(self.length, self.radius)

        self.pred = {'crowns': np.concatenate(crowns), 'implants_ai':
            np.concatenate(implants_ai),'implants_dc': np.concatenate(implants_dc),
                     'cropped_cbcts': np.concatenate(cropped_cbcts)}

    def save(self):

        crowns = self.pred['crowns']
        implants_ai = self.pred['implants_ai']
        implants_dc = self.pred['implants_dc']
        cropped_cbcts = self.pred['cropped_cbcts']

        for idx, (crown, implant_ai, implant_dc, cropped_cbct) in enumerate(zip(crowns, implants_ai, implants_dc, cropped_cbcts)):

            # save_array_as_nii(crown, self.save_path + '/crown' + str(idx), self.orignal_vox)
            # save_array_as_nii(implant_ai, self.save_path + '/implant_ai' + str(idx), self.orignal_vox)
            # save_array_as_nii(implant_dc, self.save_path + '/implant_dc' + str(idx), self.orignal_vox)
            # save_array_as_nii(cropped_cbct, self.save_path + '/cropped_cbct' + str(idx))

            save_array_as_nii(crown, self.save_path + '/crown' + str(idx))
            save_array_as_nii(implant_ai, self.save_path + '/implant_ai' + str(idx))
            save_array_as_nii(implant_dc, self.save_path + '/implant_dc' + str(idx))
            save_array_as_nii(cropped_cbct, self.save_path + '/cropped_cbct' + str(idx))

            stl_implant_ai = nii2stl(implant_ai)
            stl_implant_dc = nii2stl(implant_dc)
            stl_crown = nii2stl(crown)
            stl_implant_ai.save(self.save_path + '/implant_ai' + str(idx) + '.stl')
            stl_implant_dc.save(self.save_path + '/implant_dc' + str(idx) + '.stl')
            stl_crown.save(self.save_path + '/crown' + str(idx) + '.stl')

        self.log_book = """已将NIFTI和stl格式的植体文件保存在指定文件目录，编号顺序即为推荐的植体摆放位置顺序。"""
