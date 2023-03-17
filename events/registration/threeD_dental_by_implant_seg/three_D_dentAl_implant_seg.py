import numpy as np

from events.registration.methods import AbstractRegistrationMethod
import sys
sys.path.append('GUI/events/registration/threeD_dentAL_by_implant_seg')
from utils import nii2dcm_1file, save_array_as_nii, get_region_bbx, nii2stl,\
    patching_with_centroid, implant_augmentation, Unet, read_dcm, implant_identification
import torch
import nibabel
import skimage


class ThreeDDentALByImplantSeg(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):

        super(ThreeDDentALByImplantSeg, self).__init__(cbct_path, save_path, args)

        self.cbct = self.cbct.split(' ')
        self.patch_size = 128
        self.sample_num = 3
        self.vox = args.vox

        self.name = '3D牙冠分割辅助植牙算法'

        self.side = args.side
        self.length = 6  # pre-defined
        self.radis = 3  # pre-defined

        if len(self.cbct) == 1:
            self.cbct = nibabel.load(self.cbct[0]).get_fdata()

        else:
            self.cbct = read_dcm(self.cbct)

    def missing_tooth_localization(self):

        # cbct_patch = patching_with_centroid(self.centroid, self.patch_size, self.cbct)
        # implant_patch = patching_with_centroid(self.centroid, self.patch_size, self.implant)
        # self.log_book = """缺牙识别定位中"""
        # return implant_patch, cbct_patch
        pass

    def registration(self):

        # prepare
        device = torch.device(self.device)
        self.cbct = torch.from_numpy(self.cbct).unsqueeze(0).unsqueeze(0).to(device, torch.float)
        self.patch_size = min(self.patch_size, self.cbct.shape[-1])
        model = Unet(4, 16).to(device)
        model.load_state_dict(torch.load('H:\\dentAL\\GUI\events\\registration\\threeD_dental_by_implant_seg\\model.pkl'))

        with torch.no_grad():
            pred = model(self.cbct)

        # accuracy verification breakpoint
        save_array_as_nii(pred.squeeze().cpu().numpy(), 'pred')
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = pred.squeeze().cpu().numpy()

        pred[pred < 0.7] = 0
        pred[pred > 0.7] = 1

        # post-processing
        labeled = skimage.morphology.label(pred)

        pred_implant_num = min(self.sample_num, labeled.max())

        rank = list(zip([labeled[labeled == (i + 1)].sum() for i in range(labeled.max())], [i for i in range(labeled.max())]))
        rank.sort(key=lambda x: x[0], reverse=True)
        implants = []
        abnormal = 0

        for i in range(pred_implant_num):

            implant = np.zeros_like(pred)
            implant[labeled == rank[i][1]] = 1

            # abnormal detection
            if implant.sum() < 500:

                abnormal += 1

                continue

            implants.append(implant[np.newaxis])

        self.log_book = """算法共检测到共{}个潜在的植体摆放位置，排除掉其它异常结果后，只保留了前{}个最有可能的种植的推荐位置.""".\
            format(labeled.max(), len(implants))

        self.pred = {'implants': np.concatenate(implants)}

    def save(self):

        implants = self.pred['implants']

        for idx, implant in enumerate(implants):

            save_array_as_nii(implant, self.save_path + '/implant' + str(idx))
            stl_implant = nii2stl(implant)
            stl_implant.save(self.save_path + '/implant' + str(idx) + '.stl')

        self.log_book = """已将NIFTI和stl格式的植体文件保存在指定文件目录，编号顺序即为推荐的植体摆放位置顺序。"""

