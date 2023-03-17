import numpy as np

from registration.methods import AbstractRegistrationMethod
import sys
sys.path.append('GUI/events/registration/method_demo')
from utils import nii2dcm_1file, save_array_as_nii, get_region_bbx, patching_with_centroid, implant_augmentation, Unet, read_dcm
import torch
import nibabel


class DemoMethod(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):

        super(DemoMethod, self).__init__(cbct_path, save_path, args)

    def missing_tooth_localization(self):

        centroid = get_region_bbx(self.implant)
        self.cbct = self.cbct.split(' ')

        if len(self.cbct) == 1:
            self.cbct = nibabel.load(self.cbct[0]).get_fdata()

        else:
            self.cbct = read_dcm(self.cbct)

        cbct_patch = patching_with_centroid(centroid, 96, self.cbct)
        implant_patch = patching_with_centroid(centroid, 96, self.implant)
        self.log_book = """该演示方法使用给定的模板植体进行缺牙定位"""
        return implant_patch, cbct_patch

    def load_implant(self):

        self.implant = nibabel.load(self.implant).get_fdata()

    def registration(self):

        device = torch.device(self.device)
        self.implant = torch.from_numpy(self.implant).unsqueeze(0).to(device, torch.float)
        self.cbct = torch.from_numpy(self.cbct).unsqueeze(0).unsqueeze(0).to(device, torch.float)
        model = Unet(4, 16).to(device)
        model.load_state_dict(torch.load(self.args.model_state_path))

        self.log_book = """此早期人工智能算法随机旋转植体来模拟得到未匹配的植体，然后再进行植体辅助种植"""

        with torch.no_grad():

            pred = model(self.cbct).squeeze().cpu().numpy()

        self.pred = pred

    def save(self):

        implant = self.pred
        save_array_as_nii(implant, self.save_path + '/nifti')
        nii2dcm_1file(implant, self.save_path)
        self.log_book = """已将DICOM和NIFTI格式的植体文件保存在指定文件目录"""

