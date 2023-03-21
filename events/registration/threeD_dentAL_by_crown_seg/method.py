import numpy as np
from events.registration.methods import AbstractRegistrationMethod
import sys
sys.path.append('events/registration/threeD_dentAL_by_crown_seg')
from utils import Oral, read_dcm, Unet, read_dcm
import torch
from scipy.ndimage import zoom
import torch.nn.functional as F


class ThreeDDentALByCrownSeg(AbstractRegistrationMethod):

    def __init__(self, cbct_path, save_path, args):
        super(ThreeDDentALByCrownSeg, self).__init__(cbct_path, save_path, args)

        self.cbct_path = cbct_path
        self.device = torch.device(args.device)
        self.max_crown_num = 2
        self.oral = None
        self.model = Unet(4, 16).to(self.device)

    def load_cbct(self):
        self.oral = Oral(self.cbct_path, self.device, self.max_crown_num)

    def missing_tooth_localization(self):
        self.model.load_state_dict(
            torch.load('events/registration/threeD_dentAL_by_crown_seg/utils/model.pkl'))
        self.oral.pred_teeth(self.model)
        self.log_book = self.oral.log_book

    def registration(self):
        self.model.load_state_dict(
            torch.load('events/registration/threeD_dental_by_implant_seg/model.pkl'))
        self.oral.pred_implant(self.model)
        self.log_book = self.oral.log_book

        padding = self.oral.padding
        _, _, w, h, d = self.oral.cbct.shape
        for idx, teeth in enumerate(self.oral.tooth):

            data = F.interpolate(torch.from_numpy(teeth.data[np.newaxis, np.newaxis])[:, :, padding[-2]: w - padding[-2], padding[-4]: h - padding[-4],
               padding[-6]: d - padding[-6]], scale_factor=self.oral.env_vox / self.oral.oral_vox)

            data[data <= 0.9] = 0
            data[data > 0.9] = 1

            teeth.data = data.squeeze().numpy()

            for implant in teeth.implants:
                data = F.interpolate(torch.from_numpy(implant.data[np.newaxis, np.newaxis][:, :, padding[-2]: w - padding[-2], padding[-4]: h - padding[-4],
               padding[-6]: d - padding[-6]]),
                                     scale_factor=self.oral.env_vox / self.oral.oral_vox)

                data[data <= 0.9] = 0
                data[data > 0.9] = 1

                implant.data = data.squeeze().numpy()

                pred_implant = F.interpolate(torch.from_numpy(implant.implant.data[np.newaxis, np.newaxis][:, :, padding[-2]: w - padding[-2], padding[-4]: h - padding[-4],
               padding[-6]: d - padding[-6]]),
                                     scale_factor=self.oral.env_vox / self.oral.oral_vox)

                pred_implant[pred_implant <= 0.9] = 0
                pred_implant[pred_implant > 0.9] = 1

                implant.implant.data = pred_implant.squeeze().numpy()

    def save(self):

        oral = self.oral

        for idx, teeth in enumerate(oral.tooth):
            teeth.save_as_nii(self.save_path + '/crown' + str(idx))
            teeth.save_as_stl(self.save_path + '/crown' + str(idx), self.oral.oral_vox, self.oral.oral_position)

        self.log_book = """\nnifti和stl文件已经保存在指定位置"""
