import numpy as np
import warnings


def crown_classification(cbct, cx, ts=0.3):

    x, y, z = cbct.shape
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

    if idx > 1:

        return 'back'

    else:
        return classifier[idx]


def implant_specification(cls, side):

    if cls == 'front':

        radius, length = 1.5, 11.5

        radius = radius + 0.15 if side == 'upper' else radius - 0.15

    elif cls == 'middle':

        radius, length = 2.1, 10.0

    elif cls == 'back':

        radius, length = 2.25, 8.0

    else:

        warnings.warn('牙冠定位失败，请检查代码实现')

        return 2.0, 11.0

    return radius, length