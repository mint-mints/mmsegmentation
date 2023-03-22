# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CRACKDataset(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    CLASSES = ('background', 'crack')

    PALETTE = [[0, 0, 0], [250, 250, 250]]
    #mmseg要求mask的像素在[0，num_classes-1]范围内，比如我是2分类，背景像素值为0，那么目标像素值应该为1。
    # 如果你也是二分类，mask为单通道（8 bit）二值化的0（背景）/255（目标）图像的话,
    # 先去把图像改为0（背景）/1（目标）图像，否则能跑起来，但是指标异常，几乎全是0。（这是个大坑！！！一定要注意）

    def __init__(self, split, **kwargs):
        super(CRACKDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            split=split,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
