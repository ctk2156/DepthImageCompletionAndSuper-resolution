import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def nyuV2_load_fn(path):
    cp, rp = path.strip().split(',')[:3]
    color = Image.open(cp)
    raw = Image.open(rp)

    res_wid, res_hei = color.size
    res_wid, res_hei = res_wid - res_wid%16, res_hei - res_hei%16

    img_wid, img_hei = color.size

    start_wid, start_hei = (img_wid - res_wid)//2, (img_hei - res_hei)//2
    box = (start_wid, start_hei, start_wid+res_wid, start_hei+res_hei)  # 设置要裁剪的区域

    color = color.crop(box)
    raw = raw.crop(box)

    color = np.asarray(color).astype(np.float32) / 255.
    raw = np.asarray(raw).astype(np.float32) / 1000.

    return color, raw


class NyuV2Dataset(Dataset):

    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, index):

        _color, _raw = nyuV2_load_fn(self.file_list[index])

        _color = _color.transpose((2, 0, 1))

        _input = np.concatenate([_color, np.expand_dims(_raw, axis=0)], axis=0)

        return torch.from_numpy(_input)

    def __len__(self):
        return len(self.file_list)

