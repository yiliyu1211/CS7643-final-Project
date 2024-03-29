import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
import torch
from fastmri.data.transforms import UnetDataTransform
import numpy as np
import cv2


# Create a mask function
mask = subsample.RandomMaskFunc(
    center_fractions=[0.08, 0.04],
    accelerations=[4, 8]
)

# mask = subsample.create_mask_for_mask_type

# def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
#     # Transform the data into appropriate format
#     # Here we simply mask the k-space and return the result
#     kspace = transforms.to_tensor(kspace)
#     masked_kspace = transforms.apply_mask(kspace, mask_func)
#     return masked_kspace

