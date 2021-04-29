# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesPreprocessedDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    RAW_WIDTH = 1024
    RAW_HEIGHT = 384

    def __init__(self, *args, **kwargs):
        super(CityscapesPreprocessedDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side

    def check_depth(self):
        return False

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner

        camera_file = os.path.join(self.data_path, city, "{}_cam.txt".format(frame_name))
        camera = np.loadtxt(camera_file, delimiter=",")
        fx = camera[0]
        fy = camera[4]
        u0 = camera[2]
        v0 = camera[5]
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0,  0,  1, 0],
                               [0,  0,  0, 1]]).astype(np.float32)

        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def get_colors(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        color = self.loader(self.get_image_path(city, frame_name))
        color = np.array(color)

        w = color.shape[1] // 3
        inputs = {}
        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w:2*w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2*w:])

        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)

        return inputs

    def get_image_path(self, city, frame_name):
        return os.path.join(self.data_path, city, "{}.jpg".format(frame_name))
