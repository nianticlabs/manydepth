# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

manydepth_models_path = os.path.join(os.path.expanduser('~'), '.manydepth_models')
git_dir = os.path.join(manydepth_models_path, 'manydepth')
model_subfolder_names = {
    "KITTI_MR_640_192": "KITTI_MR",
    "KITTI_HR_1024_320": "KITTI_HR",
    "CityScapes_512_192": "CityScapes_MR"
}

def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "KITTI_MR_640_192": "https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/KITTI_MR.zip",
        "KITTI_HR_1024_320": "https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/KITTI_HR.zip",
        "CityScapes_512_192": "https://storage.googleapis.com/niantic-lon-static/research/manydepth/models/CityScapes_MR.zip"
    }

    if not os.path.exists(manydepth_models_path):
        os.makedirs(manydepth_models_path)

    model_path = os.path.join(manydepth_models_path, model_name)
    
    subfolder_name = model_subfolder_names[model_name]

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, subfolder_name, "encoder.pth")):

        model_url = download_paths[model_name]

        print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
        urllib.request.urlretrieve(model_url, model_path + ".zip")

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
