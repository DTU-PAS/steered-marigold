import h5py
import numpy
from typing import Callable
from os.path import join, realpath, dirname
from torch.utils.data import Dataset
from steeredmarigold.constants import SAMPLE_ID

RGB = "rgb"
DEPTH = "depth"
DEPTH_RAW = "depth_raw"

TEST_SUBSET_FILE = "nyuv2-test.txt"
DATA_FILE = "nyu_depth_v2_labeled.mat"

class Nyuv2(Dataset):
    def __init__(self, path: str, subset: str, preprocessing: Callable, limit_size: int = None):
        assert subset in ["all", "test"]
        
        self._path = path
        self._subset = subset
        self._preprocessing = preprocessing
        self._limit_size = limit_size

        with h5py.File(join(self._path, DATA_FILE), "r") as datafile:
            self._images = numpy.array(datafile.get("images"))
            self._images = numpy.transpose(self._images, (0, 3, 2, 1))
            self._depths = numpy.array(datafile.get("depths"))
            self._depths = numpy.transpose(self._depths, (0, 2, 1))
            self._depths_raw = numpy.array(datafile["rawDepths"])
            self._depths_raw = numpy.transpose(self._depths_raw, (0, 2, 1))
            self._ids = numpy.arange(start=1, stop=self._images.shape[0] + 1, step=1, dtype=numpy.int32)

        if subset == "test":
            with open(join(realpath(dirname(__file__)), TEST_SUBSET_FILE), "r") as f:
                val_idxs = [int(line) - 1 for line in f.readlines()]

                self._images = self._images[val_idxs, ...]
                self._depths = self._depths[val_idxs, ...]
                self._depths_raw = self._depths_raw[val_idxs, ...]
                self._ids = self._ids[val_idxs]

        if limit_size is not None:
            self._images = self._images[:self._limit_size, ...]
            self._depths = self._depths[:self._limit_size, ...]
            self._depths_raw = self._depths_raw[:self._limit_size, ...]
            self._ids = self._ids[:limit_size]

    def __getitem__(self, index):
        item = {
            RGB: self._images[index, ...],
            DEPTH: self._depths[index, ...],
            DEPTH_RAW: self._depths_raw[index, ...],
            SAMPLE_ID: str(self._ids[index])
        }

        if self._preprocessing is not None:
            item = self._preprocessing(item)

        return item

    def __len__(self):
        return self._images.shape[0]