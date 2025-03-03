import random
import numpy
import torch
import scipy
from steeredmarigold.utils import three_channel_depth


class NumpyCropCenter:
    def __init__(self, height: int, width: int):
        self._target_height = height
        self._target_width = width

    def __call__(self, image: numpy.ndarray) -> numpy.ndarray:
        height, width = image.shape[:2]

        row_start = (height - self._target_height) // 2
        row_stop = row_start + self._target_height
        col_start = (width - self._target_width) // 2
        col_stop = col_start + self._target_width

        return image[row_start:row_stop, col_start:col_stop, ...]

class FillOutsideCenter:
    def __init__(self, height: int, width: int):
        self._height = height
        self._width = width

    def __call__(self, image: numpy.ndarray, fill_value) -> numpy.ndarray:
        height, width = image.shape[:2]

        row_start = (height - self._height) // 2
        row_end = row_start + self._height
        col_start = (width - self._width) // 2
        col_end = col_start + self._width

        image[:row_start, :, ...] = fill_value
        image[row_end:, :, ...] = fill_value
        image[:, :col_start, ...] = fill_value
        image[:, col_end:, ...] = fill_value

        return image


class NumpyOneToThreeChannels:
    def __call__(self, image: numpy.ndarray) -> numpy.ndarray:
        return three_channel_depth(image)


class NumpyRgbToTensorCHW:
    def __call__(self, image: numpy.ndarray) -> numpy.ndarray:
        return torch.from_numpy(numpy.transpose(image,(2, 0, 1)))
    
    
class RandomPointsFromDepth:
    def __init__(self, preserve: float, background: float = -1.0, valid_threshold: int | float | None = None, return_mask: bool = False) -> None:
        """
        :param preserve: Expresses amount of points to be preserved from depth images (example 0.5 preserves 50% of points).
        :param background: Value of pixels that are not copied from provided depth map.
        :param valid_threshold: only values larger than valid threshold will be preserved
        :param return_mask: if true, the callable returns also a mask with values set to true in positions of sampled values
        """
        self._preserve = preserve
        self._background = background
        self._valid_threshold = valid_threshold
        self._return_mask = return_mask

    def __call__(self, depth: numpy.ndarray) -> numpy.ndarray:
        assert depth.ndim == 2, "depth must be only two dimensional"
        height, width = depth.shape
        d = depth.ravel()

        if self._valid_threshold is None:
            population = list(range(0, d.shape[0]))
        else:
            population = list(numpy.ravel(numpy.argwhere(d > self._valid_threshold)))

        k = min(round(d.shape[0] * self._preserve), len(population))
        idxs = random.sample(population=population, k=k)
        result = numpy.full(shape=d.shape, fill_value=self._background, dtype=d.dtype)
        result[idxs] = d[idxs]
        result = numpy.reshape(result, (height, width))

        if self._return_mask:
            mask = numpy.full(shape=d.shape, fill_value=False, dtype=numpy.bool_)
            mask[idxs] = True
            mask = numpy.reshape(mask, (height, width))
            return result, mask

        return result


class InterpolateLidar:
    def __init__(self, interpolation: str = "linear", fill_value: float | None = None, valid_values_threshold: float | int = None, 
                 valid_mask: numpy.ndarray = None):
        assert (valid_values_threshold is not None) ^ (valid_mask is not None), \
            "only one of valid_values_threshold or valid_mask must be provided"
        
        self._interpolation = interpolation
        self._fill_value = fill_value
        self._valid_values_threshold = valid_values_threshold
        self._valid_mask = valid_mask

    def __call__(self, depth: numpy.ndarray) -> numpy.ndarray:
        assert depth.ndim == 2, "depth must be two dimensional"
        height, width = depth.shape

        if self._valid_mask is not None:
            points = numpy.argwhere(self._valid_mask)
        else:
            points = numpy.argwhere(depth > self._valid_values_threshold)
        
        values = depth[points[:, 0], points[:, 1]]
        points = numpy.flip(points, axis=1)
        X, Y = numpy.meshgrid(numpy.arange(0, width, 1), numpy.arange(0, height, 1))
        result = scipy.interpolate.griddata(points, values, (X, Y), self._interpolation, fill_value=self._fill_value)
        return result.astype(numpy.float32)
