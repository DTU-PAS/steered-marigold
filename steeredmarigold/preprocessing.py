import numpy
from argparse import Namespace
from typing import Dict
from torchvision import transforms
from typing import Dict
from steeredmarigold.constants import *
from steeredmarigold.callables import *
from steeredmarigold.dataset import RGB, DEPTH_RAW


def evaluate_naive_lidar_metric(args: Namespace, item: Dict) -> Dict:
    assert args.naive_lidar is not None, "--naive-lidar must be set"

    rgb = item[RGB]
    gt = item[DEPTH_RAW]

    target_height, target_width = args.center_crop_height, args.center_crop_width

    rgb_tensor = NumpyCropCenter(target_height, target_width)(rgb)
    rgb_tensor = transforms.ToTensor()(rgb_tensor)
    rgb_tensor = transforms.Normalize([0.5], [0.5])(rgb_tensor)
    item[RGB_TENSOR_1_1] = rgb_tensor

    naive_lidar_metric = NumpyCropCenter(target_height, target_width)(gt)
    naive_lidar_metric = RandomPointsFromDepth(args.naive_lidar, 0.0, 0.0)(naive_lidar_metric)
    item[LIDAR_METERS] = naive_lidar_metric

    naive_lidar_metric_mask = naive_lidar_metric > 0.0
    item[LIDAR_METERS_MASK] = naive_lidar_metric_mask

    gt_metric = NumpyCropCenter(target_height, target_width)(gt)
    item[GT_METERS] = gt_metric

    gt_metric_mask = gt_metric > 0.0
    item[GT_METERS_MASK] = gt_metric_mask

    return item
    
def naive_lidar_metric_missing_area(args: Namespace, item: Dict) -> Dict:
    assert args.naive_lidar is not None, "--naive-lidar must be set"

    rgb = item[RGB]
    gt = item[DEPTH_RAW]
    frame_width = 100

    target_height, target_width = args.center_crop_height, args.center_crop_width
    
    missing_area_mask = numpy.full(shape=(target_height, target_width), fill_value=False, dtype=numpy.bool_)
    missing_area_mask[frame_width:target_height - frame_width, frame_width:target_width - frame_width] = True
    item[MISSING_AREA_MASK] = missing_area_mask

    rgb_tensor = NumpyCropCenter(target_height, target_width)(rgb)
    rgb_tensor = transforms.ToTensor()(rgb_tensor)
    rgb_tensor = transforms.Normalize([0.5], [0.5])(rgb_tensor)
    item[RGB_TENSOR_1_1] = rgb_tensor

    full_naive_lidar_metric = NumpyCropCenter(target_height, target_width)(gt)
    full_naive_lidar_metric = RandomPointsFromDepth(args.naive_lidar, 0.0, 0.0)(full_naive_lidar_metric)
    item[LIDAR_METERS_FULL] = full_naive_lidar_metric

    full_naive_lidar_metric_mask = full_naive_lidar_metric > 0.0
    item[LIDAR_METERS_FULL_MASK] = full_naive_lidar_metric_mask

    naive_lidar_metric = numpy.array(full_naive_lidar_metric, copy=True)
    naive_lidar_metric[missing_area_mask] = 0.0
    item[LIDAR_METERS] = naive_lidar_metric

    naive_lidar_metric_mask = numpy.array(full_naive_lidar_metric_mask, copy=True)
    naive_lidar_metric_mask[missing_area_mask] = False
    item[LIDAR_METERS_MASK] = naive_lidar_metric_mask
    
    gt_metric = NumpyCropCenter(target_height, target_width)(gt)
    item[GT_METERS] = gt_metric
    
    gt_metric_mask = gt_metric > 0.0
    item[GT_METERS_MASK] = gt_metric_mask
    
    return item
