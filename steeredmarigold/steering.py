import cv2
import numpy
from steeredmarigold.callables import *
from steeredmarigold.utils import recover_metric_depth


def make_interpolated_steering(x0_depth_m1to1: numpy.ndarray, lidar_meters: numpy.ndarray, 
                               lidar_meters_mask: numpy.ndarray, lidar_align: bool, neighborhood: int) -> numpy.ndarray:  
    x0_sampled, x0_sampled_mask = RandomPointsFromDepth(preserve=0.05, background=-1.0, valid_threshold=None, return_mask=True)(x0_depth_m1to1)

    lidar = numpy.full(shape=lidar_meters.shape, fill_value=-1.0)
    lidar[lidar_meters_mask] = (lidar_meters[lidar_meters_mask] - lidar_meters[lidar_meters_mask].min()) / (lidar_meters[lidar_meters_mask].max() - lidar_meters[lidar_meters_mask].min())
    lidar[lidar_meters_mask] = (lidar[lidar_meters_mask] * 2.0) - 1.0

    if lidar_align:
        lidar = recover_metric_depth(lidar, x0_depth_m1to1, lidar_meters_mask)
        lidar[numpy.logical_not(lidar_meters_mask)] = -1.0

    lidar_area = lidar_meters_mask.astype(dtype=numpy.uint8)
    lidar_area = cv2.dilate(lidar_area, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (neighborhood, neighborhood)), iterations=1)
    lidar_area = lidar_area.astype(numpy.bool_)
    not_lidar_area = numpy.logical_not(lidar_area)

    lidar_merged = numpy.copy(lidar)
    lidar_merged[not_lidar_area] = x0_sampled[not_lidar_area]

    resample_mask = numpy.logical_and(lidar_area, lidar_meters_mask)
    merged_mask = numpy.logical_or(resample_mask, numpy.logical_and(x0_sampled_mask, not_lidar_area))

    x0_sampled[lidar_area] = -1.0
    x0_sampled[resample_mask] = x0_depth_m1to1[resample_mask]
    x0_sampled_mask = None

    height, width = x0_depth_m1to1.shape
    if not merged_mask[0, 0]:
        merged_mask[0, 0] = True
        lidar_merged[0, 0] = x0_depth_m1to1[0, 0]
        x0_sampled[0, 0] = x0_depth_m1to1[0, 0]
    
    if not merged_mask[0, width - 1]:
        merged_mask[0, width - 1] = True
        lidar_merged[0, width - 1] = x0_depth_m1to1[0, width - 1]
        x0_sampled[0, width - 1] = x0_depth_m1to1[0, width - 1]

    if not merged_mask[height - 1, 0]:
        merged_mask[height - 1, 0] = True
        lidar_merged[height - 1, 0] = x0_depth_m1to1[height - 1, 0]
        x0_sampled[height - 1, 0] = x0_depth_m1to1[height - 1, 0]

    if not merged_mask[height - 1, width - 1]:
        merged_mask[height - 1, width - 1] = True
        lidar_merged[height - 1, width - 1] = x0_depth_m1to1[height - 1, width - 1]
        x0_sampled[height - 1, width - 1] = x0_depth_m1to1[height - 1, width - 1]

    merged_interpolated = InterpolateLidar("linear", -1.0, valid_mask=merged_mask)(lidar_merged)
    x0_interpolated = InterpolateLidar("linear", -1.0, valid_mask=merged_mask)(x0_sampled)

    return x0_depth_m1to1 - x0_interpolated + merged_interpolated
