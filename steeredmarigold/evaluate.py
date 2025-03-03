import numpy
from typing import Dict, Any
from argparse import Namespace
from steeredmarigold.constants import *
from steeredmarigold.callables import FillOutsideCenter
from steeredmarigold.utils import  compute_metrics, recover_metric_depth

def mde_to_dc(args: Namespace, item: Dict[str, Any], prediction_0to1: numpy.ndarray) -> Dict[str, float]:
    lidar_metric = item[LIDAR_METERS]
    lidar_metric_mask = item[LIDAR_METERS_MASK]
    lidar_eval_mask = numpy.logical_and(lidar_metric > args.min_depth, lidar_metric < args.max_depth)
    lidar_eval_mask = numpy.logical_and(lidar_metric_mask, lidar_eval_mask)

    if args.eval_center_crop_height is not None and args.eval_center_crop_width is not None:
        lidar_eval_mask = FillOutsideCenter(args.eval_center_crop_height, args.eval_center_crop_width)(lidar_eval_mask, False)

    prediction_metric = recover_metric_depth(prediction_0to1, lidar_metric, lidar_eval_mask)

    gt_metric = item[GT_METERS]
    gt_metric_mask = item[GT_METERS_MASK]
    gt_eval_mask = numpy.logical_and(gt_metric > args.min_depth, gt_metric < args.max_depth)
    gt_eval_mask = numpy.logical_and(gt_metric_mask, gt_eval_mask)

    if args.eval_center_crop_height is not None and args.eval_center_crop_width is not None:
        gt_eval_mask = FillOutsideCenter(args.eval_center_crop_height, args.eval_center_crop_width)(gt_eval_mask, False)

    return compute_metrics(prediction_metric, gt_metric, gt_eval_mask)

def mde_to_dc_only_gt_cropping(args: Namespace, item: Dict[str, Any], prediction_0to1: numpy.ndarray) -> Dict[str, float]:
    lidar_metric = item[LIDAR_METERS]
    lidar_metric_mask = item[LIDAR_METERS_MASK]
    lidar_eval_mask = numpy.logical_and(lidar_metric > args.min_depth, lidar_metric < args.max_depth)
    lidar_eval_mask = numpy.logical_and(lidar_metric_mask, lidar_eval_mask)

    prediction_metric = recover_metric_depth(prediction_0to1, lidar_metric, lidar_eval_mask)

    gt_metric = item[GT_METERS]
    gt_metric_mask = item[GT_METERS_MASK]
    gt_eval_mask = numpy.logical_and(gt_metric > args.min_depth, gt_metric < args.max_depth)
    gt_eval_mask = numpy.logical_and(gt_metric_mask, gt_eval_mask)

    if args.eval_center_crop_height is not None and args.eval_center_crop_width is not None:
        gt_eval_mask = FillOutsideCenter(args.eval_center_crop_height, args.eval_center_crop_width)(gt_eval_mask, False)

    return compute_metrics(prediction_metric, gt_metric, gt_eval_mask)
