import cv2
import json
import argparse
import importlib
import numpy
import torch
import matplotlib
from typing import Dict


ARGS_FILE = "args.json"


def getattr_from_string(function_name: str):
    components = function_name.split(".")
    module = importlib.import_module(".".join(components[:-1]))
    func = getattr(module, components[-1])
    return func


def save_arguments_to_json(args, path: str):
    argparse_dict = vars(args)

    json_data = json.dumps(argparse_dict, indent=2, sort_keys=True)

    with open(path, "w") as file:
        file.write(json_data)


def parse_val_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model")
    parser.add_argument("--pretrained-model-name-or-path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-subdir", type=str, default=None, help="Subir in the --checkpoint directory.")
    parser.add_argument("--val-dataset-name", type=str, required=True)
    parser.add_argument("--val-dataset-set", type=str, required=True)
    parser.add_argument("--val-data-dir", type=str, default=None, required=True)
    parser.add_argument("--val-preprocessing", type=str, required=True)
    parser.add_argument("--val-max-steps", type=int, default=None, help="Limits amount of samples from the dataset for validation.")
    parser.add_argument("--denoising-steps", type=int, default=None, help="Number of denoising steps performed during inference.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None, help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--weight-dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--min-depth", default=float("-inf"), type=float)
    parser.add_argument("--max-depth", default=float("inf"), type=float)
    parser.add_argument("--limit-dataset-size", default=None, type=int)

    # steering
    parser.add_argument("--steering-factor", type=float, default=None, help="Diffusion steeting factor, the larger number, the higher steering.")
    parser.add_argument("--neighborhood", type=int, default=None)
    parser.add_argument("--align", action="store_true", help="Align scale and shift of the depth condition with the depth estimate.")

    # preprocessing
    parser.add_argument("--center-crop-height", type=int, default=None, 
                        help="Crop image to the given height if the prepreocessing uses center crop.")
    parser.add_argument("--center-crop-width", type=int, default=None, 
                        help="Crop image to the given width if the preprocessing uses center crop.")
    parser.add_argument("--naive-lidar", type=float, default=None, 
                        help="Portion of valid points from ground-truth to be preserved and used as naive lidar. Examples: 0.05 preserves 5\% of points")
    
    # visualizations
    parser.add_argument("--store-raw", default=False, action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")

    # evaluation
    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--evaluation-function", type=str, default=None, 
                        help="Evaluation function used to compute metrics from the model prediction.")
    parser.add_argument("--eval-center-crop-height", type=int, default=None,
                        help="Crops evaluation area to specific height, must be specified tohether with --eval-center-crop-width.")
    parser.add_argument("--eval-center-crop-width", type=int, default=None,
                        help="Crops evaluation area to specific width, must be specified together with --eval-center-crop-height.")

    return parser.parse_args()


def vae_encode(vae, tensor: torch.Tensor) -> torch.Tensor:
    encoded = vae.encoder(tensor)
    mean, _ = torch.chunk(vae.quant_conv(encoded), 2, dim=1)
    return mean * vae.config.scaling_factor


def vae_decode(vae, tensor: torch.Tensor) -> torch.Tensor:
    scaled = tensor / vae.config.scaling_factor
    decoded = vae.post_quant_conv(scaled)
    return vae.decoder(decoded)


def float_depth_to_8bit_bgr(depth: numpy.ndarray, map:str="Spectral") -> numpy.ndarray:
    return cv2.cvtColor(float_depth_to_8bit_rgb(depth, map), cv2.COLOR_RGB2BGR)


def float_depth_to_8bit_rgb(depth: numpy.ndarray, map:str="Spectral") -> numpy.ndarray:
    cm = matplotlib.colormaps[map]
    colored = cm(depth, bytes=False)[:,:,:3]
    colored = colored * 255
    return colored.astype(numpy.uint8)


def compute_metrics(pred: numpy.ndarray, gt: numpy.ndarray, mask: numpy.ndarray) -> Dict[str, float]:
    assert pred.dtype == gt.dtype == numpy.float32

    gt_masked = gt[mask]
    pred_masked = pred[mask]

    abs_rel = numpy.mean(numpy.abs(gt_masked - pred_masked) / gt_masked)

    rms = (gt_masked - pred_masked) ** 2
    rms = numpy.sqrt(rms.mean())

    thresh = numpy.maximum((gt_masked / pred_masked), (pred_masked / gt_masked))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    mae = numpy.mean(numpy.abs(pred_masked - gt_masked))
    
    return {
        "abs_rel": abs_rel, 
        "rms": rms,
        "mae": mae, 
        "d1": d1, 
        "d2": d2, 
        "d3": d3,
        }


def recover_metric_depth(pred: numpy.ndarray, target: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    target_masked = target[mask]
    pred_masked = pred[mask]
    a, b = numpy.polyfit(pred_masked, target_masked, deg=1)
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = numpy.mean(pred_masked)
        target_mean = numpy.mean(target_masked)
        pred_metric = pred * (target_mean / pred_mean)

    return pred_metric


def three_channel_depth(one_channel_depth: numpy.ndarray) -> numpy.ndarray:
    one_channel_depth = numpy.expand_dims(one_channel_depth, axis=2)
    return numpy.concatenate((one_channel_depth, one_channel_depth, one_channel_depth), axis=2)


def one_channel_depth(three_channel_depth: numpy.ndarray) -> numpy.ndarray:
    return numpy.average(three_channel_depth, axis=2)