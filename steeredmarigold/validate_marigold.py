import os
import cv2
import torch
import numpy
from functools import partial
from os.path import join
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from steeredmarigold.average import ValidationResult
from steeredmarigold.utils import float_depth_to_8bit_bgr, save_arguments_to_json, ARGS_FILE, \
    parse_val_args, getattr_from_string, vae_encode, vae_decode
from steeredmarigold.constants import *

def validate_model(args):
    if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    save_arguments_to_json(args, join(args.output_dir, ARGS_FILE))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_dtype = torch.float32 if args.weight_dtype == "fp32" else torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant).to(device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.checkpoint, 
        subfolder=args.checkpoint_subdir, 
        in_channels=8,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    ).to(device, dtype=weight_dtype)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    preprocessing_func = partial(getattr_from_string(args.val_preprocessing), args)
    evaluation = getattr_from_string(args.evaluation_function)
    dataset = getattr_from_string(args.val_dataset_name)(args.val_data_dir, args.val_dataset_set, preprocessing_func, args.limit_dataset_size)

    result = ValidationResult()

    text_inputs = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample_data = dataset[sample_idx]

            if args.val_max_steps == sample_idx:
                break

            print(sample_data[SAMPLE_ID])

            conditioning_latents = sample_data[RGB_TENSOR_1_1]
            conditioning_latents = torch.unsqueeze(conditioning_latents, 0).to(device)
            conditioning_latents = vae_encode(vae, conditioning_latents) 
            
            prediction_latents = torch.randn(conditioning_latents.shape, device=device, dtype=weight_dtype)

            noise_scheduler.set_timesteps(args.denoising_steps, device=device)
            timesteps = noise_scheduler.timesteps 

            for t in timesteps:
                unet_input = torch.cat((conditioning_latents, prediction_latents), 1)
                noise_pred = unet(unet_input, t, encoder_hidden_states=encoder_hidden_states).sample
                prediction_latents = noise_scheduler.step(noise_pred, t, prediction_latents).prev_sample

            prediction = vae_decode(vae, prediction_latents)

            prediction = torch.clip(prediction, -1.0, 1.0)
            prediction = (prediction + 1.0) / 2.0
            prediction = prediction.cpu().numpy()
            prediction = numpy.squeeze(prediction, 0)
            prediction = numpy.average(prediction, axis=0)
            prediction = prediction.astype(numpy.float32)
            
            if args.store_raw:
                result_path = join(args.output_dir, sample_data["id"] + ".tif")
                cv2.imwrite(result_path, prediction)
            
            if args.save_visualizations:
                result_path = join(args.output_dir, sample_data["id"] + "_colored.png")
                cv2.imwrite(result_path, float_depth_to_8bit_bgr(prediction))

            if args.compute_metrics:
                metrics = evaluation(args, sample_data, prediction)
                result.add_value(sample_data["id"], metrics)

        if args.compute_metrics:
            result.save_average_to_json(join(args.output_dir, "average.json"))
            result.save(join(args.output_dir, "metrics.csv"))

    torch.cuda.empty_cache()

if __name__ == "__main__":
    validate_model(parse_val_args())
