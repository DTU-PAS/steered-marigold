import os
import cv2
import torch
import numpy
from os.path import join
from functools import partial
from typing import Union, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer
from steeredmarigold.callables import *
from steeredmarigold.constants import *
from steeredmarigold.average import ValidationResult
from steeredmarigold.utils import float_depth_to_8bit_bgr, save_arguments_to_json, parse_val_args, ARGS_FILE, \
    getattr_from_string, vae_encode, vae_decode
from steeredmarigold.steering import make_interpolated_steering

class SteeredDDPMScheduler(DDPMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        condition: numpy.ndarray,
        condition_mask: numpy.ndarray,
        condition_align: numpy.ndarray,
        vae: AutoencoderKL,
        steering_factor: float,
        neighborhood: int,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # Compute Steering Direction
        x0_depth = vae_decode(vae, pred_original_sample)
        x0_depth = tensor_to_single_channel_depth(x0_depth)

        steering = make_interpolated_steering(x0_depth, condition, condition_mask, condition_align, neighborhood)
        steering = NumpyOneToThreeChannels()(steering)
        steering = NumpyRgbToTensorCHW()(steering)
        steering = torch.unsqueeze(steering, 0)
        
        steering_encoded = vae_encode(vae, steering.to("cuda"))

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # See formula (11) from https://arxiv.org/pdf/2409.10202.pdf
        pred_prev_sample += steering_factor * (beta_prod_t ** 0.5) * (steering_encoded - pred_original_sample)

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


def rescale_to_0_1(data: numpy.ndarray) -> numpy.ndarray:
    return (data + 1.0) / 2.0


def tensor_to_single_channel_depth(tensor: torch.Tensor) -> numpy.ndarray:
    data = tensor.cpu().numpy()
    data = numpy.squeeze(data, axis=0)
    data = numpy.average(data, axis=0)
    data = numpy.clip(data, -1.0, 1.0)
    return data


def save_condition(condition: torch.Tensor | numpy.ndarray, path: str) -> None:
    data = tensor_to_single_channel_depth(condition)
    data = rescale_to_0_1(data)
    cv2.imwrite(path, float_depth_to_8bit_bgr(data))


def visualize_one_channel_depth(depth: numpy.ndarray, path:str) -> None:
    depth = rescale_to_0_1(depth)
    cv2.imwrite(path, float_depth_to_8bit_bgr(depth))


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
        subfolder="unet", 
        in_channels=8,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    ).to(device, dtype=weight_dtype)

    scheduler = SteeredDDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

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

            sample_data_id = sample_data[SAMPLE_ID]

            print(sample_data_id)

            conditioning_latents = sample_data[RGB_TENSOR_1_1]
            conditioning_latents = torch.unsqueeze(conditioning_latents, 0).to(device)            
            conditioning_latents = vae_encode(vae, conditioning_latents)

            lidar_meters = sample_data[LIDAR_METERS]
            lidar_meters_mask = sample_data[LIDAR_METERS_MASK]

            prediction_latents = torch.randn(conditioning_latents.shape, device=device, dtype=weight_dtype)
            scheduler.set_timesteps(args.denoising_steps, device=device)
            timesteps = scheduler.timesteps 

            for timestep_idx in range(len(timesteps)):                
                unet_input = torch.cat((conditioning_latents, prediction_latents), 1)
                noise_pred = unet(unet_input, timesteps[timestep_idx], encoder_hidden_states=encoder_hidden_states).sample
                prediction_latents = scheduler.step(noise_pred, timesteps[timestep_idx], prediction_latents, lidar_meters, 
                                                    lidar_meters_mask, args.align, vae, args.steering_factor, args.neighborhood).prev_sample

            prediction = vae_decode(vae, prediction_latents)
            prediction = tensor_to_single_channel_depth(prediction)
            prediction = rescale_to_0_1(prediction)
            prediction = prediction.astype(numpy.float32)
            
            if args.store_raw:
                result_path = join(args.output_dir, sample_data_id + ".tif")
                cv2.imwrite(result_path, prediction)

            if args.save_visualizations:
                result_path = join(args.output_dir, sample_data_id + "_colored.png")
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
