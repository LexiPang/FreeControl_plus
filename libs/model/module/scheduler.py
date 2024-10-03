from typing import Optional, Tuple, Union

import torch
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomDDIMScheduler(DDIMScheduler):
    @torch.no_grad()
    def step(
            self,
            model_output: torch.FloatTensor,
            timestep_idx: int,
            sample: torch.FloatTensor,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator=None,
            variance_noise: Optional[torch.FloatTensor] = None,
            return_dict: bool = True,

            # Guidance parameters
            score=None,
            guidance_scale=0.0,
            indices=None,

    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # Support IF models
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. get previous step value (=t-1)
        # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        # alpha_prod_t = self.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[self.timesteps[timestep_idx].cpu().numpy()]
        alpha_prod_t_prev = self.alphas_cumprod[self.timesteps[timestep_idx + 1].cpu().numpy()]
        # print(f"current timestep{self.timesteps[timestep_idx]}")
        # print(f"computed previous timestep{self.timesteps[timestep_idx + 1]}")

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        # 如果model是预测x0到xt的噪声，那么model_output就是noise_pred，即pred_epsilon，x0预测值（pred_original_sample）是用noise_pred算出来的
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self._get_variance(timestep, prev_timestep)
        # std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. apply guidance following the formula (14) from https://arxiv.org/pdf/2105.05233.pdf
        if score is not None and guidance_scale > 0.0:
            if indices is not None:
                assert pred_epsilon[indices].shape == score.shape, "pred_epsilon[indices].shape != score.shape"
                pred_epsilon[indices] = pred_epsilon[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score
            else:
                assert pred_epsilon.shape == score.shape
                pred_epsilon = pred_epsilon - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score
        # This is the structural guidance for FreeControl (based on the pca_loss)
        # disabling this will cause the control and appearance images to be the same

        # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        # eta=0，相当于没加噪声

        # 8. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance
        self.pred_epsilon = pred_epsilon
        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def add_noise_between_t(
        self,
        original_samples: torch.FloatTensor,
        timesteps_1: torch.IntTensor,
        timesteps_2: torch.IntTensor,
        generator,
        S_noise=1.0
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples



        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps_1 = timesteps_1.to(original_samples.device)
        timesteps_2 = timesteps_2.to(original_samples.device)

        alpha_prod_1 = alphas_cumprod[timesteps_1].flatten()
        alpha_prod_2 = alphas_cumprod[timesteps_2].flatten()

        # print(f"add noise {timesteps_1} -> {timesteps_2}", torch.sqrt((1 - alpha_prod_1) / alpha_prod_1), '->',
        #       torch.sqrt((1 - alpha_prod_2) / alpha_prod_2))
        while len(alpha_prod_1.shape) < len(original_samples.shape):
            alpha_prod_1 = alpha_prod_1.unsqueeze(-1)

        alpha_prod_2 = alpha_prod_2.view_as(alpha_prod_1)

        #noise = torch.randn_like(original_samples)
        noise = randn_tensor(
                    original_samples.shape, generator=generator, device=original_samples.device, dtype=original_samples.dtype
                )

        factor = alpha_prod_2 / alpha_prod_1
        noisy_samples = torch.sqrt(factor) * original_samples \
                        + torch.sqrt(1 - factor) * noise * S_noise
        return noisy_samples


class CustomEulerDiscreteScheduler(EulerDiscreteScheduler):
    @torch.no_grad()
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep_idx: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        
        # Guidance parameters
        score=None,
        guidance_scale=0.0,
        indices=None,
            
    ) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
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
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(self.timesteps[timestep_idx], int)
            or isinstance(self.timesteps[timestep_idx], torch.IntTensor)
            or isinstance(self.timesteps[timestep_idx], torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if self.step_index is None:
            self._init_step_index(self.timesteps[timestep_idx])

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - model_output) / sigma_hat
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
            pred_epsilon = model_output
        elif self.config.prediction_type == "v_prediction":
            # denoised = model_output * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            pred_epsilon = (sample - (model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1)))) / sigma_hat
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )
        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat
        
        # Perform structural guidance
        alpha_prod_t = self.alphas_cumprod[self.timesteps[timestep_idx].cpu().numpy()]
        if score is not None and guidance_scale > 0.0:
            if indices is not None:
                assert derivative[indices].shape == score.shape, "derivative[indices].shape != score.shape"
                derivative[indices] = derivative[indices] - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score
            else:
                assert pred_epsilon.shape == score.shape
                derivative = derivative - guidance_scale * (1 - alpha_prod_t) ** (0.5) * score
        
        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    def add_noise_between_t(
        self,
        original_samples: torch.FloatTensor,
        timesteps_1: torch.IntTensor,
        timesteps_2: torch.IntTensor,
        generator,
        S_noise=1.0
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps_1) and torch.is_floating_point(timesteps_2):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps_1 = timesteps_1.to(original_samples.device, dtype=torch.float32)
            timesteps_2 = timesteps_2.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps_1 = timesteps_1.to(original_samples.device)
            timesteps_2 = timesteps_2.to(original_samples.device)

        step_indices_1 = [(schedule_timesteps == t).nonzero().item() for t in timesteps_1]
        step_indices_2 = [(schedule_timesteps == t).nonzero().item() for t in timesteps_2]

        sigma_1 = sigmas[step_indices_1].flatten()
        sigma_2 = sigmas[step_indices_2].flatten()
        
        while len(sigma_1.shape) < len(original_samples.shape):
            sigma_1 = sigma_1.unsqueeze(-1)
            
        sigma_2 = sigma_2.view_as(sigma_1)
        
        noise = randn_tensor(
                    original_samples.shape, generator=generator, device=original_samples.device, dtype=original_samples.dtype
                )
        
        noisy_samples = original_samples + torch.sqrt(sigma_2 ** 2 - sigma_1 ** 2) * noise * S_noise

        # alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # timesteps_1 = timesteps_1.to(original_samples.device)
        # timesteps_2 = timesteps_2.to(original_samples.device)

        # alpha_prod_1 = alphas_cumprod[timesteps_1].flatten()
        # alpha_prod_2 = alphas_cumprod[timesteps_2].flatten()

        # # print(f"add noise {timesteps_1} -> {timesteps_2}", torch.sqrt((1 - alpha_prod_1) / alpha_prod_1), '->',
        # #       torch.sqrt((1 - alpha_prod_2) / alpha_prod_2))
        # while len(alpha_prod_1.shape) < len(original_samples.shape):
        #     alpha_prod_1 = alpha_prod_1.unsqueeze(-1)

        # alpha_prod_2 = alpha_prod_2.view_as(alpha_prod_1)

        # #noise = torch.randn_like(original_samples)
        # noise = randn_tensor(
        #             original_samples.shape, generator=generator, device=original_samples.device, dtype=original_samples.dtype
        #         )

        # factor = alpha_prod_2 / alpha_prod_1
        # noisy_samples = torch.sqrt(factor) * original_samples \
        #                 + torch.sqrt(1 - factor) * noise * S_noise
        return noisy_samples