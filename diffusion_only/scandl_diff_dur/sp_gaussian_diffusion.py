"""
This code is adapted from Gong et al.'s 2023 DiffuSeq Model: https://github.com/Shark-NLP/DiffuSeq
"""

import math
import numpy as np
import torch as th
import sys
import os
import torch.nn

from .utils.nn import mean_flat

sys.path.append('.')



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        one_noise_step,
        nll_in_loss,
        mask_padding,
        rescale_timesteps=False,
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl
        self.one_noise_step = one_noise_step
        self.nll_in_loss = nll_in_loss
        self.mask_padding = mask_padding

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)  # shape [diffusion_steps]
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # will approximate 0
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # shifted one to the right
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)   # shifted one to the left
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mapping_func = None # implement in train main()
        self.add_mask_noise = False # TODO


    def training_losses(
            self,
            model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  # mu * x_0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)  #  sd * noise
            * noise
        )

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask==0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            model,
            x_fix_loc,
            x_fix_dur,
            sn_input_ids_emb,
            pos_enc,
            mask_sn_padding,
            mask_transformer_att,
            t,
            clip_denoised=True,
            denoised_fn_fix_loc=None,
            denoised_fn_fix_dur=None,
            model_kwargs=None,
            subwords_list=None,
            atten_vis=None,
            atten_vis_fn=None,
            atten_vis_path=None,
            batch_idx=None,
            rank=None,
            atten_vis_sp=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t; the noised input where the embedded condition/sn was not noised
                    and the target/sp was replaced with Gaussian noise; at time step t
        :param sn_input_ids_emb: the BERT embeddings
        :param pos_enc: the positional embeddings
        :param mask_sn_padding:
        :param mask_transformer_att: the attention mask for the transformer
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample.
                            Applies before clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :param subwords_list: list of list containing the subwortokens of each instance (for attention visualization)
        :param atten_vis: bool: if True, attention is visualized (heatmaps)
        :param atten_vis_fn: the attention visualization function
        :param atten_vis_path: the path where to save the heatmaps to
        :param batch_idx: index of the current batch
        :paran rank: if parallel processing, the GPU index
        :param atten_vis_sp: bool: save attention scores of the last timestep
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_fix_loc.size(0), x_fix_loc.size(-1)
        assert t.shape == (B,)

        if not atten_vis and not atten_vis_sp:
            model_output_fix_loc = model(
                x=x_fix_loc,
                ts=self._scale_timesteps(t),
                sn_input_ids_emb=sn_input_ids_emb,
                pos_enc=pos_enc,
                attention_mask=mask_transformer_att,
                **model_kwargs,
            )
            model_output_fix_dur = model(
                x=x_fix_dur,
                ts=self._scale_timesteps(t),
                sn_input_ids_emb=sn_input_ids_emb,
                pos_enc=pos_enc,
                attention_mask=mask_transformer_att,
                **model_kwargs,
            )
        # TODO did not finish adapting the attention to both fix loc and fix dur output
        # else:
        #     # visualise the attention (heatmaps)
        #     if atten_vis:
        #         model_output_fix_loc, attention_scores_fix_loc = model(
        #             x=x_fix_loc,
        #             ts=self._scale_timesteps(t),
        #             sn_input_ids_emb=sn_input_ids_emb,
        #             pos_enc=pos_enc,
        #             attention_mask=mask_transformer_att,
        #             atten_vis=atten_vis,
        #             **model_kwargs,
        #         )
        #         # visualise attention for last denoising step
        #         if t[0] % 200 == 0:
        #
        #             atten_vis_fn(
        #                 attention_scores=attention_scores_fix_loc,
        #                 subwords_list=subwords_list,
        #                 batch_idx=batch_idx,
        #                 path_to_dir=atten_vis_path,
        #                 denoising_step=t[0].item(),
        #                 aggregate=True,
        #                 rank=rank,
        #             )
        #     if atten_vis_sp:
        #
        #         model_output, attention_scores = model(
        #             x=x,
        #             ts=self._scale_timesteps(t),
        #             sn_input_ids_emb=sn_input_ids_emb,
        #             pos_enc=pos_enc,
        #             attention_mask=mask_transformer_att,
        #             atten_vis=True,
        #             **model_kwargs,
        #         )
        #         if t[0] == 0:
        #
        #             out_path_heatmaps_sp = os.path.join(atten_vis_path, 'heatmaps_sps')
        #             if not os.path.exists(out_path_heatmaps_sp):
        #                 os.makedirs(out_path_heatmaps_sp)
        #             filename = f'att_scores_rank{rank}_batch{batch_idx}.pt'
        #             path_to_file = os.path.join(out_path_heatmaps_sp, filename)
        #             torch.save(attention_scores, path_to_file)

        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        model_variance = _extract_into_tensor(model_variance, t, x_fix_loc.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_fix_loc.shape)

        # The denoised_fn is applied to x_start (the model output) before it is used for sampling
        def process_xstart_fix_loc(x):
            """ here x is the model output """
            if denoised_fn_fix_loc is not None:
                # print(denoised_fn)
                x = denoised_fn_fix_loc(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        def process_xstart_fix_dur(x):
            if denoised_fn_fix_dur is not None:
                x = denoised_fn_fix_dur(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            # the denoised fn is applied to the model output
            pred_xstart_fix_loc = process_xstart_fix_loc(model_output_fix_loc)
            pred_xstart_fix_dur = process_xstart_fix_dur(model_output_fix_dur)
        else:
            ### model is used to predict eps
            pred_xstart_fix_loc = process_xstart_fix_loc(
                self._predict_xstart_from_eps(x_t=x_fix_loc, t=t, eps=model_output_fix_loc)
            )
            pred_xstart_fix_dur = process_xstart_fix_dur(
                self._predict_xstart_from_eps(x_t=x_fix_dur, t=t, eps=model_output_fix_dur)
            )

        # this is the mean of the posterior distribution q(x_{t-1} | x_t, x_0), estimated from x_t, which is the noised
        # input, and pred_xstart, which is what the model predicted to be x_0 from the noised input x_noised/x_t
        model_mean_fix_loc, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart_fix_loc, x_t=x_fix_loc, t=t
        )
        model_mean_fix_dur, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart_fix_dur, x_t=x_fix_dur, t=t
        )

        assert (
            model_mean_fix_loc.shape == model_log_variance.shape == pred_xstart_fix_loc.shape == x_fix_loc.shape
        )
        assert (
                model_mean_fix_dur.shape == model_log_variance.shape == pred_xstart_fix_dur.shape == x_fix_dur.shape
        )
        return {
            "mean_fix_loc": model_mean_fix_loc,
            'mean_fix_dur': model_mean_fix_dur,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart_fix_loc": pred_xstart_fix_loc,
            'pred_xstart_fix_dur': pred_xstart_fix_dur,
        }

    def p_sample(
            self,
            model,
            x_fix_loc,
            x_fix_dur,
            sn_input_ids_emb,
            pos_enc,
            mask_sn_padding,
            mask_transformer_att,
            t,
            clip_denoised=True,
            denoised_fn_fix_loc=None,
            denoised_fn_fix_dur=None,
            model_kwargs=None,
            top_p=None,
            mask=None,
            x_start=None,
            subwords_list=None,
            atten_vis=None,
            atten_vis_fn=None,
            atten_vis_path=None,
            batch_idx=None,
            rank=None,
            atten_vis_sp=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from; the transformer model that learned the denoising
        :param x: the current tensor at x_{t-1}.
        :param sn_input_ids_emb: the BERT embeddings
        :param pos_enc: the positional embeddings
        :param mask_sn_padding:
        :param mask_transformer_att: the attention mask for the transformer
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample.
        :param top_p:
        :param mask: anchoring masked position to x_start
        :param x_start:
        :param subwords_list: list of list containing the subwortokens of each instance (for attention visualization)
        :param atten_vis: bool: if True, attention is visualized (heatmaps)
        :param atten_vis_fn: the attention visualization function
        :param atten_vis_path: the path where to save the heatmaps to
        :param batch_idx: index of the current batch
        :paran rank: if parallel processing, the GPU index
        :param atten_vis_sp: bool: save attention scores of the last timestep
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model=model,
            x_fix_loc=x_fix_loc,
            x_fix_dur=x_fix_dur,
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            mask_sn_padding=mask_sn_padding,
            mask_transformer_att=mask_transformer_att,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn_fix_loc=denoised_fn_fix_loc,
            denoised_fn_fix_dur=denoised_fn_fix_dur,
            model_kwargs=model_kwargs,
            subwords_list=subwords_list,
            atten_vis=atten_vis,
            atten_vis_fn=atten_vis_fn,
            atten_vis_path=atten_vis_path,
            batch_idx=batch_idx,
            rank=rank,
            atten_vis_sp=atten_vis_sp,
        )

        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x_fix_loc)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x_fix_loc)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_fix_loc.shape) - 1)))
        )  # no noise when t == 0

        sample_fix_loc = out["mean_fix_loc"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        sample_fix_dur = out['mean_fix_dur'] + nonzero_mask * th.exp(0.5 * out['log_variance']) * noise

        if mask == None:
            pass
        else:
            # the original embedding for the sn, and the predicted sample for the sp
            sample_fix_loc = th.where(mask==0, x_start, sample_fix_loc)
            sample_fix_dur = th.where(mask == 0, x_start, sample_fix_dur)

        return {
            "sample_fix_loc": sample_fix_loc,
            "sample_fix_dur": sample_fix_dur,
            "pred_xstart_fix_loc": out["pred_xstart_fix_loc"],
            "pred_xstart_fix_dur": out["pred_xstart_fix_dur"],
            #"greedy_mean_fix_loc": out["mean_fix_loc"],
            #"greedy_mean_fix_dur": out["mean_fix_dur"],
            "mean_fix_loc": out["mean_fix_loc"],
            "mean_fix_dur": out["mean_fix_dur"],
            "out": out,
        }


    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        sn_input_ids_emb=None,
        pos_enc=None,
        mask_sn_padding=None,
        mask_transformer_att=None,
        clip_denoised=True,
        denoised_fn_fix_loc=None,
        denoised_fn_fix_dur=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        subwords_list=None,
        atten_vis=None,
        atten_vis_fn=None,
        atten_vis_path=None,
        batch_idx=None,
        gap=1,
        rank=None,
        atten_vis_sp=None,
        eta=None,  # used only for ddim sampling
        langevin_fn=None,  # used only for ddim sampling
    ):
        """
        Generate samples from the model.

        :param model: the transformer model that was trained to learn the denoising
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: the Gaussian noise that should be denoised at inference (the replaced word ID emb)
        :param sn_input_ids_emb: the BERT embeddings
        :param pos_enc: the positional embeddings
        :param mask_sn_padding:
        :param mask_transformer_att: the attention mask for the transformer
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn_fix_loc: if not None, a function applied to the x_start prediction of the fixation locations before it is used to sample (projected back to discrete space and back to continuous)
        :param denoised_fn_fix_dur: if not None, a function applied to the x_start prediction of the fixation durations before it is used to sample (projected back to discrete space and back to continuous)
        :param model_kwargs: if not None, a dict of extra keyword arguments to  pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on. If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param top_p:
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param mask: anchoring masked position to x_start
        :param x_start: the summation of sn_sp_emb and fix_dur_emb before the scanpath part is replaced by noise
        :param subwords_list: list of list containing the subwortokens of each instance (for attention visualization)
        :param atten_vis: bool: if True, attention is visualized (heatmaps)
        :param atten_vis_fn: the attention visualization function
        :param atten_vis_path: the path where to save the heatmaps to
        :param batch_idx: index of the current batch
        :param gap:
        :paran rank: if parallel processing, the GPU index
        :param atten_vis_sp: bool: save attention scores of the last timestep
        :return: a non-differentiable batch of samples.
        """
        final_fix_loc = []
        final_fix_dur = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            mask_sn_padding=mask_sn_padding,
            mask_transformer_att=mask_transformer_att,
            clip_denoised=clip_denoised,
            denoised_fn_fix_loc=denoised_fn_fix_loc,
            denoised_fn_fix_dur=denoised_fn_fix_dur,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start,
            subwords_list=subwords_list,
            atten_vis=atten_vis,
            atten_vis_fn=atten_vis_fn,
            atten_vis_path=atten_vis_path,
            batch_idx=batch_idx,
            rank=rank,
            atten_vis_sp=atten_vis_sp,
        ):
            final_fix_loc.append(sample['sample_fix_loc'])
            final_fix_dur.append(sample['sample_fix_dur'])
        return final_fix_loc, final_fix_dur

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        sn_input_ids_emb=None,
        pos_enc=None,
        mask_sn_padding=None,
        mask_transformer_att=None,
        clip_denoised=True,
        denoised_fn_fix_loc=None,
        denoised_fn_fix_dur=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        subwords_list=None,
        atten_vis=None,
        atten_vis_fn=None,
        atten_vis_path=None,
        batch_idx=None,
        rank=None,
        atten_vis_sp=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # noise/sample_x is the input that was noised: the concatenated sn-sp embedding where the sp was completely
        # replaced with Gaussian noise from the standard normal distribution
        if noise is not None:
            sample_x_fix_loc = noise
            sample_x_fix_dur = noise
        else:
            sample_x_fix_loc = th.randn(*shape, device=device)
            sample_x_fix_dur = th.randn(*shape, device=device)

        # the number of diffusion steps in reverse order
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # denoising from the number of diffusion steps T to t=0
        for i in indices: # from T to 0

            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur_fix_loc = None
                    denoised_fn_cur_fix_dur = None
                else:
                    denoised_fn_cur_fix_loc = denoised_fn_fix_loc
                    denoised_fn_cur_fix_dur = denoised_fn_fix_dur
            else:
                if i >= clamp_step:
                    denoised_fn_cur_fix_loc = denoised_fn_fix_loc
                    denoised_fn_cur_fix_dur = denoised_fn_fix_dur
                else:
                    denoised_fn_cur_fix_loc = None
                    denoised_fn_cur_fix_dur = None

            with th.no_grad():
                out = self.p_sample(
                    model=model,
                    x_fix_loc=sample_x_fix_loc,
                    x_fix_dur=sample_x_fix_dur,
                    sn_input_ids_emb=sn_input_ids_emb,
                    pos_enc=pos_enc,
                    mask_sn_padding=mask_sn_padding,
                    mask_transformer_att=mask_transformer_att,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn_fix_loc=denoised_fn_cur_fix_loc,
                    denoised_fn_fix_dur=denoised_fn_cur_fix_dur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    subwords_list=subwords_list,
                    x_start=x_start,
                    atten_vis=atten_vis,
                    atten_vis_fn=atten_vis_fn,
                    atten_vis_path=atten_vis_path,
                    batch_idx=batch_idx,
                    rank=rank,
                    atten_vis_sp=atten_vis_sp,
                )
                yield out
                sample_x_fix_loc = out["sample_fix_loc"]
                sample_x_fix_dur = out["sample_fix_dur"]


    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )

    def _get_x_start_fixdur(self, sn_sp_emb, fix_dur_emb, std):
        """
        Summing up of fix loc and fix dur embeddings; adding one step of noise (from x to z_0)
        """
        noise = th.randn_like(sn_sp_emb)
        assert noise.shape == sn_sp_emb.shape
        return (
            sn_sp_emb + fix_dur_emb + std * noise
        )

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  #  shape [microbatch size, seq_len, vocabulary]
        # print(logits.shape)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll


    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def training_losses_seq2seq(
            self,
            model,  # the transformer model
            t,  # the number of noise adding steps for each instance in the microbatch
            sn_sp_repr,
            mask,
            sn_input_ids,
            indices_pos_enc,
            sn_sp_fix_dur,
            mask_sn_padding,
            mask_transformer_att,
            noise=None,
    ):
        """
        Compute training losses for a single timestep.

        :param model: the transformer model
        :param t: a batch of timestep indices.
        :param sn_sp_repr:  the word IDs of sn and sp
        :param mask:  masking the sn
        :param sn_input_ids:  the tokenizer input IDs
        :param indices_pos_enc: the indices for pos enc
        :param_mask_sn_padding:
        :param mask_transformer_att: the transformer att
        :param model_kwargs: if not None, a dict of extra keyword arguments to  pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        microbatch_size, seq_len = sn_sp_repr.shape

        # get the word ID embedding, BERT embedding, positional embedding
        sn_sp_emb, pos_enc, sn_input_ids_emb, fix_dur_emb = model.model.module.get_embeds(  # TODO removed fix_dur_emb
            sn_sp_repr=sn_sp_repr,
            sn_input_ids=sn_input_ids,
            indices_pos_enc=indices_pos_enc,
            sn_sp_fix_dur=sn_sp_fix_dur,
        )

        # get the standard deviation, shape [microbatch, args.seq_len, hidden_size=768]
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(sn_sp_emb.device),
                                   sn_sp_emb.shape)

        # map sn_sp_emb to x_start, which is a one-step noised sn_sp_emb (in paper it's z_0)
        if self.one_noise_step:  # this should always be true actually (without it performance is bad)
           # x_start = self._get_x_start(sn_sp_emb, std)

            # now two embeddings will have to be denoised; we sum them up and add one noise step (projection x -> z_0)
            x_start = self._get_x_start_fixdur(
                sn_sp_emb=sn_sp_emb,
                fix_dur_emb=fix_dur_emb,
                std=std,
            )
        else:
            x_start = sn_sp_emb

        # sample noise in the same shape as our input
        if noise is None:
            noise = th.randn_like(x_start)

        # get the noised sample x_t, which is still of shape [microbatch, args.seq_len, hidden_size=768]
        # the condition/sn is not noised (hence the input mask)
        # each instance in the microbatch receives a different amount of noise (t noising steps, as given in vector t)
        x_t = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
            mask=mask,
        )

        terms = {}

        target = x_start

        # model_output is of shape [microbatch, args.seq_len, emb_dim=768]
        model_output = model(
            x=x_t,
            ts=self._scale_timesteps(t),
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            attention_mask=mask_transformer_att,
        )
        assert model_output.shape == target.shape == x_start.shape

        # Loss 1: Mean Squared Error (MSE) (L_{VLB})
        terms["mse"] = mean_flat((target - model_output) ** 2)
        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']
        t0_mask = (t == 0) # mask that says true for every instance where no noise was received, i.e. t=0
        # MSE between the model output and the embedded input before the one noise step
        t0_loss = mean_flat((sn_sp_emb - model_out_x_start) ** 2)
        # update the MSE between the model output and the one-step noised input embeddings with the MSE between the
        # model output and the embeddings before the one noise step wherever there was no noise received in the noising
        # process (i.e., wherever t was 0)
        terms['mse'] = th.where(t0_mask, t0_loss, terms['mse'])

        # Loss 2: L_{round}
        out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        tT_loss = mean_flat(out_mean ** 2)

        # for the NLL losses, we need to convert the model output into logits
        get_logits = model.model.module.get_logits

        # Loss 3: L_{EMB}
        # compute the NLL between the one-noised embeddings and the initial representation (word IDs)
        # embedding regularisation
        decoder_nll = self._token_discrete_loss(x_start, get_logits, sn_sp_repr)

        # # Loss 4: L_{EMB-fix-dur}
        get_logits_fix_dur = model.model.module.get_logits_fix_dur
        decoder_nll_fix_dur = self._token_discrete_loss(x_start, get_logits_fix_dur, sn_sp_fix_dur)

        # unused Loss
        terms['nll'] = self._token_discrete_loss(model_output, get_logits, sn_sp_repr, mask=mask)

        # combined loss
        if self.nll_in_loss:  # should be False; model performance drops if nll included
            terms['loss'] = terms['mse'] + tT_loss + decoder_nll + terms['nll']
        else:
            terms['loss'] = terms['mse'] + tT_loss + decoder_nll + decoder_nll_fix_dur

        return terms


    def ddim_sample(
        self,
        model,
        #x,
        x_fix_loc,
        x_fix_dur,
        sn_input_ids_emb,
        pos_enc,
        mask_sn_padding,
        mask_transformer_att,
        t,
        clip_denoised=True,
        denoised_fn_fix_loc=None,
        denoised_fn_fix_dur=None,
        model_kwargs=None,
        eta=None,
        langevin_fn=None,
        mask=None,
        x_start=None,
        subwords_list=None,
        atten_vis=None,
        atten_vis_fn=None,
        atten_vis_path=None,
        batch_idx=None,
        rank=None,
        atten_vis_sp=None,
        top_p=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            #x,
            x_fix_loc=x_fix_loc,
            x_fix_dur=x_fix_dur,
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            mask_sn_padding=mask_sn_padding,
            mask_transformer_att=mask_transformer_att,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn_fix_loc=denoised_fn_fix_loc,
            denoised_fn_fix_dur=denoised_fn_fix_dur,
            model_kwargs=model_kwargs,
            subwords_list=subwords_list,
            atten_vis=atten_vis,
            atten_vis_fn=atten_vis_fn,
            atten_vis_path=atten_vis_path,
            batch_idx=batch_idx,
            rank=rank,
            atten_vis_sp=atten_vis_sp,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        #eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        eps_fix_loc = self._predict_eps_from_xstart(x_t=x_fix_loc, t=t, pred_xstart=out["pred_xstart_fix_loc"])
        eps_fix_dur = self._predict_eps_from_xstart(x_t=x_fix_dur, t=t, pred_xstart=out["pred_xstart_fix_dur"])

        # alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        # sigma = (
        #     eta
        #     * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        #     * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        # )
        alpha_bar_fix_loc = _extract_into_tensor(self.alphas_cumprod, t, x_fix_loc.shape)
        alpha_bar_fix_dur = _extract_into_tensor(self.alphas_cumprod, t, x_fix_dur.shape)
        alpha_bar_prev_fix_loc = _extract_into_tensor(self.alphas_cumprod_prev, t, x_fix_loc.shape)
        alpha_bar_prev_fix_dur = _extract_into_tensor(self.alphas_cumprod_prev, t, x_fix_dur.shape)

        sigma_fix_loc = (
            eta
            * th.sqrt((1 - alpha_bar_prev_fix_loc) / (1 - alpha_bar_fix_loc))
            * th.sqrt(1 - alpha_bar_fix_loc / alpha_bar_prev_fix_loc)
        )
        sigma_fix_dur = (
            eta
            * th.sqrt((1 - alpha_bar_prev_fix_dur) / (1 - alpha_bar_fix_dur))
            * th.sqrt(1 - alpha_bar_fix_dur / alpha_bar_prev_fix_dur)
        )


        # Equation 12.
        assert x_fix_loc.shape == x_fix_dur.shape
        noise = th.randn_like(x_fix_loc)
        # mean_pred = (
        #     out["pred_xstart"] * th.sqrt(alpha_bar_prev)
        #     + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        # )
        mean_pred_fix_loc = (
            out["pred_xstart_fix_loc"] * th.sqrt(alpha_bar_prev_fix_loc)
            + th.sqrt(1 - alpha_bar_prev_fix_loc - sigma_fix_loc ** 2) * eps_fix_loc
        )
        mean_pred_fix_dur = (
            out["pred_xstart_fix_dur"] * th.sqrt(alpha_bar_prev_fix_dur)
            + th.sqrt(1 - alpha_bar_prev_fix_dur - sigma_fix_dur ** 2) * eps_fix_dur
        )
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_fix_loc.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        #sample = mean_pred + nonzero_mask * sigma * noise
        sample_fix_loc = mean_pred_fix_loc + nonzero_mask * sigma_fix_loc * noise
        sample_fix_dur = mean_pred_fix_dur + nonzero_mask * sigma_fix_dur * noise

        if langevin_fn:
            print(t.shape)
            #sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
            sample_fix_loc = langevin_fn(sample_fix_loc, mean_pred_fix_loc, sigma_fix_loc, self.alphas_cumprod_prev[t[0]], t, x_fix_loc)
            sample_fix_dur = langevin_fn(sample_fix_dur, mean_pred_fix_dur, sigma_fix_dur, self.alphas_cumprod_prev[t[0]], t, x_fix_dur)

        if mask == None:
            pass
        else:
            #sample = th.where(mask==0, x_start, sample)
            sample_fix_loc = th.where(mask == 0, x_start, sample_fix_loc)
            sample_fix_dur = th.where(mask == 0, x_start, sample_fix_dur)

        #return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        return {
            "sample_fix_loc": sample_fix_loc,
            "sample_fix_dur": sample_fix_dur,
            "pred_xstart_fix_loc": out["pred_xstart_fix_loc"],
            "pred_xstart_fix_dur": out["pred_xstart_fix_dur"],
            "mean_fix_loc": out["mean_fix_loc"],
            "mean_fix_dur": out["mean_fix_dur"],
            "out": out,
        }

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        sn_input_ids_emb=None,
        pos_enc=None,
        mask_sn_padding=None,
        mask_transformer_att=None,
        clip_denoised=True,
        denoised_fn_fix_loc=None,
        denoised_fn_fix_dur=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        subwords_list=None,
        atten_vis=None,
        atten_vis_fn=None,
        atten_vis_path=None,
        batch_idx=None,
        gap=1,
        eta=0.0,
        rank=None,
        atten_vis_sp=None,
        langevin_fn=None,
    ):
        """
        Generate samples from the model using DDIM (Denoising Diffusion Implicit Models).
        :param model: the transformer model that was trained to learn the denoising
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: the Gaussian noise that should be denoised at inference: the unnoised sentence and the noise replacing the fix dur and fix loc embeddings
        :param sn_input_ids_emb: the BERT embeddings
        :param pos_enc: the positional embeddings
        :param mask_sn_padding: 
        :param mask_transformer_att: the attention mask for the transformer
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn_fix_loc: if not None, a function applied to the x_start prediction of the fixation locations before it is used to sample (projected back to discrete space and back to continuous)
        :param denoised_fn_fix_dur: if not None, a function applied to the x_start prediction of the fixation durations before it is used to sample (projected back to discrete space and back to continuous)
        :param model_kwargs: if not None, a dict of extra keyword arguments to  pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on. If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param top_p:
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param mask: anchoring masked position to x_start
        :param x_start: the summation of sn_sp_emb and fix_dur_emb before the scanpath part is replaced by noise
        :param subwords_list: list of list containing the subwortokens of each instance (for attention visualization)
        :param atten_vis: bool: if True, attention is visualized (heatmaps)
        :param atten_vis_fn: the attention visualization function
        :param atten_vis_path: the path where to save the heatmaps to
        :param batch_idx: index of the current batch
        :param gap: compute ddim sampling for each {gap} step
        :param eta: the noise level for the reverse ODE  # TODO is this true?
        :param rank: if parallel processing, the GPU index
        :param atten_vis_sp: bool: save attention scores of the last timestep
        :param langevin_fn: the langevin dynamics function
        :return: a non-differentiable batch of samples.

        Same usage as p_sample_loop().
        """
        #final = []
        final_fix_dur = []
        final_fix_loc = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            mask_sn_padding=mask_sn_padding,
            mask_transformer_att=mask_transformer_att,
            clip_denoised=clip_denoised,
            denoised_fn_fix_loc=denoised_fn_fix_loc,
            denoised_fn_fix_dur=denoised_fn_fix_dur,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            eta=eta,
            langevin_fn=langevin_fn,
            mask=mask,
            x_start=x_start,
            gap = gap,
            subwords_list=subwords_list,
            atten_vis=atten_vis,
            atten_vis_fn=atten_vis_fn,
            atten_vis_path=atten_vis_path,
            batch_idx=batch_idx,
            rank=rank,
            atten_vis_sp=atten_vis_sp,
            top_p=top_p,
        ):
            #final.append(sample['sample'])
            final_fix_loc.append(sample['sample_fix_loc'])
            final_fix_dur.append(sample['sample_fix_dur'])
        return final_fix_loc, final_fix_dur

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        sn_input_ids_emb=None,
        pos_enc=None,
        mask_sn_padding=None,
        mask_transformer_att=None,
        clip_denoised=True,
        denoised_fn_fix_loc=None,
        denoised_fn_fix_dur=None,
        model_kwargs=None,
        device=None,
        progress=False,
        clamp_step=None,
        clamp_first=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1,
        subwords_list=None,
        atten_vis=None,
        atten_vis_fn=None,
        atten_vis_path=None,
        batch_idx=None,
        rank=None,
        atten_vis_sp=None,
        top_p=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        # if noise/sample_x is the input that was noised: the concatenated sn-sp embedding where the sp was completely
        # replaced with Gaussian noise from the standard normal distribution
        if noise is not None:
            #sample_x = noise
            sample_x_fix_loc = noise
            sample_x_fix_dur = noise
        else:
            #sample_x = th.randn(*shape, device=device)
            sample_x_fix_loc = th.randn(*shape, device=device)
            sample_x_fix_dur = th.randn(*shape, device=device)

        # the number of diffusion steps in reverse order
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # denoising from the number of diffusion steps T to t=0 (with ddim sampling)
        for i in indices:

            t = th.tensor([i] * shape[0], device=device)

            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur_fix_loc = None
                    denoised_fn_cur_fix_dur = None
                else:
                    denoised_fn_cur_fix_loc = denoised_fn_fix_loc
                    denoised_fn_cur_fix_dur = denoised_fn_fix_dur
            else:
                if i >= clamp_step:
                    denoised_fn_cur_fix_loc = denoised_fn_fix_loc
                    denoised_fn_cur_fix_dur = denoised_fn_fix_dur
                else:
                    denoised_fn_cur_fix_loc = None
                    denoised_fn_cur_fix_dur = None

            with th.no_grad():
                out = self.ddim_sample(
                    model=model,
                    x_fix_loc=sample_x_fix_loc,
                    x_fix_dur=sample_x_fix_dur,
                    sn_input_ids_emb=sn_input_ids_emb,
                    pos_enc=pos_enc,
                    mask_sn_padding=mask_sn_padding,
                    mask_transformer_att=mask_transformer_att,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn_fix_loc=denoised_fn_cur_fix_loc,
                    denoised_fn_fix_dur=denoised_fn_cur_fix_dur,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start,
                    eta=eta,
                    langevin_fn=langevin_fn,
                    subwords_list=subwords_list,
                    atten_vis=atten_vis,
                    atten_vis_fn=atten_vis_fn,
                    atten_vis_path=atten_vis_path,
                    batch_idx=batch_idx,
                    rank=rank,
                    atten_vis_sp=atten_vis_sp,
                    top_p=top_p,
                )
                yield out
                #sample_x = out["sample"]
                sample_x_fix_loc = out["sample_fix_loc"]
                sample_x_fix_dur = out["sample_fix_dur"]

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # print(kwargs.keys())
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called training_losses')
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        # print(ts)
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        # print(new_ts)
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        # temp = self.model(x, new_ts, **kwargs)
        # print(temp.shape)
        # return temp
        # print(new_ts)
        return self.model(x, new_ts, **kwargs)
