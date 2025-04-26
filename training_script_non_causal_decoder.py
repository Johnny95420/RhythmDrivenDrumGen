# %%
import os

os.chdir("/home/practice02")

import copy
from random import seed, shuffle

import tqdm

seed(42)
import numpy as np

np.random.seed(42)
import torch

torch.manual_seed(42)
from functools import partial
from pathlib import Path

import numpy as np
import torch.nn as nn
import wandb
from accelerate import Accelerator
from data_aug import DrumCoditionRollDataset, roll_collate_fn
from model.layers import (
    BetaLatent,
    Decoder,
    Encoder,
    EncoderInputAdapter,
    LatentClassifier,
    VectorQuantizer,
)
from model.model import GANWrapper, VAEModelWrapper
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from utils.utils import discretize_cont_var, cosine_cyclical_annealing

# %%
# wandb.init(
#     project="drum conditional generation",
#     config={
#         "learning_rate": 3e-4,
#         "steps": 50000,
#         "batch_size": 128,
#     },
# )
TOTAL_STEPS = 25000
# %%
if __name__ == "__main__":
    processed_midi_paths = list(
        Path("/home/groove-v1.0.0-midionly/processed_midi_file_clip").glob("**/*.mid*")
    )
    shuffle(processed_midi_paths)
    l = len(processed_midi_paths)
    val_size, test_size = int(l * 0.2), int(l * 0.1)
    test_data, val_data, train_data = (
        processed_midi_paths[:test_size],
        processed_midi_paths[test_size : test_size + val_size],
        processed_midi_paths[test_size + val_size :],
    )
    train_dataset = DrumCoditionRollDataset(train_data, 4)
    val_dataset = DrumCoditionRollDataset(val_data, 4)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, collate_fn=roll_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=True, collate_fn=roll_collate_fn
    )
    # %%
    vae_model = VAEModelWrapper(
        encoder_input_emb=EncoderInputAdapter(
            input_dim=[9, 9, 9],
            hidden_dim=[32, 32, 32],
            d_model=32,
            dropout_vel=0.3,
            dropout_time=0.3,
            dropout=0.2,
            max_len=33,
        ),
        encoder=Encoder(
            d_model=32,
            nhead=4,
            dim_feedforward=256,
            num_encoder_layers=8,
            dropout=0.1,
        ),
        intermediate=BetaLatent(33, 32, 1024),
        decoder=Decoder(
            latent_dim=1024,
            d_model=32,
            num_decoder_layers=12,
            nhead=4,
            dim_feedforward=256,
            output_max_len=33,
            output_size=9,
            dropout=0.1,
            n_style=17,
            style_emb_dim=2,
            n_inst=9,
            inst_emb_dim=8,
        ),
    ).cuda()
    gan_model = GANWrapper(
        models=[
            LatentClassifier(1024, 10),
            LatentClassifier(1024, 10),
            LatentClassifier(1024, 17),
            LatentClassifier(1024, 9),
        ],
        preprocesseors=[
            partial(discretize_cont_var, min_val=-0.3, scale=0.6),
            partial(discretize_cont_var, min_val=-0.3, scale=0.6),
            None,
            None,
        ],
        loss_funcs=[
            nn.functional.cross_entropy,
            nn.functional.cross_entropy,
            nn.functional.cross_entropy,
            nn.functional.binary_cross_entropy_with_logits,
        ],
    ).cuda()
    # %%
    vae_params = [
        {"params": [], "weight_decay": 0.002},
        {"params": [], "weight_decay": 0},
    ]
    for name, para in vae_model.named_parameters():
        if "bias" in name or "embedding" in name or "norm" in name:
            vae_params[1]["params"].append(para)
        else:
            vae_params[0]["params"].append(para)
    gan_params = [
        {"params": [], "weight_decay": 0.002},
        {"params": [], "weight_decay": 0},
    ]
    for name, para in gan_model.named_parameters():
        if "bias" in name or "embedding" in name or "norm" in name:
            gan_params[1]["params"].append(para)
        else:
            gan_params[0]["params"].append(para)

    vae_optimizer = AdamW(vae_params, lr=3e-4)
    gan_optimizer = AdamW(gan_params, lr=5e-4)

    scheduler = get_cosine_schedule_with_warmup(vae_optimizer, 1000, TOTAL_STEPS)
    accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")
    (
        vae_model,
        gan_model,
        vae_optimizer,
        gan_optimizer,
        train_loader,
        val_loader,
        scheduler,
    ) = accelerator.prepare(
        vae_model,
        gan_model,
        vae_optimizer,
        gan_optimizer,
        train_loader,
        val_loader,
        scheduler,
    )
    # %%
    gan_loss_weight = np.concatenate(
        [
            np.repeat(0, 5000),
            np.linspace(0, 0.2, 5000),
            np.linspace(0.2, 0.3, 5000),
            np.repeat(0.3, TOTAL_STEPS - 15000),
        ]
    )
    _, vae_beta_weight = cosine_cyclical_annealing(
        TOTAL_STEPS, start=0.01, stop=0.18, n_cycle=5, ratio=0.6
    )
    step = 0
    gradient_norms = []
    total_loss, onset_loss, vel_loss, time_loss, vae_looss, gan_loss = 0, 0, 0, 0, 0, 0
    update_interval = 100
    update_val_interval = 500
    p_bar = tqdm.tqdm()
    while step < TOTAL_STEPS:
        for data in train_loader:

            vae_model.train()
            gan_model.train()
            data = {key: data[key].to("cuda") for key in data}
            output = vae_model(
                onset_reduce=data["onset_reduce"],
                vel_reduce=data["vel_reduce"],
                timeoffset_reduce=data["timeoffset_reduce"],
                density=data["density"],
                intensity=data["intensity"],
                style=data["style"],
                inst=data["instrument"],
                masks=data["masks"] * 1.0,
                onset=data["onset"],
                vel=data["vel"],
                time=data["timeoffset"],
            )
            gan_loss, _ = gan_model(
                output["z"],
                label_data=[
                    data["density"].flatten(),
                    data["intensity"].flatten(),
                    data["style"].flatten(),
                    data["instrument"][:, 0, :],
                ],
            )
            beta = vae_beta_weight[step]
            gamma = gan_loss_weight[step]
            loss = (
                output["onset_loss"]
                + output["vel_loss"]
                + output["time_loss"]
                + beta * output["vae_loss"]
                - gamma * gan_loss
            )

            vae_optimizer.zero_grad()
            accelerator.backward(loss)
            clip_grad_norm_(vae_model.parameters(), 3)
            vae_optimizer.step()
            scheduler.step()

            total_norm = 0
            for param in vae_model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            gradient_norms.append(total_norm)

            gan_loss, _ = gan_model(
                output["z"].detach(),
                label_data=[
                    data["density"].flatten(),
                    data["intensity"].flatten(),
                    data["style"].flatten(),
                    data["instrument"][:, 0, :],
                ],
            )
            gan_optimizer.zero_grad()
            accelerator.backward(gan_loss)
            gan_optimizer.step()

            step += 1
            p_bar.set_postfix(
                {"batch_loss": loss.item(), "GAN_loss": gan_loss.item(), "step": step}
            )

            total_loss += loss.item() / update_interval
            onset_loss += output["onset_loss"].item() / update_interval
            vel_loss += output["vel_loss"].item() / update_interval
            time_loss += output["time_loss"].item() / update_interval
            vae_looss += output["vae_loss"].item() / update_interval
            gan_loss += gan_loss.item() / update_interval

            if step % update_interval == 0:
                wandb.log({"Train/Loss": total_loss}, step=step)
                wandb.log({"Train/Loss Onset": onset_loss}, step=step)
                wandb.log({"Train/Loss Vel": vel_loss}, step=step)
                wandb.log({"Train/Loss Time": time_loss}, step=step)
                wandb.log({"Train/Loss VAE": vae_looss}, step=step)
                wandb.log({"Train/Loss GAN": gan_loss}, step=step)
                wandb.log({"Train/Gradient_Norm": np.mean(gradient_norms)}, step=step)
                total_loss, onset_loss, vel_loss, time_loss, vae_looss, gan_loss = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                gradient_norms = []
            if step % update_val_interval == 0:
                vae_model.eval()
                gan_model.eval()
                (
                    val_total_loss,
                    val_onset_loss,
                    val_vel_loss,
                    val_time_loss,
                    val_vae_looss,
                    val_gan_loss,
                ) = (0, 0, 0, 0, 0, 0)
                for data in val_loader:
                    data = {key: data[key].to("cuda") for key in data}
                    output = vae_model(
                        onset_reduce=data["onset_reduce"],
                        vel_reduce=data["vel_reduce"],
                        timeoffset_reduce=data["timeoffset_reduce"],
                        density=data["density"],
                        intensity=data["intensity"],
                        style=data["style"],
                        inst=data["instrument"],
                        masks=data["masks"] * 1.0,
                        onset=data["onset"],
                        vel=data["vel"],
                        time=data["timeoffset"],
                    )
                    gan_loss, _ = gan_model(
                        output["z"],
                        label_data=[
                            data["density"].flatten(),
                            data["intensity"].flatten(),
                            data["style"].flatten(),
                            data["instrument"][:, 0, :],
                        ],
                    )
                    loss = (
                        output["onset_loss"]
                        + output["vel_loss"]
                        + output["time_loss"]
                        + beta * output["vae_loss"]
                        - gamma * gan_loss
                    )
                    val_total_loss += loss.item() / len(val_loader)
                    val_onset_loss += output["onset_loss"].item() / len(val_loader)
                    val_vel_loss += output["vel_loss"].item() / len(val_loader)
                    val_time_loss += output["time_loss"].item() / len(val_loader)
                    val_vae_looss += output["vae_loss"].item() / len(val_loader)
                    val_gan_loss += gan_loss.item() / len(val_loader)

                wandb.log({"Val/Loss": val_total_loss}, step=step)
                wandb.log({"Val/Loss Onset": val_onset_loss}, step=step)
                wandb.log({"Val/Loss Vel": val_vel_loss}, step=step)
                wandb.log({"Val/Loss Time": val_time_loss}, step=step)
                wandb.log({"Val/Loss VAE": val_vae_looss}, step=step)
                wandb.log({"Val/Loss GAN": val_gan_loss}, step=step)
                torch.save(
                    {
                        "vae_model": vae_model.state_dict(),
                        "gan_model": gan_model.state_dict(),
                        "vae_optimizer": vae_optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "gan_optimizer": gan_optimizer.state_dict(),
                        "step": step,
                    },
                    f"/home/vae_ckpt/ckpt_{step}.ckpt",
                )
    # %%
