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
from pathlib import Path

import numpy as np
import wandb
from accelerate import Accelerator
from data_aug import DrumCoditionRollDataset, roll_collate_fn
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from utils.model_create import get_model, get_training_params
from utils.utils import compute_grad_norm, cosine_cyclical_annealing

conf = OmegaConf.load("config.yaml")
# %%
wandb.init(project="drum conditional generation", config=dict(conf))


def forward(model, data, step, conf):
    with torch.autocast("cuda", torch.bfloat16):
        z, vae_loss = model.encode(
            data["onset_reduce"],
            data["vel_reduce"],
            data["timeoffset_reduce"],
            data["masks"],
            16,
        )
        z = torch.repeat_interleave(z, 4 * conf.Data.down_sample_tpq, dim=1)
        density = torch.repeat_interleave(
            data["density"], 4 * conf.Data.down_sample_tpq, dim=1
        )
        intensity = torch.repeat_interleave(
            data["intensity"], 4 * conf.Data.down_sample_tpq, dim=1
        )
        instrument = torch.repeat_interleave(
            data["instrument"], 4 * conf.Data.down_sample_tpq, dim=1
        )
        style = torch.repeat_interleave(data["style"], data["onset"].size(1), dim=1)
        input_ids, condition, masks = model.process_decoder_input(
            data["onset"],
            data["vel"],
            data["timeoffset"],
            z,
            data["masks"],
            density,
            intensity,
            style,
            instrument,
        )
        output = model.decode(
            input_ids=input_ids[:, :, :-1],
            conditions=condition,
            attention_mask=masks[:, :-1],
        )
        onset_loss, vel_loss, time_loss = model.compute_reconstruction_loss(
            data["onset"],
            data["vel"],
            data["timeoffset"],
            output[0],
            output[1],
            output[2],
            data["masks"],
        )
        beta = vae_beta_weight[step]
        loss = onset_loss + vel_loss + time_loss + beta * vae_loss
    return loss, onset_loss, vel_loss, time_loss, vae_loss


# %%
if __name__ == "__main__":
    processed_midi_paths = list(
        Path(f"{conf.Data.data_dir}/{conf.Data.slice_folder}").glob("**/*.mid*")
    )
    shuffle(processed_midi_paths)
    l = len(processed_midi_paths)
    val_size, test_size = int(l * conf.Data.data_split.val_prop), int(
        l * conf.Data.data_split.test_prop
    )
    test_data, val_data, train_data = (
        processed_midi_paths[:test_size],
        processed_midi_paths[test_size : test_size + val_size],
        processed_midi_paths[test_size + val_size :],
    )
    train_dataset = DrumCoditionRollDataset(train_data)
    val_dataset = DrumCoditionRollDataset(val_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.Data.batch_size.train,
        shuffle=True,
        collate_fn=roll_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=conf.Data.batch_size.val, collate_fn=roll_collate_fn
    )
    # %%
    model = get_model(conf)
    params = get_training_params(model, conf.Training.weight_decay)
    optimizer = AdamW(params, lr=conf.Training.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, conf.Training.warmup_steps, conf.Training.training_steps
    )
    accelerator = Accelerator(mixed_precision="bf16")
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
    )
    _, vae_beta_weight = cosine_cyclical_annealing(
        conf.Training.training_steps,
        start=conf.Training.VAE_params.start,
        stop=conf.Training.VAE_params.end,
        n_cycle=conf.Training.VAE_params.n_cycle,
        ratio=conf.Training.VAE_params.ratio,
    )
    vae_beta_weight[: conf.Training.VAE_params.start_step] = 0
    # %%
    step = 0
    loss_accumulator = {
        "Loss": 0,
        "Loss Onset": 0,
        "Loss Vel": 0,
        "Loss Time": 0,
        "Loss VAE": 0,
        "Gradient_Norm": 0,
    }
    update_interval, update_val_interval = 100, 500
    p_bar = tqdm.tqdm()
    while step <= conf.Training.training_steps:
        for data in train_loader:
            model.train()
            data = {key: data[key].cuda() for key in data}
            loss, onset_loss, vel_loss, time_loss, vae_loss = forward(
                model, data, step, conf
            )
            optimizer.zero_grad()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            scheduler.step()
            total_norm = compute_grad_norm(model)
            step += 1

            p_bar.set_postfix(
                {"batch_loss": loss.item(), "Grad Norm": total_norm, "step": step}
            )
            for key, val in zip(
                [
                    "Loss",
                    "Loss Onset",
                    "Loss Vel",
                    "Loss Time",
                    "Loss VAE",
                    "Gradient_Norm",
                ],
                [loss, onset_loss, vel_loss, time_loss, vae_loss, total_norm],
            ):
                loss_accumulator[key] += (
                    val.item() / update_interval
                    if isinstance(val, torch.Tensor)
                    else val / update_interval
                )

            if step % update_interval == 0:
                for key, val in loss_accumulator.items():
                    wandb.log({f"Train/{key}": val}, step=step)
                loss_accumulator = {
                    "Loss": 0,
                    "Loss Onset": 0,
                    "Loss Vel": 0,
                    "Loss Time": 0,
                    "Loss VAE": 0,
                    "Gradient_Norm": 0,
                }

            if step % update_val_interval == 0:
                model.eval()
                val_loss_accumulator = {
                    "Loss": 0,
                    "Loss Onset": 0,
                    "Loss Vel": 0,
                    "Loss Time": 0,
                    "Loss VAE": 0,
                }
                with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
                    for data in val_loader:
                        data = {key: data[key].to("cuda") for key in data}
                        (
                            val_loss,
                            val_onset_loss,
                            val_vel_loss,
                            val_time_loss,
                            val_vae_loss,
                        ) = forward(model, data, step, conf)
                        for key, val in zip(
                            [
                                "Loss",
                                "Loss Onset",
                                "Loss Vel",
                                "Loss Time",
                                "Loss VAE",
                            ],
                            [
                                val_loss,
                                val_onset_loss,
                                val_vel_loss,
                                val_time_loss,
                                val_vae_loss,
                            ],
                        ):
                            val_loss_accumulator[key] += (
                                val.item() / len(val_loader)
                                if isinstance(val, torch.Tensor)
                                else val / len(val_loader)
                            )

                for key, val in val_loss_accumulator.items():
                    wandb.log({f"Val/{key}": val}, step=step)

                torch.save(
                    {
                        "config": conf,
                        "vae_model": model.state_dict(),
                        "vae_optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                    },
                    f"/home/vae_causal_ckpt_4bar/ckpt_{step}.ckpt",
                )

    # %%
