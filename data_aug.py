# %%
import copy
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from symusic import Score
from torch.utils.data import Dataset
from utils.mapping import mapping, style_mapping

conf = OmegaConf.load("config.yaml")
STYLE_MAPPING = style_mapping
NORMALIZE_FACTOR = conf.Data.normalize_factor
DENSITY_ZERO = -NORMALIZE_FACTOR["density_mean"] / NORMALIZE_FACTOR["density_std"]
INTENSITY_ZERO = -NORMALIZE_FACTOR["intensity_mean"] / NORMALIZE_FACTOR["intensity_std"]
# %%
class DrumCoditionRollDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        down_sample_tpq: int = conf.Data.down_sample_tpq,
        occured_notes: Iterable = set(mapping.values()),
    ):
        self.data_path = data_path
        self.down_sample_tpq = down_sample_tpq
        self.occured_notes = occured_notes
        self.mapping_table = {n: idx for idx, n in enumerate(occured_notes)}
        self.style_mapping_table = STYLE_MAPPING
        self.bar_size = down_sample_tpq * 4

    def __len__(self):
        return len(self.data_path)

    def compute_time_shift(self, fine_grained_time, corase_grained_time, min_ticks):
        return (fine_grained_time - corase_grained_time * min_ticks) / min_ticks

    def generate_piano_roll(self, tpq, max_time, np_score, np_ds_score):
        piano_roll_onset, piano_roll_vel, piano_roll_timeoffset = (
            np.zeros([max_time, 9]),
            np.zeros([max_time, 9]),
            np.zeros([max_time, 9]),
        )
        min_ticks = tpq / self.down_sample_tpq
        time_shift_arry = self.compute_time_shift(
            np_score["time"], np_ds_score["time"], min_ticks
        )
        size = np_ds_score["time"].shape[0]
        for i in range(size):
            row_pos = np_ds_score["time"][i]
            if row_pos >= max_time:
                continue
            col_pos = self.mapping_table.get(np_ds_score["pitch"][i], 3)
            vel = np_ds_score["velocity"][i]
            time_offset = time_shift_arry[i]

            piano_roll_onset[row_pos, col_pos] = 1
            piano_roll_vel[row_pos, col_pos] = vel / 127.0
            piano_roll_timeoffset[row_pos, col_pos] = time_offset

        return piano_roll_onset, piano_roll_vel, piano_roll_timeoffset

    def aug(self, max_time, piano_roll_onset, piano_roll_vel, piano_roll_timeoffset):
        piano_roll_onset_reduce, piano_roll_vel_reduce, piano_roll_timeoffset_reduce = (
            np.zeros([max_time, 9]),
            np.zeros([max_time, 9]),
            np.zeros([max_time, 9]),
        )
        vel_argmax = np.argmax(piano_roll_vel, axis=-1)

        piano_roll_onset_reduce[:, 3] = piano_roll_onset[
            np.arange(piano_roll_onset.shape[0]), vel_argmax
        ]
        piano_roll_vel_reduce[:, 3] = piano_roll_vel[
            np.arange(piano_roll_onset.shape[0]), vel_argmax
        ]
        piano_roll_timeoffset_reduce[:, 3] = piano_roll_timeoffset[
            np.arange(piano_roll_onset.shape[0]), vel_argmax
        ]
        return (
            piano_roll_onset_reduce,
            piano_roll_vel_reduce,
            piano_roll_timeoffset_reduce,
        )

    def get_style(self, file_path):
        file_path = str(file_path)
        return file_path.split("/")[-1].split("_")[2].split("-")[0]

    def compute_dentisy(self, piano_roll_onset):
        return np.mean(piano_roll_onset)

    def compute_intensity(self, piano_roll_vel, piano_roll_onset):
        return piano_roll_vel.sum() / (piano_roll_onset.sum() + 1e-5)

    def compute_inst_density(self, piano_roll_onset):
        return piano_roll_onset.mean(axis=0)

    def __getitem__(self, idx):
        f = self.data_path[idx]
        str_style = self.get_style(f)
        style = self.style_mapping_table[str_style]

        score = Score(f)
        down_sample_score = score.resample(tpq=4)
        np_score = score.tracks[0].notes.numpy()
        np_ds_score = down_sample_score.tracks[0].notes.numpy()
        max_time = np_ds_score["time"][-1]
        piano_roll_onset, piano_roll_vel, piano_roll_timeoffset = (
            self.generate_piano_roll(score.tpq, max_time, np_score, np_ds_score)
        )
        (
            piano_roll_onset_reduce,
            piano_roll_vel_reduce,
            piano_roll_timeoffset_reduce,
        ) = self.aug(max_time, piano_roll_onset, piano_roll_vel, piano_roll_timeoffset)

        densities, intensities, instruments = [], [], []
        for i in range(0, piano_roll_onset.shape[0], self.bar_size):
            density = self.compute_dentisy(piano_roll_onset[i : i + self.bar_size, :])
            densities.append(density)
            intensity = self.compute_intensity(
                piano_roll_vel[i : i + self.bar_size, :],
                piano_roll_onset[i : i + self.bar_size, :],
            )
            intensities.append(intensity)
            inst_intensity = self.compute_inst_density(
                piano_roll_onset[i : i + self.bar_size, :]
            )
            instruments.append(inst_intensity)

        densities = (
            np.array(densities) - NORMALIZE_FACTOR["density_mean"]
        ) / NORMALIZE_FACTOR["density_std"]
        intensities = (
            np.array(intensities) - NORMALIZE_FACTOR["intensity_mean"]
        ) / NORMALIZE_FACTOR["intensity_std"]
        instruments = np.array(instruments)

        return {
            "onset": piano_roll_onset,
            "vel": piano_roll_vel,
            "timeoffset": piano_roll_timeoffset,
            "onset_reduce": piano_roll_onset_reduce,
            "vel_reduce": piano_roll_vel_reduce,
            "timeoffset_reduce": piano_roll_timeoffset_reduce,
            "style": style,
            "density": densities,
            "intensity": intensities,
            "instrument": instruments,
        }


def roll_collate_fn(batch):
    max_size = conf.Data.clip_bar * conf.Data.down_sample_tpq * 4
    max_condition_size = conf.Data.clip_bar

    final_data = {
        "onset": [],
        "vel": [],
        "timeoffset": [],
        "onset_reduce": [],
        "vel_reduce": [],
        "timeoffset_reduce": [],
        "masks": [],
        "style": [],
        "density": [],
        "intensity": [],
        "instrument": [],
    }
    for data in batch:
        curr_size = data["onset"].shape[0]
        pad_len = max_size - curr_size
        cond_pad_len = max_condition_size - data["density"].shape[0]
        mask = np.ones([max_size], dtype=int)
        if pad_len > 0:
            mask[-pad_len:] = 0
            pad = np.zeros([pad_len, 9])
            for key in [
                "onset",
                "vel",
                "timeoffset",
                "onset_reduce",
                "vel_reduce",
                "timeoffset_reduce",
            ]:
                data[key] = np.concatenate([data[key], pad])

            condition_pad = np.ones([cond_pad_len])
            data["density"] = np.concatenate(
                [data["density"], condition_pad * DENSITY_ZERO]
            )
            data["intensity"] = np.concatenate(
                [data["intensity"], condition_pad * INTENSITY_ZERO]
            )
            inst_pad = np.zeros([cond_pad_len, 9])
            data["instrument"] = np.concatenate([data["instrument"], inst_pad])

        onset = torch.tensor(data["onset"]).unsqueeze(0)
        vel = torch.tensor(data["vel"]).unsqueeze(0)
        timeoffset = torch.tensor(data["timeoffset"]).unsqueeze(0)
        onset_reduce = torch.tensor(data["onset_reduce"]).unsqueeze(0)
        vel_reduce = torch.tensor(data["vel_reduce"]).unsqueeze(0)
        timeoffset_reduce = torch.tensor(data["timeoffset_reduce"]).unsqueeze(0)

        mask = torch.tensor(mask).unsqueeze(0)
        style = torch.tensor(data["style"])[None, None]
        density = torch.tensor(data["density"])[None, :, None]
        intensity = torch.tensor(data["intensity"])[None, :, None]
        instrument = torch.tensor(data["instrument"])[None]

        final_data["onset"].append(onset)
        final_data["vel"].append(vel)
        final_data["timeoffset"].append(timeoffset)

        final_data["onset_reduce"].append(onset_reduce)
        final_data["vel_reduce"].append(vel_reduce)
        final_data["timeoffset_reduce"].append(timeoffset_reduce)
        final_data["masks"].append(mask)
        final_data["style"].append(style)
        final_data["density"].append(density)
        final_data["intensity"].append(intensity)
        final_data["instrument"].append(instrument)

    for key in final_data:
        final_data[key] = torch.concat(final_data[key], dim=0)
        if final_data[key].dtype == torch.float64:
            final_data[key] = final_data[key].to(torch.float32)
    return final_data
