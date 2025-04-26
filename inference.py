# %%
from pathlib import Path
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data_aug import DrumCoditionRollDataset, roll_collate_fn
from model.layers import BetaLatent, Decoder, Encoder, EncoderInputAdapter
from model.model import VAEModelWrapper
from symusic import Note, Score, Track
from torch.utils.data import DataLoader


# %%
def convert_to_midi(onset, vel, timeoffset, inverse_mapping, path):
    np_score = {"time": [], "duration": [], "pitch": [], "velocity": []}
    size = onset.shape[1]
    for i in range(size):
        onset_idx = torch.where(onset[:, i, :])[1]
        for idx in onset_idx:
            temp_time = i * 120 + timeoffset[:, i, idx] * 120
            if temp_time < 0:
                temp_time = torch.tensor(0)
            temp_time = int(temp_time.item())
            np_score["time"].append(temp_time)
            np_score["pitch"].append(inverse_mapping[idx.item()])
            np_score["velocity"].append(int(127 * vel[:, i, idx]))
            np_score["duration"].append(int(480 / 4))

    np_score = {key: np.array(np_score[key]) for key in np_score}
    notes = Note.from_numpy(**np_score)
    track = Track(is_drum=True)
    track.notes.extend(notes)
    score = Score(480)
    score.tracks.append(track)
    score.dump_midi(path)
    return np_score


def generate_piano_roll(max_time, np_score, np_ds_score, org_tpq, down_sample_tpq):
    piano_roll_onset = np.zeros([max_time, 9])
    piano_roll_vel = np.zeros([max_time, 9])
    piano_roll_timeoffset = np.zeros([max_time, 9])

    min_ticks = org_tpq / down_sample_tpq
    time_shift_arry = (np_score["time"] - np_ds_score["time"] * min_ticks) / min_ticks
    size = np_ds_score["time"].shape[0]
    for i in range(size):
        row_pos = np_ds_score["time"][i]
        col_pos = 3
        vel = np_ds_score["velocity"][i]
        time_offset = time_shift_arry[i]
        if vel / 127.0 > piano_roll_vel[row_pos, col_pos]:
            piano_roll_onset[row_pos, col_pos] = 1
            piano_roll_vel[row_pos, col_pos] = vel / 127.0
            piano_roll_timeoffset[row_pos, col_pos] = time_offset

    return piano_roll_onset, piano_roll_vel, piano_roll_timeoffset


def process_midi(
    file_path: str,
    down_sample_tpq: int = 4,
    device: str = "cuda",
):
    """
    Process a MIDI file to generate downsampled piano roll tensors.

    Args:
        file_path (str): Path to the MIDI file.
        down_sample_tpq (int): Target TPQ for downsampling.
        seq_len (int): Sequence length to truncate the piano roll.
        device (str): Device to store tensors (default: "cuda").

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Processed piano roll tensors (onset, velocity, time offset).
    """
    # Load and downsample MIDI
    score = Score(file_path)
    down_sample_score = score.resample(tpq=down_sample_tpq)

    # Convert notes to numpy arrays
    np_score = score.tracks[0].notes.numpy()
    np_ds_score = down_sample_score.tracks[0].notes.numpy()
    max_time = np_ds_score["time"][-1] + 1

    # Generate piano roll representations
    piano_roll_onset, piano_roll_vel, piano_roll_timeoffset = generate_piano_roll(
        max_time, np_score, np_ds_score, score.tpq, down_sample_tpq
    )

    # Convert to PyTorch tensors and move to device
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)

    piano_roll_onset, piano_roll_vel, piano_roll_timeoffset = (
        to_tensor(piano_roll_onset),
        to_tensor(piano_roll_vel),
        to_tensor(piano_roll_timeoffset),
    )

    # Truncate sequence length
    return (
        piano_roll_onset,
        piano_roll_vel,
        piano_roll_timeoffset,
    )


def slice_piano_roll(
    piano_roll_onset, piano_roll_vel, piano_roll_timeoffset, window_size, hop_size
):
    """
    Slice piano roll tensors into overlapping windows.

    Args:
        piano_roll_onset (torch.Tensor): Onset tensor of shape (1, T, D).
        piano_roll_vel (torch.Tensor): Velocity tensor of shape (1, T, D).
        piano_roll_timeoffset (torch.Tensor): Time offset tensor of shape (1, T, D).
        window_size (int): Size of each window.
        hop_size (int): Step size for overlapping.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: List of sliced windows.
    """
    slices = []
    total_length = piano_roll_onset.shape[1]

    for start in range(0, total_length - window_size + 1, hop_size):
        slices.append(
            (
                piano_roll_onset[:, start : start + window_size],
                piano_roll_vel[:, start : start + window_size],
                piano_roll_timeoffset[:, start : start + window_size],
            )
        )

    return slices


def assign_generate_paramter(density, intensity, style, inst, device):
    # {
    #     "reggae": 0,
    #     "gospel": 1,
    #     "latin": 2,
    #     "country": 3,
    #     "punk": 4,
    #     "highlife": 5,
    #     "rock": 6,
    #     "hiphop": 7,
    #     "pop": 8,
    #     "middleeastern": 9,
    #     "neworleans": 10,
    #     "jazz": 11,
    #     "blues": 12,
    #     "soul": 13,
    #     "afrocuban": 14,
    #     "afrobeat": 14,
    #     "dance": 15,
    #     "funk": 16,
    # }
    density = torch.tensor([[[density]]]).to(torch.float32).to(device)
    intensity = torch.tensor([[[intensity]]]).to(torch.float32).to(device)
    style = torch.tensor([[style]]).to(torch.long).to(device)
    # kcik
    # snare
    # close hi hat
    # floor tom
    # open hi hat
    # low mid tom
    # crash
    # high tom
    # ride
    inst = torch.tensor([[inst]]).to(torch.float32).to(device)
    return density, intensity, style, inst


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

    test_dataset = DrumCoditionRollDataset(test_data, 4)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=roll_collate_fn
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
    ckpt = torch.load("/home/vae_ckpt/ckpt_25000.ckpt")
    vae_model.load_state_dict(ckpt["vae_model"])
    # %%
    test_loader = iter(test_loader)
    inverse_mapping = {val: key for key, val in test_dataset.mapping_table.items()}
    # %%
    data = next(test_loader)
    data = {key: data[key].cuda() for key in data}
    # %%
    # torch.tensor([[11]]).cuda(),
    vae_model.eval()
    # data["instrument"][:, :, 0] = 0.3 # kcik
    # data["instrument"][:, :, 1] = 0.0 # snare
    # data["instrument"][:, :, 2] = 0.5 # close hi hat
    # data["instrument"][:, :, 3] = 0.3 # floor tom
    # data["instrument"][:, :, 4] = 0.4 # open hi hat
    # data["instrument"][:, :, 5] = 0.4 # low mid tom
    # data["instrument"][:, :, 6] = 0.4 # crash
    # data["instrument"][:, :, 7] = 0.4 # high tom
    # data["instrument"][:, :, 8] = 0.4 # ride
    output = vae_model(
        data["onset_reduce"],
        data["vel_reduce"],
        data["timeoffset_reduce"],
        data["density"] + 1.0,
        data["intensity"] - 3.5,
        data["style"],
        data["instrument"],
        data["masks"] * 1.0,
        onset=data["onset"],
        vel=data["vel"],
        time=data["timeoffset"],
    )
    onset, vel, time = output["pred"]
    onset, vel, time = onset.sigmoid(), vel.sigmoid(), 0.5 * nn.functional.tanh(time)

    idx = 0
    onset = 1.0 * (onset[[idx]] >= 0.4)
    np_score = convert_to_midi(
        onset[:, :-1, :],
        vel[[idx]],
        time[[idx]] - time[[idx]].mean(),
        inverse_mapping,
        "/home/test.midi",
    )
    _ = convert_to_midi(
        1.0 * (data["onset"][[idx]] >= 0.4),
        data["vel"][[idx]],
        data["timeoffset"][[idx]],
        inverse_mapping,
        "/home/org.midi",
    )

    _ = convert_to_midi(
        1.0 * (data["onset_reduce"][[idx]] >= 0.4),
        data["vel_reduce"][[idx]],
        data["timeoffset_reduce"][[idx]],
        inverse_mapping,
        "/home/prompt.midi",
    )
    # %%
    file_path = "/home/test_clip/Cadence Whole Tone Ascending D Major 77 bpm.mid"
    piano_roll_onset, piano_roll_vel, piano_roll_timeoffset = process_midi(file_path)
    piano_roll_onset = torch.cat(
        [piano_roll_onset, torch.zeros([1, 4, 9], device="cuda"), piano_roll_onset], dim=1
    )
    piano_roll_vel = torch.cat(
        [piano_roll_vel, torch.zeros([1, 4, 9], device="cuda"), piano_roll_vel], dim=1
    )
    piano_roll_timeoffset = torch.cat(
        [
            piano_roll_timeoffset,
            torch.zeros([1, 4, 9], device="cuda"),
            piano_roll_timeoffset,
        ],
        dim=1,
    )
    piano_roll_onset_reduce, piano_roll_vel_reduce, piano_roll_timeoffset_reduce = (
        piano_roll_onset[:, :33],
        piano_roll_vel[:, :33],
        piano_roll_timeoffset[:, :33],
    )

    # %%
    density, intensity, style, inst = assign_generate_paramter(
        -2, -2, 11, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0], "cuda"
    )
    vae_model.eval()
    output = vae_model(
        piano_roll_onset_reduce,
        piano_roll_vel_reduce,
        piano_roll_timeoffset_reduce,
        density,
        intensity,
        style,
        inst,
    )

    onset, vel, time = output["pred"]
    onset, vel, time = onset.sigmoid(), vel.sigmoid(), 0.5 * nn.functional.tanh(time)
    idx = 0
    onset = 1.0 * (onset[[idx]] >= 0.4)
    np_score = convert_to_midi(
        onset[:, :-1, :],
        vel[[idx]],
        time[[idx]],
        inverse_mapping,
        "/home/test.midi",
    )
# %%
