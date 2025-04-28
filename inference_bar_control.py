# %%
import numpy as np
import torch
import torch.nn as nn
from symusic import Note, Score, Track
from utils.mapping import mapping
from utils.model_create import get_model


# %%

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

# kcik 0
# snare 1
# close hi hat 2
# floor tom 3
# open hi hat 4
# low mid tom 5
# crash 6
# high tom 7
# ride 8


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
    seq_len: int = 64,
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
    curr_seq_len = piano_roll_onset.shape[1]
    if curr_seq_len > seq_len:
        return (
            piano_roll_onset[:, :seq_len],
            piano_roll_vel[:, :seq_len],
            piano_roll_timeoffset[:, :seq_len],
        )
    elif curr_seq_len < seq_len:
        padding = torch.zeros([1, seq_len - curr_seq_len, 9]).cuda()
        return (
            torch.cat([piano_roll_onset, padding], dim=1),
            torch.cat([piano_roll_vel, padding], dim=1),
            torch.cat([piano_roll_timeoffset, padding], dim=1),
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



def generate_control_params(
    control_points,
    params,
    seq_len,
    interpolation_type,
    device="cuda:0",
    is_cat: bool = False,
):
    control_points = (seq_len * np.array(control_points)).astype(int)
    params = np.array(params)
    if params.ndim == 1:
        params = params.reshape([-1, 1])
    output = np.zeros([seq_len, params.shape[-1]])
    for i in range(control_points.shape[0] - 1):
        start = control_points[i]
        end = control_points[i + 1]
        if interpolation_type == "left":
            output[start:end] = params[i]
        elif interpolation_type == "right":
            output[start:end] = params[i + 1]
        elif interpolation_type == "linear":
            output[start:end] = np.linspace(params[i], params[i + 1], end - start)
        else:
            raise ValueError("Only support left, right and linear ")
    output = torch.tensor(output, device=device).unsqueeze(0)
    if is_cat:
        return output.to(torch.long)
    return output

def generate(piano_roll_onset,
             piano_roll_vel,
             piano_roll_timeoffset,
             density,
             intensity,
             instrument,
             style,
             masks,
             generate_seq_len):
    
    if piano_roll_onset.size(1)>128:
        raise ValueError("The input sequence length must be less than or equal to 128.")
    
    if generate_seq_len % 4!=0:
        raise ValueError("The generated sequence length needs to be a multiple of 4.")
    
    if generate_seq_len > density.size(1):
        raise ValueError("The length of control parameters must be grater or equal to the generated sequence length")
        
    density = density.to(torch.bfloat16)
    intensity = intensity.to(torch.bfloat16)
    instrument = instrument.to(torch.bfloat16)
    
    with torch.autocast("cuda", torch.bfloat16):    
        z, _ = model.encode(
            piano_roll_onset,
            piano_roll_vel,
            piano_roll_timeoffset,
            masks,
            16,
        )
        z = torch.repeat_interleave(z, 16, dim=1)
        input_ids, condition, masks = model.process_decoder_input(
            torch.zeros([1, 1, 9], device="cuda"),
            torch.zeros([1, 1, 9], device="cuda"),
            torch.zeros([1, 1, 9], device="cuda"),
            z,
            masks,
            density,
            intensity,
            style,
            instrument,
        )
        condition_embeddings = [
            condition_layer(condition)
            for condition_layer in model.decoder.condition_adapter
        ]
        input_ids = input_ids[:, :, :1, :]
        for i in range(generate_seq_len):
            output = model.decoder(
                input_ids=input_ids, input_condition_embeddings=condition_embeddings
            )
            output = model.output_head(output.last_hidden_state)
            h = torch.bernoulli(output[0][:, -1, :].sigmoid())
            v = output[1][:, -1, :].sigmoid()
            o = 0.5 * nn.functional.tanh(output[2][:, -1, :])
            pos = h == 1
            v[~pos] = 0
            o[~pos] = 0
            new_input = torch.cat(
                [h[:, None, None], v[:, None, None], o[:, None, None]], dim=1
            )
            input_ids = torch.cat([input_ids, new_input], dim=2)

        onset, vel, time = input_ids[:, 0, 1:], input_ids[:, 1, 1:], input_ids[:, 2, 1:]
    return onset, vel, time
# %%
if __name__ == "__main__":
    mapping_table = {n: idx for idx, n in enumerate(set(mapping.values()))}
    inverse_mapping = {val: key for key, val in mapping_table.items()}
    ckpt = torch.load("/home/vae_causal_ckpt_4bar/ckpt_44500.ckpt")
    model = get_model(ckpt["config"])
    model.load_state_dict(ckpt["vae_model"])
    model.eval()
    
    (
        piano_roll_onset,
        piano_roll_vel,
        piano_roll_timeoffset,
    ) = process_midi("/home/Vamp Upbeat Jazz Fusion D Minor 103 bpm.mid")

    masks = generate_control_params([0, 1], [1, 1], 64, "left", "cuda:0", True)[:, :, 0]
    density = generate_control_params(
        [0, 0.25, 0.5, 0.75, 1], [-2, 1, 1, 2, 1], 64, "left"
    )
    intensity = generate_control_params(
        [0, 0.25, 0.5, 0.75, 1], [5, -1, -1, -2, 0.0], 64, "left"
    )

    instrument = generate_control_params(
        [0, 0.25, 0.5, 0.75, 1],
        [
            [0.25, -1.0, 0.0, 0, 0, 0, 0, 0, 0],
            [0.25, 0.1, 0.5, 0, 0, 0, 0, 0, 0],
            [0.25, 0.5, 0.5, 0, 0.1, 0, 0, 0, 0.2],
            [0.25, 0.5, 0.5, 0, 0, 0, 0, 0, 0.2],
            [0.25, 0.05, 0.1, 0, 0, 0, 0, 0, 0],
        ],
        64,
        "left",
    )
    style = generate_control_params([0, 1], [2, 2], 64, "left", is_cat=True)[:, :, 0]
    onset, vel, time = generate(piano_roll_onset,
                                piano_roll_vel,
                                piano_roll_timeoffset,
                                density,
                                intensity,
                                instrument,
                                style,
                                masks,
                                64)
    idx = 0
    np_score = convert_to_midi(
        onset,
        vel,
        time,
        inverse_mapping,
        "/home/test.midi",
    )
    # %%
