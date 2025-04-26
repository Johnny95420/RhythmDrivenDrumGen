# %%
import copy
import os
from pathlib import Path

import numpy as np
import tqdm
from omegaconf import OmegaConf
from symusic import Note, Score, Track
from utils.mapping import mapping_notes_func

conf = OmegaConf.load("config.yaml")
DATA_DIR = conf.Data.data_dir
RAW_FOLDER = conf.Data.raw_folder
REMAPPING_FOLDER = conf.Data.remapping_folder
SLICE_FOLDER = conf.Data.slice_folder


def segment_file(track, tpq, clip_size, hop_size, new_dir, name):
    track_info = track.notes.numpy()
    track_end = track_info["time"].max()

    count = 0
    start = 0
    end = start + tpq * 4 * clip_size
    step = tpq * 4 * hop_size
    while end < track_end:
        keep_idx = np.where(
            ((start - 29) <= track_info["time"]) & (track_info["time"] <= (end + 29))
        )[0]
        empty_score = Score(x=tpq)
        temp_track = copy.deepcopy(track)
        temp_track.notes = temp_track.notes[keep_idx[0] : keep_idx[-1] + 1]
        for i in range(len(temp_track.notes)):
            temp_track.notes[i].time -= start
            if temp_track.notes[i].time < 0:
                temp_track.notes[i].time = 0
        empty_score.tracks = [temp_track]
        empty_score.dump_midi(f"{new_dir}/{count}_{name}")

        count += 1
        start += step
        end = start + tpq * 4 * clip_size

    keep_idx = np.where(
        ((start - 29) <= track_info["time"]) & (track_info["time"] <= (end + 29))
    )[0]
    empty_score = Score(x=tpq)
    temp_track = copy.deepcopy(track)
    temp_track.notes = temp_track.notes[keep_idx[0] : keep_idx[-1] + 1]
    for i in range(len(temp_track.notes)):
        temp_track.notes[i].time -= start
        if temp_track.notes[i].time < 0:
            temp_track.notes[i].time = 0
    empty_score.tracks = [temp_track]
    empty_score.dump_midi(f"{new_dir}/{count}_{name}")


def mapping_notes(f):
    name = str(f).split("/")[-1].replace(".mid", "")
    sig = name.split("_")[-1]
    if sig != "4-4":
        return

    midi_tab = Score(f)
    track = midi_tab.tracks[0]

    track_info = track.notes.numpy()
    track_info["pitch"] = mapping_notes_func(track_info["pitch"])
    track.notes = Note.from_numpy(**track_info)

    org_path = str(f)
    new_path = org_path.replace(
        f"{DATA_DIR}/{RAW_FOLDER}",
        f"{DATA_DIR}/{REMAPPING_FOLDER}",
    )
    new_parent = os.path.split(new_path)[0]
    os.makedirs(new_parent, exist_ok=True)
    midi_tab.dump_midi(new_path)


if __name__ == "__main__":

    midi_paths = list(Path(DATA_DIR).glob("**/*.mid*"))
    for f in tqdm.tqdm(midi_paths):
        mapping_notes(f)
    processed_midi_paths = list(
        Path(f"{DATA_DIR}/{REMAPPING_FOLDER}").glob("**/*.mid*")
    )
    for f in tqdm.tqdm(processed_midi_paths):
        midi_tab = Score(f)
        track = midi_tab.tracks[0]
        midi_tab.tempos = []
        parent, name = os.path.split(f)
        new_parent = parent.replace(
            f"{DATA_DIR}/{REMAPPING_FOLDER}",
            f"{DATA_DIR}/{SLICE_FOLDER}",
        )
        os.makedirs(new_parent, exist_ok=True)
        segment_file(
            track,
            midi_tab.tpq,
            conf.Data.clip_bar,
            conf.Data.hop_size,
            new_parent,
            name,
        )
