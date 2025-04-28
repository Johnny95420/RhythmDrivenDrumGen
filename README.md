# RhythmDrivenDrumGen

## Project Overview

**RhythmDrivenDrumGen** is a deep learning-based system that generates drum patterns based on rhythmic control signals. It leverages modern sequence modeling techniques, including causal transformers, to produce musically coherent drum tracks from piano roll inputs and control parameters such as density and intensity.

This project was inspired by GrooVAE (https://magenta.tensorflow.org/groovae), but uses different encoder and decoder models. It also extends the generated sequence length and introduces finer-grained rhythm and parameter control.

The method for conditioning on parameters and inserting latent vectors into the decoder was adapted from MuseMorphose (https://github.com/YatingMusic/MuseMorphose).

---

## Preprocessing

The preprocessing pipeline is handled via `preprcoessing.py` and includes:

- **Mapping MIDI to Piano Roll Representations**: Onset, velocity, and time-offset data are extracted.
- **Control Parameter Preparation**: Density and intensity values are computed and aligned.
- **Data Augmentation** (`data_aug.py`): Techniques such as pitch shifting, time stretching, and random masking are applied to enhance training robustness.
- **Mask Generation**: Masks are created for sequence modeling to handle variable lengths.

All preprocessing parameters are configurable via `config.yaml`.

---

## Model Architecture

The model architecture is defined in the `model/` directory:

- **Encoder**: Encodes input piano roll features using stacked transformer layers.
- **Decoder**: A causal decoder predicts future tokens based on previous outputs and encoded context.
- **Conditional Controls**: Density, intensity, and instrument embeddings are injected to guide generation.
- **Core Model Files**:
  - `layers.py`: Basic building blocks (Attention, MLPs, Residuals)
  - `model.py`: Full model definition
  - `phi3_arch_model.py`: Specialized architecture based on Phi-3 transformer designs

---

## Training

Training is managed through `training_script_causal_decoder.py`:

- **Objective**: Autoregressive prediction of next piano roll tokens.
- **Optimization**: Uses mixed precision training (bfloat16) for efficiency.
- **Loss Function**: Cross-entropy over token predictions.
- **Augmentations**: Dynamically applied during training.
- **Configurable Parameters**: Batch size, learning rate, number of layers, dropout, etc., controlled via `config.yaml`.

Training can be resumed or fine-tuned from checkpoints.

---

## Inference

Inference is managed through `inference_bar_control.py`:

- **Input**: Piano roll onset/velocity/timeoffset features and control parameters.
- **Sequence Length Constraint**: Input length must be <= 128; generated sequence length must be a multiple of 4.
- **Autocast Inference**: Model inference is performed under mixed precision (bfloat16) for faster generation.
- **Condition Injection**: Density, intensity, and instrument embeddings influence the generation behavior.

---

## Usage

### Training

```bash
python training_script_causal_decoder.py
```

Training hyperparameters (like learning rate, batch size) are adjustable inside `config.yaml`.

### Inference

```python
mapping_table = {n: idx for idx, n in enumerate(set(mapping.values()))}
inverse_mapping = {val: key for key, val in mapping_table.items()}
ckpt = torch.load("CKPT_PATH")
model = get_model(ckpt["config"])
model.load_state_dict(ckpt["vae_model"])
model.eval()

(
    piano_roll_onset,
    piano_roll_vel,
    piano_roll_timeoffset,
) = process_midi("rhythmic_pattern_midi.mid")

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
np_score = convert_to_midi(
    onset,
    vel,
    time,
    inverse_mapping,
    "/home/test.midi",
)
```

---

