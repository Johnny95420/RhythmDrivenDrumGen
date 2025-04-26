# %%
import torch
import torch.nn as nn


# %%
class VAEModelWrapper(nn.Module):
    def __init__(self, encoder_input_emb, encoder, intermediate, decoder):
        super().__init__()
        self.encoder_input_emb = encoder_input_emb
        self.encoder = encoder
        self.intermediate = intermediate
        self.decoder = decoder

    def forward(
        self,
        onset_reduce,
        vel_reduce,
        timeoffset_reduce,
        density,
        intensity,
        style,
        inst,
        masks=None,
        onset=None,
        vel=None,
        time=None,
    ):
        input_emb = self.encoder_input_emb(onset_reduce, vel_reduce, timeoffset_reduce)
        x = self.encoder(input_emb, masks)
        z, vq_loss = self.intermediate(x)
        pred_onset, pred_vel, pred_time = self.decoder(
            z, density, intensity, style, inst, masks
        )

        if onset is not None and vel is not None and time is not None:
            onset_loss, vel_loss, time_loss = self.compute_reconstruction_loss(
                onset, vel, time, pred_onset, pred_vel, pred_time
            )
            return {
                "onset_loss": onset_loss,
                "vel_loss": vel_loss,
                "time_loss": time_loss,
                "vae_loss": vq_loss,
                "z": z,
                "pred": (pred_onset, pred_vel, pred_time),
            }
        return {
            "vae_loss": vq_loss,
            "z": z,
            "pred": (pred_onset, pred_vel, pred_time),
        }

    def compute_reconstruction_loss(
        self, onset, vel, time, pred_onset, pred_vel, pred_time
    ):

        onset_pos = onset == 1

        smoothing_onset = torch.where(onset == 1, 0.9, 0.1)
        pred_onset_p = pred_onset.sigmoid()
        onset_loss = -(
            smoothing_onset * (pred_onset_p + 1e-5).log()
            + (1 - smoothing_onset) * (1 - pred_onset_p + 1e-5).log()
        ).mean()
        pred_val = nn.functional.sigmoid(pred_vel)
        vel_loss = nn.functional.mse_loss(vel[onset_pos], pred_val[onset_pos])

        pred_time = 0.5 * nn.functional.tanh(pred_time)
        time_loss = nn.functional.mse_loss(time[onset_pos], pred_time[onset_pos])
        return onset_loss, vel_loss, time_loss


class GANWrapper(nn.Module):
    def __init__(self, models, preprocesseors, loss_funcs):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.preprocesseors = preprocesseors
        self.loss_funcs = loss_funcs

    def forward(self, z, label_data):
        loss_spread = []
        losses = 0
        for idx, y in enumerate(label_data):
            if self.preprocesseors[idx] is not None:
                y = self.preprocesseors[idx](y)
            pred_y = self.models[idx](z)
            print(idx, y, y.shape, pred_y.shape)
            loss_val = self.loss_funcs[idx](pred_y, y)
            loss_spread.append(loss_val)
            losses += loss_val
        return losses, loss_spread


class CausalDecoderWrapper(nn.Module):
    def __init__(self, encoder_input_emb, encoder, intermediate, decoder, output_head):
        super().__init__()
        self.encoder_input_emb = encoder_input_emb
        self.encoder = encoder
        self.intermediate = intermediate
        self.decoder = decoder
        self.output_head = output_head

    def encode(
        self, onset_reduce, vel_reduce, timeoffset_reduce, masks, step_size=None
    ):
        input_emb = self.encoder_input_emb(
            onset_reduce,
            vel_reduce,
            timeoffset_reduce,
        )
        x = self.encoder(input_emb, 1.0 * masks)
        if step_size is None:
            x = x.mean(dim=1, keepdims=True)
            z, vae_loss = self.intermediate(x)
        else:
            temp_masks = masks.unsqueeze(-1)
            temp_masks = temp_masks.reshape(
                [temp_masks.size(0), temp_masks.size(1) // step_size, step_size, 1]
            )
            temp_masks = temp_masks.sum(2)
            bar_mask = (temp_masks.squeeze(-1) > 0).to(torch.float)

            idx = torch.arange(0, x.size(1), step_size)
            x = x[:, idx, :]
            z, vae_loss = self.intermediate(x, bar_mask)
        return z, vae_loss

    def process_decoder_input(
        self, onset, vel, timeoffset, z, masks, density, intensity, style, inst
    ):
        device = onset.device
        b, _, _ = onset.shape
        h_start, v_start, o_start = self.decoder.embed_tokens.get_start_token(b, device)
        onset = torch.cat([h_start, onset], dim=1)
        vel = torch.cat([v_start, vel], dim=1)
        timeoffset = torch.cat([o_start, timeoffset], dim=1)
        input_ids = torch.cat(
            [onset.unsqueeze(1), vel.unsqueeze(1), timeoffset.unsqueeze(1)], dim=1
        )
        masks = torch.cat([torch.ones([b, 1], device=device), masks], dim=-1)
        conditions = [density, intensity, style, inst, z]
        return input_ids, conditions, masks

    def decode(
        self,
        input_ids,
        conditions,
        input_condition_embeddings=None,
        attention_mask=None,
    ):
        if input_condition_embeddings is None:
            output = self.decoder(
                input_ids=input_ids,
                conditions=conditions,
                attention_mask=attention_mask,
            )
        else:
            output = self.decoder(
                input_ids=input_ids,
                input_condition_embeddings=input_condition_embeddings,
                attention_mask=attention_mask,
            )
        output = self.output_head(output.last_hidden_state)
        return output

    def compute_reconstruction_loss(
        self, onset, vel, time, pred_onset, pred_vel, pred_time, masks
    ):
        pos = masks == 1
        onset_pos = onset == 1

        pred_onset_p = pred_onset.sigmoid()
        onset_loss = -(
            onset * (pred_onset_p + 1e-5).log()
            + (1 - onset) * (1 - pred_onset_p + 1e-5).log()
        )
        onset_loss = onset_loss.mean(dim=-1)[pos].mean()

        pred_val = nn.functional.sigmoid(pred_vel)
        vel_loss = nn.functional.mse_loss(vel[onset_pos], pred_val[onset_pos])

        pred_time = 0.5 * nn.functional.tanh(pred_time)
        time_loss = nn.functional.mse_loss(time[onset_pos], pred_time[onset_pos])
        return onset_loss, vel_loss, time_loss

    def forward(
        self,
        onset_reduce,
        vel_reduce,
        timeoffset_reduce,
        onset,
        vel,
        timeoffset,
        desity,
        intensity,
        style,
        instrument,
        masks,
    ): ...


# %%
