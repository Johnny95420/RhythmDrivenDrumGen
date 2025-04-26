# %%
from utils.utils import PositionalEmbedding
import torch
import torch.nn as nn


# %%
class LatentClassifier(torch.nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(LatentClassifier, self).__init__()
        self.fc_layer_1 = nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_activation_1 = nn.GELU()
        self.fc_layer_2 = nn.Linear(latent_dim, latent_dim, bias=True)
        self.fc_activation_2 = nn.GELU()
        self.fc_output_layer = nn.Linear(latent_dim, n_classes)

    def forward(self, z):
        x = self.fc_layer_1(z)
        x = self.fc_activation_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_activation_2(x)
        logits = self.fc_output_layer(x)
        return logits


# %%
class EncoderInputAdapter(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        d_model,
        dropout_vel,
        dropout_time,
        dropout,
        max_len,
    ):
        super().__init__()
        self.vel_drop = nn.Dropout(dropout_vel)
        self.time_drop = nn.Dropout(dropout_time)

        self.onset_linear = nn.Linear(input_dim[0], hidden_dim[0], bias=True)
        self.vel_linear = nn.Linear(input_dim[1], hidden_dim[1], bias=True)
        self.time_linear = nn.Linear(input_dim[2], hidden_dim[2], bias=True)
        self.linear = nn.Linear(sum(hidden_dim), d_model)

        self.act = torch.nn.GELU()
        self.pos_emb = PositionalEmbedding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        for layer in [self.onset_linear, self.time_linear, self.linear]:
            layer.bias.data.zero_()
            layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, onset, vel, time):
        vel = self.vel_drop(vel)
        time = self.time_drop(time)
        x = torch.cat(
            [self.onset_linear(onset), self.vel_linear(vel), self.time_linear(time)],
            dim=-1,
        )
        x = self.linear(x)
        x = self.act(x)
        x = self.pos_emb(x)
        return x


class DecoderInputAdapter(nn.Module):
    def __init__(
        self, latent_dim, style_emb_dim, inst_emb_dim, d_model, n_style, n_inst, max_len
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.temp_dim = int(d_model - inst_emb_dim - style_emb_dim - 2)
        self.inst_emb = nn.Linear(n_inst, inst_emb_dim)
        self.style_emb = nn.Embedding(n_style, style_emb_dim)
        self.latent_linear = nn.Linear(latent_dim, max_len * self.temp_dim)

    def init_weights(self, initrange=0.1):
        self.latent_linear.bias.data.zero_()
        self.latent_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, z, density, intensity, style, inst):
        z = self.latent_linear(z)
        z = z.view([-1, self.max_len, self.temp_dim])
        density = density.tile([1, z.size(1), 1])
        intensity = intensity.tile([1, z.size(1), 1])
        style = self.style_emb(style).tile([1, z.size(1), 1])
        inst = self.inst_emb(inst).tile([1, z.size(1), 1])
        x = torch.cat((z, density, intensity, style, inst), dim=-1)
        return x


class InAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        dim_feedforward,
        dropout,
        num_encoder_layers,
        n_style,
        style_embeddig_dim,
        n_inst,
        inst_embedding_dim,
    ):
        super().__init__()
        layers = []
        for _ in range(num_encoder_layers):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model, n_head, dim_feedforward, dropout, batch_first=True
                )
            )
        self.layers = nn.ModuleList(layers)
        self.inst_para_emb = nn.Linear(n_inst, inst_embedding_dim)
        self.cat_para_emb = nn.Embedding(n_style, style_embeddig_dim)
        self.para_proj = nn.Linear(style_embeddig_dim + inst_embedding_dim + 2, d_model)

    def forward(self, x, density, intensity, style, inst, mask=None):
        style = self.cat_para_emb(style)
        inst = self.inst_para_emb(inst)
        parameters = torch.concat((density, intensity, style, inst), dim=-1)
        parameters = self.para_proj(parameters)
        for mod in self.layers:
            x += parameters
            x = mod(x, src_key_padding_mask=mask)
        return x


class OutputAdapter(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()

        self.output_dim = output_dim * 3
        self.linear = nn.Linear(d_model, self.output_dim, bias=True)
        self.init_weights()

    def init_weights(self, initrange=0.1):
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        y = self.linear(x)
        y = torch.reshape(y, (y.shape[0], y.shape[1], 3, self.output_dim // 3))
        h_logits = y[:, :, 0, :]
        v_logits = y[:, :, 1, :]
        o_logits = y[:, :, 2, :]

        return h_logits, v_logits, o_logits


class Decoder(torch.nn.Module):

    def __init__(
        self,
        latent_dim,
        d_model,
        num_decoder_layers,
        nhead,
        dim_feedforward,
        output_max_len,
        output_size,
        dropout,
        n_style,
        style_emb_dim,
        n_inst,
        inst_emb_dim,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.output_max_len = output_max_len
        self.output_size = output_size
        self.dropout = dropout
        self.n_style = n_style
        self.style_emb_dim = style_emb_dim
        self.n_inst = n_inst
        self.inst_emb_dim = inst_emb_dim

        self.decoder_input_adapter = DecoderInputAdapter(
            latent_dim=latent_dim,
            d_model=self.d_model,
            n_style=self.n_style,
            style_emb_dim=self.style_emb_dim,
            max_len=self.output_max_len,
            n_inst=n_inst,
            inst_emb_dim=inst_emb_dim,
        )

        self.InAttentionDecoder = InAttentionEncoder(
            d_model=self.d_model,
            n_head=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_encoder_layers=self.num_decoder_layers,
            n_style=self.n_style,
            style_embeddig_dim=self.style_emb_dim,
            n_inst=n_inst,
            inst_embedding_dim=inst_emb_dim,
        )

        self.decoder_output_adapter = OutputAdapter(
            d_model=self.d_model, output_dim=output_size
        )

    def forward(self, z, density, intensity, style, inst, mask):
        pre_out = self.decoder_input_adapter(z, density, intensity, style, inst)
        decoder_ = self.InAttentionDecoder(
            pre_out, density, intensity, style, inst, mask
        )
        h_logits, v_logits, o_logits = self.decoder_output_adapter(decoder_)

        return h_logits, v_logits, o_logits


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        num_encoder_layers,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, src, mask=None):
        out = self.encoder(src, src_key_padding_mask=mask)
        return out


# %%
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        restart_threshold=1e-3,
        reset_interval=1000,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.reset_embeddings()

        self.reset_interval = reset_interval
        self.restart_threshold = restart_threshold
        self.step = 0
        self.register_buffer(
            "usage_count", torch.zeros(num_embeddings, dtype=torch.int32)
        )

    @torch.no_grad
    def reset_embeddings(self, indices=None, batch_samples=None):
        if indices is None:
            self.embedding.weight.data.uniform_(
                -1 / self.num_embeddings, 1 / self.num_embeddings
            )
        else:
            if batch_samples is not None:
                self.embedding.weight.data[indices] = batch_samples
            else:
                self.embedding.weight.data[indices].uniform_(
                    -1 / self.num_embeddings, 1 / self.num_embeddings
                )

    @torch.no_grad
    def check_dead_codes(self, batch_samples):
        if not self.training or (self.step + 1) % self.reset_interval != 0:
            return

        usage_rate = self.usage_count.float() / self.usage_count.sum()
        dead_codes = torch.where(usage_rate < self.restart_threshold)[0]
        if len(dead_codes) > 0:
            batch_size = batch_samples.shape[0]
            sample_indices = torch.randint(
                0, batch_size, (len(dead_codes),), device=batch_samples.device
            )
            reset_values = batch_samples[sample_indices]
            self.reset_embeddings(dead_codes, reset_values)
        self.step = 0
        self.usage_count.zero_()

    @torch.no_grad
    def update_usage_count(self, encoding_indices):
        if self.training:
            self.step += 1
            unique_indices = torch.unique(encoding_indices)
            self.usage_count[unique_indices] += 1

    def forward(self, inputs):
        input_shape = inputs.shape
        inputs_flat = inputs.view(-1, self.embedding_dim)
        self.check_dead_codes(inputs_flat)

        distances = torch.cdist(
            inputs_flat.unsqueeze(0), self.embedding.weight.unsqueeze(0)
        ).squeeze(0)
        encoding_indices = torch.argmin(distances, dim=1)
        self.update_usage_count(encoding_indices)
        quantized = self.embedding(encoding_indices).view(input_shape)
        loss = torch.mean(
            (quantized.detach() - inputs) ** 2
        ) + self.commitment_cost * torch.mean((quantized - inputs.detach()) ** 2)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss


class BetaLatent(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        dim = int(max_len * input_dim)
        self.mu_proj = nn.Linear(dim, hidden_dim)
        self.log_var_proj = nn.Linear(dim, hidden_dim)

    def reparametrize(self, mu, log_var):
        var = log_var.exp()
        std = var.pow(0.5)

        eps = torch.normal(
            torch.zeros(mu.shape, device=mu.device),
            torch.ones(std.shape, device=std.device),
        )
        z = mu + eps * std
        kl_loss = 0.5 * (mu.pow(2) + var - 1 - log_var).sum(dim=-1).mean()
        return z, kl_loss

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_proj(x)
        log_var = self.log_var_proj(x)
        z, kl_loss = self.reparametrize(mu, log_var)
        return z, kl_loss


class BetaLatentMultiDim(nn.Module):
    def __init__(self, input_dim, hidden_dim, free_bits):
        super().__init__()
        self.mu_proj = nn.Linear(input_dim, hidden_dim)
        self.log_var_proj = nn.Linear(input_dim, hidden_dim)
        self.free_bits = free_bits

    def reparametrize(self, mu, log_var, masks):
        var = log_var.exp()
        std = var.pow(0.5)

        eps = torch.normal(
            torch.zeros(mu.shape, device=mu.device),
            torch.ones(std.shape, device=std.device),
        )
        z = mu + eps * std
        kl_loss = 0.5 * (mu.pow(2) + var - 1 - log_var).mean(dim=-1)
        kl_loss = torch.clamp(kl_loss, min=self.free_bits)
        if masks is not None:
            kl_loss = (kl_loss * masks).mean()
        else:
            kl_loss = kl_loss.mean()
        return z, kl_loss

    def forward(self, x, masks=None):
        mu = self.mu_proj(x)
        log_var = self.log_var_proj(x)
        z, kl_loss = self.reparametrize(mu, log_var, masks)
        return z, kl_loss
