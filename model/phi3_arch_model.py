# %%
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3PreTrainedModel,
    Phi3RMSNorm,
    Phi3Attention,
    Phi3FlashAttention2,
    Phi3SdpaAttention,
    Phi3MLP,
)


# %%
class InputAdapter(nn.Module):
    def __init__(
        self,
        input_size=[9, 9, 9],
        hidden_size=[64, 64, 64],
        output_size=512,
        vel_dropout=0.2,
        time_dropout=0.2,
    ):
        super().__init__()
        self.onset_proj = nn.Linear(input_size[0], hidden_size[0])
        self.vel_proj = nn.Linear(input_size[1], hidden_size[1])
        self.time_proj = nn.Linear(input_size[2], hidden_size[2])
        self.output_proj = nn.Linear(sum(hidden_size), output_size)

        self.vel_drop = nn.Dropout(vel_dropout)
        self.time_drop = nn.Dropout(time_dropout)

        self.h_start_token = nn.Embedding(1, input_size[0])
        self.v_start_token = nn.Embedding(1, input_size[1])
        self.o_start_token = nn.Embedding(1, input_size[2])

    def get_start_token(self, b_size, device):
        h_start = self.h_start_token(
            torch.zeros([b_size, 1], dtype=torch.long, device=device)
        )
        v_start = self.v_start_token(
            torch.zeros([b_size, 1], dtype=torch.long, device=device)
        )
        o_start = self.o_start_token(
            torch.zeros([b_size, 1], dtype=torch.long, device=device)
        )
        return h_start, v_start, o_start

    def add_latent_injection(self, x, z):
        if z is not None:
            return x + z
        else:
            return x

    def forward(self, x, z=None):
        onset, vel, time = x[:, 0], self.vel_drop(x[:, 1]), self.time_drop(x[:, 2])
        x = torch.cat(
            [self.onset_proj(onset), self.vel_proj(vel), self.time_proj(time)], dim=-1
        )
        x = self.add_latent_injection(self.output_proj(x), z)
        return x


class ConditionAdapter(nn.Module):
    def __init__(self, condition_params: List, output_size: int):
        super().__init__()
        # condition_params
        # [[dim1, type1, proj_dim1],[dim2, type2, proj_dim2],...]
        # type: val, cat
        self.cond_len = len(condition_params)
        cond_proj_modules = []
        size = 0
        for i in range(self.cond_len):
            dim, t, proj_dim = condition_params[i]
            if t == "val":
                cond_proj_modules.append(nn.Linear(dim, proj_dim))
            elif t == "cat":
                cond_proj_modules.append(nn.Embedding(dim, proj_dim))
            elif t == "none":
                cond_proj_modules.append(nn.Identity())
            size += proj_dim

        self.cond_proj_modules = nn.ModuleList(cond_proj_modules)
        self.output_proj = nn.Linear(size, output_size, bias=False)

    def forward(self, conditions):
        output = []
        for i in range(self.cond_len):
            x = self.cond_proj_modules[i](conditions[i])
            output.append(x)
        output = torch.cat(output, dim=-1)
        output = self.output_proj(output)
        if len(output.shape) == 2:
            output = output.unsqueeze(1)
        return output


PHI3_ATTENTION_CLASSES = {
    "eager": Phi3Attention,
    "flash_attention_2": Phi3FlashAttention2,
    "sdpa": Phi3SdpaAttention,
}


class Phi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx=layer_idx
        )

        self.mlp = Phi3MLP(config)
        self.input_layernorm1 = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm2 = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm1(hidden_states)
        hidden_states = hidden_states + condition_embeddings
        hidden_states = self.input_layernorm2(hidden_states)
        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Phi3GrooveModel(Phi3PreTrainedModel):
    def __init__(self, config: Phi3Config, groove_adapter_paras):

        super().__init__(config)
        self.embed_tokens = InputAdapter(
            input_size=groove_adapter_paras["input_size"],
            hidden_size=groove_adapter_paras["hidden_size"],
            output_size=config.hidden_size,
            vel_dropout=groove_adapter_paras["vel_dropout"],
            time_dropout=groove_adapter_paras["time_dropout"],
        )
        self.condition_adapter = nn.ModuleList(
            [
                ConditionAdapter(
                    condition_params=groove_adapter_paras["condition_params"],
                    output_size=config.hidden_size,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [
                Phi3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        conditions: List = None,
        input_condition_embeddings: Tuple = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            if len(input_ids.shape) == 4:
                batch_size, _, seq_length = input_ids.shape[:3]
            else:
                batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, (condition_adapter, decoder_layer) in enumerate(
            zip(self.condition_adapter, self.layers)
        ):
            if input_condition_embeddings is None:
                condition_embeddings = condition_adapter(conditions)
            else:
                condition_embeddings = input_condition_embeddings[idx]

            condition_embeddings = condition_embeddings[:, : hidden_states.size(1), :]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    condition_embeddings,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    condition_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# %%
if __name__ == "__main__":
    default_config = Phi3Config(
        vocab_size=100,
        pad_token_id=1,
        hidden_size=384,
        intermediate_size=384 * 4,
        num_hidden_layers=6,
        _attn_implementation="flash_attention_2",
    )
    groove_adapter_paras = {
        "input_size": [9, 9, 9],
        "hidden_size": [32, 32, 32],
        "vel_dropout": 0.2,
        "time_dropout": 0.2,
        "condition_params": [[1, "none", 1], [1, "none", 1], [17, "cat", 4]],
    }
    model = Phi3GrooveModel(default_config, groove_adapter_paras).cuda()
    # %%
    input_ids = torch.cat(
        [
            torch.ones([32, 182, 9], device="cuda", dtype=torch.float32).unsqueeze(1),
            torch.ones([32, 182, 9], device="cuda", dtype=torch.float32).unsqueeze(1),
            torch.ones([32, 182, 9], device="cuda", dtype=torch.float32).unsqueeze(1),
        ],
        dim=1,
    )
    conditions = [
        torch.ones([32, 1], device="cuda", dtype=torch.float32),
        torch.ones([32, 1], device="cuda", dtype=torch.float32),
        torch.ones([32], device="cuda", dtype=torch.long),
    ]
    with torch.autocast("cuda", torch.float16):
        output = model(input_ids, conditions)

# %%
