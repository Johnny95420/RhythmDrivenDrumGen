from model.layers import (
    BetaLatentMultiDim,
    Encoder,
    EncoderInputAdapter,
    OutputAdapter,
)
from transformers.models.phi3 import Phi3Config
from model.phi3_arch_model import Phi3GrooveModel
from model.model import CausalDecoderWrapper


def get_model(conf):
    default_config = Phi3Config(
        vocab_size=100,
        pad_token_id=0,
        hidden_size=conf.Model.Decoder.hidden_size,
        intermediate_size=conf.Model.Decoder.hidden_size * 4,
        num_hidden_layers=conf.Model.Decoder.num_hidden_layers,
        _attn_implementation="flash_attention_2",
        num_attention_heads=conf.Model.Decoder.num_attention_heads,
        embd_pdrop=conf.Model.Decoder.embd_pdrop,
        attention_dropout=conf.Model.Decoder.attention_dropout,
        use_cache=False,
    )
    groove_adapter_paras = {
        "input_size": conf.Model.Decoder.token_input_size,
        "hidden_size": conf.Model.Decoder.token_hidden_size,
        "vel_dropout": conf.Model.Decoder.vel_dropout,
        "time_dropout": conf.Model.Decoder.time_dropout,
        # density,intensity,style,inst,latent z
        "condition_params": conf.Model.Decoder.condition_params,
    }
    model = CausalDecoderWrapper(
        encoder_input_emb=EncoderInputAdapter(
            input_dim=conf.Model.Encoder.input_dim,
            hidden_dim=conf.Model.Encoder.hidden_dim,
            d_model=conf.Model.Encoder.d_model,
            dropout_vel=conf.Model.Encoder.dropout_vel,
            dropout_time=conf.Model.Encoder.dropout_time,
            dropout=conf.Model.Encoder.emb_dropout,
            max_len=conf.Model.Encoder.max_len,
        ),
        encoder=Encoder(
            d_model=conf.Model.Encoder.d_model,
            nhead=conf.Model.Encoder.n_head,
            dim_feedforward=int(conf.Model.Encoder.d_model * 4),
            num_encoder_layers=conf.Model.Encoder.n_layer,
            dropout=conf.Model.Encoder.att_dropout,
        ),
        intermediate=BetaLatentMultiDim(
            conf.Model.Encoder.d_model,
            conf.Model.Intermediate.hidden_size,
            conf.Model.Intermediate.free_bits,
        ),
        decoder=Phi3GrooveModel(default_config, groove_adapter_paras),
        output_head=OutputAdapter(
            conf.Model.Decoder.hidden_size, conf.Model.Output.size
        ),
    ).cuda()
    return model


def get_training_params(model, wd):
    params = [
        {"params": [], "weight_decay": wd},
        {"params": [], "weight_decay": 0},
    ]
    for name, para in model.named_parameters():
        if "bias" in name or "emb" in name or "norm" in name:
            params[1]["params"].append(para)
        else:
            params[0]["params"].append(para)
    return params
