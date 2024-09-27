import logging
from pathlib import Path
import re

import torch
import esm
from carbonmatrix.model.lm.model_lm import ModelLM

def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return 'lora' not in model_name and not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)

def load_model_and_alphabet_local(model_location, load_regression=False, lora_config={}):
    # load from local checkpoint
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if load_regression and _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None

    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    # load
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    model_state = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ModelLM(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
        lora_config=lora_config,
    )

    # verified the keys
    # expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    expected_missing = set()

    missing = (expected_keys - found_keys) - expected_missing
    if missing:
        logging.warn(f"Missing key(s) in load esm2 state_dict: {missing}.")

    unexpected = found_keys - expected_keys
    if unexpected:
        logging.warn.append(f"Unexpected key(s) in state_dict: {unexpected}.")

    model.load_state_dict(model_state, strict=False)

    return model, alphabet
