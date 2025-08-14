import numpy as np

def get_model_configs(opt):
    MODEL_CONFIGS = {
        "40m": {
            "width": 512,
            "layers": 12,
            "heads": 8,
            "cond_drop_prob": 0.1,
            "input_channels": opt.input_channels,
            "output_channels": opt.input_channels,
            "use_flash2": True,
        },
        "300m": {
            "width": 1024,
            "layers": 24,
            "heads": 16,
            "cond_drop_prob": 0.1,
            "input_channels": opt.input_channels,
            "output_channels": opt.input_channels,
            "use_flash2": True,
        },
        "1b": {
            "width": 2048,
            "layers": 24,
            "heads": 32,
            "cond_drop_prob": 0.1,
            "input_channels": opt.input_channels,
            "output_channels": opt.input_channels,
            "use_flash2": True,
        },
    }
    
    # --- Select and return the appropriate configurations ---
    try:
        model_config = MODEL_CONFIGS[opt.base_name]
    except KeyError:
        raise ValueError(f"Configuration for base_name '{opt.base_name}' not found.")

    return model_config


    