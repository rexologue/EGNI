import os
from typing import Literal

import torch
import torchvision
from torch.hub import load_state_dict_from_url

# URL for downloading pretrained models
URLS_DICT = {
    # ResNets (-50, -101, -152)
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-f82ba261.pth",

    # EfficientNets (S, M, L)
    "efficientnet_v2_s": 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
    "efficientnet_v2_m": 'https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth',
}

# Save path
MODEL_DIR = os.path.dirname(__file__)

def load_model(
    model_name: Literal["resnet50", "resnet101", "resnet152", "efficientnet_v2_s", "efficientnet_v2_m", "all"],
    load_in_mem: bool = True
) -> torch.nn.Module | None:
    
    # Weights loading
    if model_name == "all":
        for model_name, url in URLS_DICT.items():
            load_state_dict_from_url(url, model_dir=MODEL_DIR, file_name=model_name)

    else:
        if model_name not in URLS_DICT.keys():
            raise ValueError(f"Unknown model {model_name}")
        
        state_dict = load_state_dict_from_url(URLS_DICT[model_name], model_dir=MODEL_DIR, file_name=model_name)

        if load_in_mem:
            model = getattr(torchvision.models, model_name)()
            model.load_state_dict(state_dict)

            return model
        

    