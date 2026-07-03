from DisGlioma.src.model.disgene import CustomCLIP
import torch
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

model = CustomCLIP()  

ckpt_path = (PROJECT_ROOT / 'save_model' / 'disgene.pt').resolve()
ckpt = torch.load(ckpt_path, map_location='cpu') 
new_state_dict = OrderedDict()
for key, value in ckpt['state_dict'].items():    
    new_key = key.replace('model.', '') if key.startswith('model.') else key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

cls_embed_weights = model.cluster_embed.data
print(f"cls_embed shape: {cls_embed_weights.shape}")  # [3, 4, 512]

save_path = (PROJECT_ROOT / 'src' / 'model' / 'prompt_st3.pt').resolve()
torch.save(cls_embed_weights, save_path)

