import torch
from DisGlioma.src.PathBINN.encoder import PathwayEncoder, PathwayEncoderConfig
from DisGlioma.src.utils.dataset_binn import RNASeqDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from pathlib import Path
    
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def extract_po_features_from_dataloader(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    po_features_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            
            if isinstance(batch, dict):
                input_data = {}
                for k, v in batch.items():
                    if hasattr(v, 'to'):
                        input_data[k] = v.to(device)
                    else:
                        input_data[k] = v
            else:
                if isinstance(batch[0], dict):
                    input_data = {}
                    for k, v in batch[0].items():
                        if hasattr(v, 'to'):
                            input_data[k] = v.to(device)
                        else:
                            input_data[k] = v
                else:
                    # Tuple (input_ids, values, router, ...)
                    input_data = {
                        'input_ids': batch[0].to(device),
                        'values': batch[1].to(device),
                        'router': batch[2].to(device)
                    }
            
            
            required_keys = ['input_ids', 'values', 'router']
            if not all(k in input_data for k in required_keys):
                raise ValueError(f"missing keys: {required_keys}")
            
            _, po = model(input_data)
            po_features_list.append(po.cpu())  
            
    return torch.cat(po_features_list, dim=0)  

def load_model():
    
    config = PathwayEncoderConfig()
    model = PathwayEncoder(config)

    ckpt_path = (PROJECT_ROOT / 'save_path' / 'path_binn.pt')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    root_path = (PROJECT_ROOT / 'data').resolve()
    csv_path = (PROJECT_ROOT / 'data' / 'clinical' / 'combined_data_os.csv').resolve()
    ds_train = RNASeqDataset(root_path, csv_path, 'combined_data', mode='all', fold=None)
    
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=16)

    po_features = extract_po_features_from_dataloader(model, dl_train, device='cuda' if torch.cuda.is_available() else 'cpu')
    return po_features


if __name__ == '__main__':
    po_features = load_model()
    features = po_features.detach().cpu().numpy()

    model = AgglomerativeClustering(
        n_clusters=3,
        linkage='ward'   # 常用: ward / complete / average / single
    )
    cluster_labels = model.fit_predict(features)

    csv_path = (PROJECT_ROOT / 'data' / 'clinical' / 'combined_data_os.csv').resolve()
    df = pd.read_csv(csv_path)

    df['cluster'] = cluster_labels.astype(str)
    df.to_csv(csv_path, index=False)

