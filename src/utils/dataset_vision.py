import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from monai.transforms import Compose,LoadImaged,NormalizeIntensityd
from monai.transforms import ToTensord,EnsureChannelFirstd,Resized, RandFlipd, RandRotated
from monai.data.dataset import CacheDataset

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pathlib import Path

from DisGlioma.src.utils.transform import clinical_variable_token_list, encode_local_clinical

class Universal:
    def __init__(self, root_dir, csv_path, mode="train", ds_name=None ,fold=1):
        super(Universal, self).__init__()
        self.root_dir = root_dir
        self.image_size = (128,128,96)  
        self.mode = mode

        self.clinical = pd.read_csv(csv_path)
        if 'cluster' not in self.clinical.columns:
            self.clinical['cluster'] = -1
        else:
            pass

        if ds_name is not None:
            self.clinical['tag'] = [ds_name for i in range(len(self.clinical))]
        self._filter_invalid_clinical_rows()
        self._discretize_survival_months()
        if fold is not None:
            self._split_kfold(fold)
        
        if mode == "train":
            self.clinical = self.clinical[:int(0.7*len(self.clinical))]
        elif mode == "valid":
            self.clinical = self.clinical[int(0.7*len(self.clinical)):int(0.8*len(self.clinical))]
        elif mode == "test":
            self.clinical = self.clinical[int(0.8*len(self.clinical)):]
        else:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.subtype_caption = []

        self.subtype_caption = {'input_ids':[], 'attn_mask':[]}
        self._init_languages()
    
    def _init_languages(self):
        input_ids = []
        attn_mask = []

        caption_path = Path(__file__).resolve().parent / "subtype_caption.txt"

        with open(caption_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            text = [line.strip() for line in lines]
            encode = self.tokenize_text(text)

            input_ids.append(encode['input_ids'])
            attn_mask.append(encode['attention_mask'])

        self.subtype_caption['input_ids'] = torch.cat(input_ids, dim=0)
        self.subtype_caption['attention_mask'] = torch.cat(attn_mask, dim=0)    


    def cache_dataset(self):
        ds_list = self.create_dataset()
        trans = self.transform(self.image_size)
        ds = CacheDataset(ds_list, trans)

        return ds
    
    def create_dataset(self):
        ds_list = []
        for idx in range(len(self.clinical)):
            sample_info = self.clinical.iloc[idx]
            sample_id, os_time, censor = sample_info["id"], sample_info["os"], sample_info["censor"]
            dis_label = sample_info["label"]
            cluster = sample_info['cluster']
            os_time = float(os_time)
            os_time = round(os_time, 2) 
            tag = sample_info['tag']

            clinical_vars = encode_local_clinical(sample_info)

            image_path = os.path.join(self.root_dir, tag)
            data_dict = {
                "image":"{}/{}/{}.nii.gz".format(image_path,sample_id,sample_id),
                "os":os_time,
                "censor":censor,
                "label":dis_label,
                "cluster":cluster,
                'text_ids': self.subtype_caption
            }
            
            for key in clinical_variable_token_list:
                data_dict[key] = torch.tensor(clinical_vars[key], dtype=torch.long)
            ds_list.append(data_dict)
        
        return ds_list

    def transform(self,image_size=(128,128,96)):
        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image"], reader='ITKReader',image_only=True),
                EnsureChannelFirstd(["image"]),
                NormalizeIntensityd(['image'],channel_wise=True),
                Resized(['image'],spatial_size=image_size,mode=['trilinear']),
                RandFlipd(["image"],prob=0.5,spatial_axis=-3),
                RandRotated(["image"],prob=0.5,range_x=np.pi/6,range_y=np.pi/6,range_z=np.pi/6,keep_size=True),
                ToTensord(['image']),
            ])
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image"], reader='ITKReader',image_only=True),
                EnsureChannelFirstd(["image"]),
                NormalizeIntensityd(['image'],channel_wise=True),
                Resized(['image'],spatial_size=image_size,mode=['trilinear']),
                ToTensord(['image']),
            ])
        return trans
       
    def tokenize_text(self, text, max_length=160):
        encoded = self.tokenizer(
            text,
            truncation=True,           
            padding='max_length',      
            max_length=max_length,
            return_tensors='pt'        
        )
        
        return encoded['input_ids']
    

    def _split_kfold(self, fold):
        """
        Args:
            - self
            - fold : int
        Returns:
            - None
        """
        n = len(self.clinical)
        split_size = n // 5
        splits = [self.clinical[i*split_size:(i+1)*split_size] for i in range(5)]
    
        if n % 5 != 0:
            splits[-1] = pd.concat([splits[-1], self.clinical[5*split_size:]], axis=0)
        
        if fold == 5:
            pass
        else:
            rotated = pd.concat([*splits[:fold], *splits[fold+1:], splits[fold]], axis=0)
            self.clinical = rotated
            print("using fold{}".format(fold))
    
    def _discretize_survival_months(self, eps=1e-5):
        r"""
        This is where we convert the regression survival problem into a classification problem. We bin all survival times into 
        quartiles and assign labels to patient based on these bins.
        
        Args:
            - self
            - eps : Float 
            - uncensored_df : pd.DataFrame
        
        Returns:
            - None 
        
        """
        # cut the data into n_bins (4= quantiles)
        n_bins = 4

        uncensored_df = self.clinical[self.clinical['censor'] > 0]
        disc_labels, q_bins = pd.qcut(uncensored_df['os'], q=n_bins, retbins=True, labels=False)

        q_bins[-1] = 1e6  # set rightmost edge to be infinite
        # q_bins[-1] = self.clinical['os'].max() + eps
        q_bins[0] = 0  # set leftmost edge to be 0
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.clinical['os'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.clinical.insert(1, 'label', disc_labels.values.astype(int))
        # bins = q_bins

    def _filter_invalid_clinical_rows(self):
        required_cols = ["id", "os", "censor"]
        missing_mask = self.clinical[required_cols].isna().any(axis=1)
        if missing_mask.any():
            dropped = int(missing_mask.sum())
            print(f"Dropping {dropped} clinical rows with missing id/os/censor")
            self.clinical = self.clinical.loc[~missing_mask].copy()

        self.clinical["os"] = pd.to_numeric(self.clinical["os"], errors="coerce")
        self.clinical["censor"] = pd.to_numeric(self.clinical["censor"], errors="coerce")