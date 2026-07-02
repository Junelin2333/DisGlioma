import os
import torch
import anndata as ad
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from ..scGPT.tokenizer import scGPTTokenizer
from typing import Union


class RNASeqDataset(Dataset):
    def __init__(self, root_path:str, csv_path:str, ds_name:str ,mode:str = "train", fold:int=1):
        """
        Args:
            data (array-like): n_features equals the number of all genes.
        """
        if os.path.exists(os.path.join(root_path, "bulk", "{}_fpkm.h5ad".format(ds_name))):
            expression_data = os.path.join(root_path, "bulk", "{}_fpkm.h5ad".format(ds_name))
        elif os.path.exists(os.path.join(root_path, "bulk", "{}_combat.h5ad".format(ds_name))):
            expression_data = os.path.join(root_path, "bulk", "{}_combat.h5ad".format(ds_name))
        else:
            raise ValueError("{} does not exist".format(ds_name))

        df = pd.read_csv(os.path.join(root_path, "data", "pathway", "pathway331.csv"))
        gene_list = df.loc[:, df.sum(axis=0) != 0].columns.to_list()
        self.router = df.loc[:, df.sum(axis=0) != 0].to_numpy(dtype=np.bool_)

        data = ad.read_h5ad(expression_data)
        self.data = data[:, data.var.index.isin(gene_list)]

        adj = np.load(os.path.join(root_path, "data", "pathway", "bionx_331.npy"))
        self.adj = torch.from_numpy(adj).to(torch.float32)

        print('Using ' + ds_name)
    
        self.clinical = pd.read_csv(csv_path, dtype={'id': str})
        self._filter_invalid_clinical_rows()
        self._discretize_survival_months()
        if fold is not None:
            self.clinical = self.clinical.sample(frac=1, random_state=42)
            self._split_kfold(fold)

        if mode == "train":
            self.clinical = self.clinical[:int(0.8*len(self.clinical))]
        elif mode == "valid":
            self.clinical = self.clinical[int(0.8*len(self.clinical)):int(0.9*len(self.clinical))]
        elif mode == "test":
            self.clinical = self.clinical[int(0.8*len(self.clinical)):]
        else:
            pass
        
        self.tokenizer = scGPTTokenizer()

    def __len__(self):
        return len(self.clinical)

    def __getitem__(self, idx):

        sample_info = self.clinical.iloc[idx]
        sample_id, os, censor = sample_info["id"], sample_info["os"], sample_info["censor"]
        dis_label = sample_info["label"]
        if pd.isna(dis_label) or dis_label < 0 or dis_label >= 4:
            raise ValueError(f"Invalid survival label {dis_label} for sample {sample_id}")
        os = float(os)
        os = round(os, 2)  

        expression = self.data[sample_id].X[0]   # 4998
        expression = np.asarray(expression, dtype=np.float32)  

        gene_symbols = self.data.var["gene_symbol"].to_list()
        genes, values = self.tokenizer.tokenize_cell_vectors(expression, gene_symbols)

        gene_dict = {
            'input_ids': genes,
            "values": values,
            "router":torch.from_numpy(self.router),
            "adj_matrix": self.adj,
        }
        return gene_dict, os, censor, dis_label

    def _filter_invalid_clinical_rows(self):
        required_cols = ["id", "os", "censor"]
        missing_mask = self.clinical[required_cols].isna().any(axis=1)
        if missing_mask.any():
            dropped = int(missing_mask.sum())
            print(f"Dropping {dropped} clinical rows with missing id/os/censor")
            self.clinical = self.clinical.loc[~missing_mask].copy()

        self.clinical["os"] = pd.to_numeric(self.clinical["os"], errors="coerce")
        self.clinical["censor"] = pd.to_numeric(self.clinical["censor"], errors="coerce")

        finite_mask = np.isfinite(self.clinical["os"].to_numpy()) & np.isfinite(self.clinical["censor"].to_numpy())
        if not finite_mask.all():
            dropped = int((~finite_mask).sum())
            print(f"Dropping {dropped} clinical rows with non-finite os/censor")
            self.clinical = self.clinical.loc[finite_mask].copy()

        sample_mask = self.clinical["id"].isin(self.data.obs_names)
        if not sample_mask.all():
            dropped = int((~sample_mask).sum())
            print(f"Dropping {dropped} clinical rows missing expression data")
            self.clinical = self.clinical.loc[sample_mask].copy()
    
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
        if len(uncensored_df) < n_bins:
            raise ValueError(f"Not enough uncensored samples to build {n_bins} survival bins")
        disc_labels, q_bins = pd.qcut(uncensored_df['os'], q=n_bins, retbins=True, labels=False)

        # q_bins[-1] = 1e6  # set rightmost edge to be infinite
        q_bins[-1] = self.clinical['os'].max() + eps
        q_bins[0] = 0  # set leftmost edge to be 0
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.clinical['os'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        if disc_labels.isna().any():
            bad_rows = self.clinical.loc[disc_labels.isna(), ['id', 'os', 'censor']]
            raise ValueError(f"Found {len(bad_rows)} samples that could not be assigned to a survival bin:\n{bad_rows.head(10).to_string(index=False)}")
        self.clinical.insert(1, 'label', disc_labels.to_numpy(dtype=np.int64))
        # bins = q_bins
