import numpy as np
import json
from typing import List, Tuple
from pathlib import Path

def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = False,
    include_zero_gene: bool = True,
    cls_id: str = "<cls>",
) -> List[Tuple]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_names, counts) of non zero gene expressions.
    """
    
    current_dir = Path(__file__).resolve().parent
    vocab_path = current_dir / "vocab.json"

    with open(vocab_path, "r") as f:
        vocab_map = json.load(f)

    if data.shape[0] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[0]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    
    # data = data[0]

    if include_zero_gene:
        values = data
        genes = gene_ids
    else:
        idx = np.nonzero(data)
        values = [data[i] for i in idx]
        genes = [gene_ids[i] for i in idx]
        values = np.asarray(values)
        genes = np.asarray(genes)
    if append_cls:
        genes = np.insert(genes, 0, cls_id)
        values = np.insert(values, 0, 0)
    if return_pt:
        import torch
        genes = torch.tensor([vocab_map.get(x, 0) for x in genes], dtype=torch.int64)
        values = torch.from_numpy(values).float()

    return genes, values


class scGPTTokenizer:

    def __init__(self):
        pass

    @classmethod
    def tokenize_cell_vectors(cls, data, gene_names):
        """
        Tokenizing single-cell gene expression vectors formatted as anndata types
        """
        return tokenize_batch(data, gene_names)
