# DisGlioma: Radiogenomics-based Distilled Prompt Learning for Non-invasive Survival Prediction in Adult Diffuse Glioma

## Table of Contents
- [Introduction](#introduction)
- [Data preparation](#data-preparation)
- [Requirements](#requirements)
- [Training](#train)
- [License & Citation](#license--citation)

## Introduction

![DisPro](./imgs/Figure1.png)

We introduce DisGlioma, a radiogenomics multimodal framework that integrates distilled genetic embeddings with MRI features for non-invasive survival prediction in adult diffuse gliomas. 
DisGlioma leverages biological-informed neural network and large language model with prompt learning to distill subtype-specific genetic representations, which are then fused with MRI features through cross-attention. During inference, DisGlioma requires only preoperative MRI and clinical information, enabling more accurate and clinically applicable non-invasive survival prediction. 

## Data preparation
### MR Images
The patients confirmed to have a complete series of the requisite MRI sequences(T1W,T2W,T1CE,FLAIR), was processed further using the Cancer Imaging Phenomics Toolkit(CaPTk), version 1.9.0, with following steps: \
(1) re-orientation to a reference coordinate system (here, left-posterior-superior (LPS)); \
(2) co-registration and resampling to an isotropic resolution of $1mm^3$ based on a common anatomical SRI24 atlas, with spatial dimensions of $240 \times 240 \times 155$ ( $height \times weight \times depth$ ) voxels; \
(3) removes non-brain structures (skull-stripping) using HD-BET; \
(4) all datasets were performed N4 bias corrected to remove RF inhomogeneities, and intensity normalized to zero-mean and unit variance.

The merged multi-sequence MRI data for each patient are stored as .`nii.gz` files in `./0-data/image/`.

### Gene Expression Data
Multiple gene expression datasets are used in this study. These data consist of mRNA sequencing and microarray profiles derived from multiple platforms. The bulk RNA-seq data is normalized to $\log_2(\text{FPKM} + 1)$, while the microarray data remains unaltered. After that, these datasets are merged together and applied the ComBat algorithm to reduce cross-platform differences and batch effects.

The merged gene expression data are stored in `./0-data/bulk/`, as `h5ad` format with a data matrix of `n_obs × n_vars`, where `obs` includes `sample_id` and `batch` (i.e. dataset labels), and `var` includes `gene symbols`.

### Clinical & Survival Information
Clinical and survival information are stored in CSV format, with the header shown below.
```
sample_id | os | censor | sex | dataset_tag
```
where `os` denotes overall survival time in months, and `censor` indicates the survival event: `censor = 1` means deceased, and `censor = 0` means alive.

**Before start training, please make sure all processed data are stored according to the file tree below**. 

```
0-data/
├── bulk/          # Gene expression data
│    ├── gene_expr_public.h5ad
│    ├── gene_expr_in_house.h5ad
├── clincial/      # Clinical & Survival information     
│    ├── clinical_gene.csv
│    ├── clinical_gene_external.csv
│    ├── clinical_image.csv
│    ├── clinical_image_external.csv            
├── image/         # MRI dataset
│    ├── dataset1/          
│    │   ├── patient1/
│    │   │   ├── patient1.nii.gz
│    ├── dataset2/
│    │   ...
│    └── datasetn/      
└── pathway_info.csv     # Pathway annotation
```

## Requirements
Key Requirements list here:
```
einops==0.8.2
itk==5.4.5
lifelines==0.30.3
monai==1.5.2
lightning==2.6.1
pandas==2.3.3
pytorch-lightning==2.6.1
scanpy==1.12.1
scikit-survival==0.27.0
torch == 2.8.0
torchsurv==0.1.6
torch-geometric==2.7.0
transformers==5.5.4
```

This study relies on Python 3.13 and CUDA 12.8. 
See `requirements.txt` for the full list.

## Training
DisGlioma comprises a gene branch and an image branch. The Gene branch and imaging branch of DisGlioma are trained separately. 
### Stage 1: Training Gene Branch
Before training, the scGPT weights should be downloaded from
#### Training BINN
First, the BINN in the Gene branch was trained using the NLL loss as the optimization objective to obtain pathway-level embeddings and a global embedding, and hierarchical clustering was then used to annotate each sample in the dataset with a subtype label. 

```bash
bash training_scripts/1-train_binn.sh
```
#### Genetic Embeddings Distillation
Next, the BINN was frozen, and prompt learning with a pretrained BioClinicalBERT was used to perform subtype-specific genetic embeddings distillation. During this process, only the prompt embeddings were optimized, while both the BINN and the pretrained BioClinicalBERT remained frozen. 

```bash
bash training_scripts/2-genetic_distillation.sh
```

### Stage 2: Training Image Branch
After completing the training of the gene branch, the image branch of DisGlioma was trained. First, the visual encoder extracted visual features and mapped them to subtypes. During this mapping process, $\mathcal{L}_{surv}$ , $\mathcal{L}_{cls}$ , and $\mathcal{L}_{center}$ were used as optimization objectives to train the visual encoder.

Next, the visual encoder was frozen, and the visual features and subtype-specific genetic features were fused through a cross-attention module. $\mathcal{L}_{surv}$ was used to supervise the model and improve survival prediction performance. 

```bash
bash training_scripts/3-map_vision.sh
```

Finally, both the visual encoder and the cross-attention module were frozen, and the enhanced visual features together with the encoded clinical information were fed into the risk decoder to enable non-invasive prediction based on preoperative multimodal information. In this final step, only the risk decoder was trained. 

```bash
bash training_scripts/4-train_disglioma.sh
```

## License & Citation
This project is licensed under the GPL-3.0 License.
