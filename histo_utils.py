import argparse
import json
import os
import pdb
import random
from itertools import cycle
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import pairwise_distances

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from sksurv.metrics import cumulative_dynamic_auc
from scipy.stats import percentileofscore, pearsonr, spearmanr
from sklearn.metrics import r2_score

from PIL import Image, ImageDraw, ImageFont
import cv2
import h5py

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

import gc
import seaborn as sns


def eval_transforms(pretrained=False, crop_size=224):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transforms_val = transforms.Compose(
        [
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return transforms_val


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


class Dataset_All_Bags(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]


# training
# 256x256 --> 224x224, 96x96 (the same object) --> dino(teature, student) -->

# using the encoder
# 256x256 --> crop_size(128x128 or 224x224) --> (center_x, center_y) --> 128x128 or 224x224 --> resnet18 --> 512


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


label_col_dict = {
    # 'stage': 'ajcc_pathologic_stage',
    'subtype': 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype',
    'Molecular_Subtype': 'Molecular_Subtype',  # this is the same as subtype
    'IHC_HER2': 'IHC_HER2',
    'HistoAnno': 'HistoAnno',
    'CLS_Molecular_Subtype': 'CLS_Molecular_Subtype',  # this is the same as subtype
    'CLS_IHC_HER2': 'CLS_IHC_HER2',
    'CLS_HistoAnno': 'CLS_HistoAnno',
    'SurvivalCls': 'disc_label',
    'TP53': 'TP53',
    'PIK3CA': 'PIK3CA',
    'KRAS': 'KRAS',
    'CDH1_cls': 'CDH1_cls',
    'GATA3_cls': 'GATA3_cls',
    'PIK3CA_cls': 'PIK3CA_cls',
    'TP53_cls': 'TP53_cls',
    'KRAS_cls': 'KRAS_cls',
    'ARID1A_cls': 'ARID1A_cls',
    'PIK3R1_cls': 'PIK3R1_cls',
    'PTEN_cls': 'PTEN_cls',
    'APC_cls': 'APC_cls',
    'ATRX_cls': 'ATRX_cls',
    'IDH1_cls': 'IDH1_cls',
    'BRAF_cls': 'BRAF_cls'
}
classification_dict_all = {
    # 'stage': ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
    'subtype': ['LumA', 'LumB', 'Basal', 'HER2E', 'normal-like'],
    'Molecular_Subtype': ['LumA', 'LumB', 'Basal', 'HER2E', 'normal-like'],
    'IHC_HER2': ['Negative', 'Positive', 'Other'],
    'HistoAnno': ['Invasive lobular carcinoma', 'Invasive ductal carcinoma', 'Other'],
    'CLS_Molecular_Subtype': ['LumA', 'LumB', 'Basal', 'HER2E', 'normal-like'],
    'CLS_IHC_HER2': ['Negative', 'Positive', 'Other'],
    'CLS_HistoAnno': ['Invasive lobular carcinoma', 'Invasive ductal carcinoma', 'Other'],
    'SurvivalCls': ['level0', 'level1', 'level2', 'level3'],
    'TP53': ['Gain_Or_Unknown', 'Loss', 'Other'],
    'PIK3CA': ['Loss_Or_Unknown', 'Gain', 'Other'],
    'KRAS': ['Loss_Or_Unknown', 'Gain', 'Other'],
    # 'TP53_cls': ['Gain_Or_Unknown', 'Loss', 'Other'],
    # 'PIK3CA_cls': ['Loss_Or_Unknown', 'Gain', 'Other'],
    # 'KRAS_cls': ['Loss_Or_Unknown', 'Gain', 'Other'],
    'CDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'GATA3_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'PIK3CA_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
    'TP53_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'KRAS_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
    'ARID1A_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'PIK3R1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'PTEN_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'APC_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'ATRX_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss', 'Other'],
    'IDH1_cls': ['Gain_Or_Unknown_Or_NaN', 'Loss_Or_Switch', 'Other'],
    'BRAF_cls': ['Loss_Or_Unknown_Or_NaN', 'Gain', 'Other'],
}
ignore_index_dict = {
    # 'stage': 3,
    'subtype': -100,
    'Molecular_Subtype': 4,
    'IHC_HER2': 2,
    'HistoAnno': -100,
    'CLS_Molecular_Subtype': 4,
    'CLS_IHC_HER2': 2,
    'CLS_HistoAnno': -100,
    'SurvivalCls': -100,
    'TP53': 2,
    'PIK3CA': 2,
    'KRAS': 2,
    'CDH1_cls': 2,
    'GATA3_cls': 2,
    'PIK3CA_cls': 2,
    'TP53_cls': 2,
    'KRAS_cls': 2,
    'ARID1A_cls': 2,
    'PIK3R1_cls': 2,
    'PTEN_cls': 2,
    'APC_cls': 2,
    'ATRX_cls': 2,
    'IDH1_cls': 2,
    'BRAF_cls': 2
}

CytoSig_list_all = ['CytoSig_Activin_A',
                    'CytoSig_BDNF',
                    'CytoSig_BMP2',
                    'CytoSig_BMP4',
                    'CytoSig_BMP6',
                    'CytoSig_CD40L',
                    'CytoSig_CXCL12',
                    'CytoSig_EGF',
                    'CytoSig_FGF2',
                    'CytoSig_GCSF',
                    'CytoSig_GDF11',
                    'CytoSig_GMCSF',
                    'CytoSig_HGF',
                    'CytoSig_IFN1',
                    'CytoSig_IFNG',
                    'CytoSig_IFNL',
                    'CytoSig_IL10',
                    'CytoSig_IL12',
                    'CytoSig_IL13',
                    'CytoSig_IL15',
                    'CytoSig_IL17A',
                    'CytoSig_IL1A',
                    'CytoSig_IL1B',
                    'CytoSig_IL2',
                    'CytoSig_IL21',
                    'CytoSig_IL22',
                    'CytoSig_IL27',
                    'CytoSig_IL3',
                    'CytoSig_IL36',
                    'CytoSig_IL4',
                    'CytoSig_IL6',
                    'CytoSig_LIF',
                    'CytoSig_LTA',
                    'CytoSig_MCSF',
                    'CytoSig_NO',
                    'CytoSig_OSM',
                    'CytoSig_TGFB1',
                    'CytoSig_TGFB3',
                    'CytoSig_TNFA',
                    'CytoSig_TRAIL',
                    'CytoSig_TWEAK',
                    'CytoSig_VEGFA',
                    'CytoSig_WNT3A']

Receptor_Cytokine_list_all = [
    "Cytokine_Receptor_CSF_MCSF_GeneSum",
    "Cytokine_Receptor_CSF_MCSF_ReceptorSum",
    "Cytokine_Receptor_CSF_GMCSF_GeneSum",
    "Cytokine_Receptor_CSF_GMCSF_ReceptorSum",
    "Cytokine_Receptor_CSF_GCSF_GeneSum",
    "Cytokine_Receptor_CSF_GCSF_ReceptorSum",
    "Cytokine_Receptor_Interferon_IFNA_GeneSum",
    "Cytokine_Receptor_Interferon_IFNA_ReceptorSum",
    "Cytokine_Receptor_Interferon_IFNB_GeneSum",
    "Cytokine_Receptor_Interferon_IFNB_ReceptorSum",
    "Cytokine_Receptor_Interferon_IFNG_GeneSum",
    "Cytokine_Receptor_Interferon_IFNG_ReceptorSum",
    "Cytokine_Receptor_Interferon_IFNL_GeneSum",
    "Cytokine_Receptor_Interferon_IFNL_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL1A_GeneSum",
    "Cytokine_Receptor_Interleukin_IL1A_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL1B_GeneSum",
    "Cytokine_Receptor_Interleukin_IL1B_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL1RA_GeneSum",
    "Cytokine_Receptor_Interleukin_IL1RA_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL2_GeneSum",
    "Cytokine_Receptor_Interleukin_IL3_GeneSum",
    "Cytokine_Receptor_Interleukin_IL3_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL4_GeneSum",
    "Cytokine_Receptor_Interleukin_IL4_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL5_GeneSum",
    "Cytokine_Receptor_Interleukin_IL5_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL6_GeneSum",
    "Cytokine_Receptor_Interleukin_IL6_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL7_GeneSum",
    "Cytokine_Receptor_Interleukin_IL7_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL9_GeneSum",
    "Cytokine_Receptor_Interleukin_IL9_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL10_GeneSum",
    "Cytokine_Receptor_Interleukin_IL10_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL11_GeneSum",
    "Cytokine_Receptor_Interleukin_IL11_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL13_GeneSum",
    "Cytokine_Receptor_Interleukin_IL13_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL15_GeneSum",
    "Cytokine_Receptor_Interleukin_IL16_GeneSum",
    "Cytokine_Receptor_Interleukin_IL16_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL17A_GeneSum",
    "Cytokine_Receptor_Interleukin_IL17A_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL17C_GeneSum",
    "Cytokine_Receptor_Interleukin_IL17C_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL17F_GeneSum",
    "Cytokine_Receptor_Interleukin_IL17F_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL18_GeneSum",
    "Cytokine_Receptor_Interleukin_IL18_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL19_GeneSum",
    "Cytokine_Receptor_Interleukin_IL19_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL20_GeneSum",
    "Cytokine_Receptor_Interleukin_IL21_GeneSum",
    "Cytokine_Receptor_Interleukin_IL21_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL22_GeneSum",
    "Cytokine_Receptor_Interleukin_IL22_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL24_GeneSum",
    "Cytokine_Receptor_Interleukin_IL25_GeneSum",
    "Cytokine_Receptor_Interleukin_IL25_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL26_GeneSum",
    "Cytokine_Receptor_Interleukin_IL26_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL27_GeneSum",
    "Cytokine_Receptor_Interleukin_IL27_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL31_GeneSum",
    "Cytokine_Receptor_Interleukin_IL31_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL33_GeneSum",
    "Cytokine_Receptor_Interleukin_IL33_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL34_GeneSum",
    "Cytokine_Receptor_Interleukin_IL34_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL36A_GeneSum",
    "Cytokine_Receptor_Interleukin_IL36A_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL36B_GeneSum",
    "Cytokine_Receptor_Interleukin_IL36B_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL36G_GeneSum",
    "Cytokine_Receptor_Interleukin_IL36G_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL36RA_GeneSum",
    "Cytokine_Receptor_Interleukin_IL36RA_ReceptorSum",
    "Cytokine_Receptor_Interleukin_IL37_GeneSum",
    "Cytokine_Receptor_Interleukin_IL37_ReceptorSum",
    "Cytokine_Receptor_Interleukin_TSLP_GeneSum",
    "Cytokine_Receptor_Interleukin_TSLP_ReceptorSum",
    "Cytokine_Receptor_Interleukin_LIF_GeneSum",
    "Cytokine_Receptor_Interleukin_LIF_ReceptorSum",
    "Cytokine_Receptor_Interleukin_OSM_GeneSum",
    "Cytokine_Receptor_TNF_TNFA_GeneSum",
    "Cytokine_Receptor_TNF_LTA_GeneSum",
    "Cytokine_Receptor_TNF_LTB_GeneSum",
    "Cytokine_Receptor_TNF_CD40L_GeneSum",
    "Cytokine_Receptor_TNF_CD40L_ReceptorSum",
    "Cytokine_Receptor_TNF_FASL_GeneSum",
    "Cytokine_Receptor_TNF_FASL_ReceptorSum",
    "Cytokine_Receptor_TNF_CD27L_GeneSum",
    "Cytokine_Receptor_TNF_CD27L_ReceptorSum",
    "Cytokine_Receptor_TNF_CD30L_GeneSum",
    "Cytokine_Receptor_TNF_CD30L_ReceptorSum",
    "Cytokine_Receptor_TNF_41BBL_GeneSum",
    "Cytokine_Receptor_TNF_41BBL_ReceptorSum",
    "Cytokine_Receptor_TNF_TRAIL_GeneSum",
    "Cytokine_Receptor_TNF_OPGL_GeneSum",
    "Cytokine_Receptor_TNF_OPGL_ReceptorSum",
    "Cytokine_Receptor_TNF_ARPIL_GeneSum",
    "Cytokine_Receptor_TNF_LIGHT_GeneSum",
    "Cytokine_Receptor_TNF_TWEAK_GeneSum",
    "Cytokine_Receptor_TNF_TWEAK_ReceptorSum",
    "Cytokine_Receptor_TNF_BAFF_GeneSum",
    "Cytokine_Receptor_Unassigned_TGFB1_GeneSum",
    "Cytokine_Receptor_Unassigned_TGFB2_GeneSum",
    "Cytokine_Receptor_Unassigned_TGFB3_GeneSum",
    "Cytokine_Receptor_Unassigned_MIF_GeneSum",
    "Cytokine_Receptor_Extra_IFN1_GeneSum",
    "Cytokine_Receptor_Extra_IFN1_ReceptorSum",
    "Cytokine_Receptor_Extra_IL1_GeneSum",
    "Cytokine_Receptor_Extra_IL1_ReceptorSum",
    "Cytokine_Receptor_Extra_IL36_GeneSum",
    "Cytokine_Receptor_Extra_IL36_ReceptorSum"
]

MsigDB_Hallmark_list = [
    "HALLMARK_ADIPOGENESIS_sum",
    "HALLMARK_ALLOGRAFT_REJECTION_sum",
    "HALLMARK_ANDROGEN_RESPONSE_sum",
    "HALLMARK_ANGIOGENESIS_sum",
    "HALLMARK_APICAL_JUNCTION_sum",
    "HALLMARK_APICAL_SURFACE_sum",
    "HALLMARK_APOPTOSIS_sum",
    "HALLMARK_BILE_ACID_METABOLISM_sum",
    "HALLMARK_CHOLESTEROL_HOMEOSTASIS_sum",
    "HALLMARK_COAGULATION_sum",
    "HALLMARK_COMPLEMENT_sum",
    "HALLMARK_DNA_REPAIR_sum",
    "HALLMARK_E2F_TARGETS_sum",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION_sum",
    "HALLMARK_ESTROGEN_RESPONSE_EARLY_sum",
    "HALLMARK_ESTROGEN_RESPONSE_LATE_sum",
    "HALLMARK_FATTY_ACID_METABOLISM_sum",
    "HALLMARK_G2M_CHECKPOINT_sum",
    "HALLMARK_GLYCOLYSIS_sum",
    "HALLMARK_HEDGEHOG_SIGNALING_sum",
    "HALLMARK_HEME_METABOLISM_sum",
    "HALLMARK_HYPOXIA_sum",
    "HALLMARK_IL2_STAT5_SIGNALING_sum",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING_sum",
    "HALLMARK_INFLAMMATORY_RESPONSE_sum",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE_sum",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE_sum",
    "HALLMARK_KRAS_SIGNALING_DN_sum",
    "HALLMARK_KRAS_SIGNALING_UP_sum",
    "HALLMARK_MITOTIC_SPINDLE_sum",
    "HALLMARK_MTORC1_SIGNALING_sum",
    "HALLMARK_MYC_TARGETS_V1_sum",
    "HALLMARK_MYC_TARGETS_V2_sum",
    "HALLMARK_MYOGENESIS_sum",
    "HALLMARK_NOTCH_SIGNALING_sum",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION_sum",
    "HALLMARK_P53_PATHWAY_sum",
    "HALLMARK_PANCREAS_BETA_CELLS_sum",
    "HALLMARK_PEROXISOME_sum",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING_sum",
    "HALLMARK_PROTEIN_SECRETION_sum",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY_sum",
    "HALLMARK_SPERMATOGENESIS_sum",
    "HALLMARK_TGF_BETA_SIGNALING_sum",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB_sum",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE_sum",
    "HALLMARK_UV_RESPONSE_DN_sum",
    "HALLMARK_UV_RESPONSE_UP_sum",
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING_sum",
    "HALLMARK_XENOBIOTIC_METABOLISM_sum"
]

regression_list_all = [
                          'Stage',
                          'MDSC', 'CAF', 'M2', 'Dysfunction', 'CTL',
                          'SurvivalReg',
                          'TIDE_MDSC', 'TIDE_CAF', 'TIDE_M2', 'TIDE_Dysfunction', 'Cytotoxic_T_Lymphocyte', 'TIDE_Dys'
                      ] + CytoSig_list_all

reg_normed_columns_all = {
    'TCGA-BRCA':
        {
            'TIDE_MDSC': 0.2,
            'TIDE_M2': 0.2,
            'TIDE_CAF': 0.3,
            'Cytotoxic_T_Lymphocyte': 3.0,
            'TIDE_Dys': 0.25,
            'CytoSig_Activin_A': 6.0,
            'CytoSig_BDNF': 7.5,
            'CytoSig_BMP2': 15.0,
            'CytoSig_BMP4': 14.0,
            'CytoSig_BMP6': 6.0,
            'CytoSig_CD40L': 7.5,
            'CytoSig_CXCL12': 8.0,
            'CytoSig_EGF': 30.0,
            'CytoSig_FGF2': 15.0,
            'CytoSig_GCSF': 10.0,
            'CytoSig_GDF11': 10.0,
            'CytoSig_GMCSF': 10.0,
            'CytoSig_HGF': 15.0,
            'CytoSig_IFN1': 20.0,
            'CytoSig_IFNG': 15.0,
            'CytoSig_IFNL': 20.0,
            'CytoSig_IL10': 10.0,
            'CytoSig_IL12': 10.0,
            'CytoSig_IL13': 7.5,
            'CytoSig_IL15': 10.0,
            'CytoSig_IL17A': 10.0,
            'CytoSig_IL1A': 8.0,
            'CytoSig_IL1B': 10.0,
            'CytoSig_IL2': 15.0,
            'CytoSig_IL21': 11.0,
            'CytoSig_IL22': 20.0,
            'CytoSig_IL27': 15.0,
            'CytoSig_IL3': 10.0,
            'CytoSig_IL36': 10.0,
            'CytoSig_IL4': 11.0,
            'CytoSig_IL6': 10.0,
            'CytoSig_LIF': 10.0,
            'CytoSig_LTA': 6.0,
            'CytoSig_MCSF': 15.0,
            'CytoSig_NO': 15.0,
            'CytoSig_OSM': 12.0,
            'CytoSig_TGFB1': 20.0,
            'CytoSig_TGFB3': 20.0,
            'CytoSig_TNFA': 13.0,
            'CytoSig_TRAIL': 10.0,
            'CytoSig_TWEAK': 15.0,
            'CytoSig_VEGFA': 15.0,
            'CytoSig_WNT3A': 6.0
        },
    'TCGA-PAAD':
        {
            'TIDE_MDSC': 0.3,
            'TIDE_M2': 0.13,
            'TIDE_CAF': 0.4,
            'Cytotoxic_T_Lymphocyte': 2.5,
            'TIDE_Dys': 0.25,
            'CytoSig_Activin_A': 6.0,
            'CytoSig_BDNF': 8.0,
            'CytoSig_BMP2': 15.0,
            'CytoSig_BMP4': 12.0,
            'CytoSig_BMP6': 5.0,
            'CytoSig_CD40L': 8.0,
            'CytoSig_CXCL12': 6.0,
            'CytoSig_EGF': 30.0,
            'CytoSig_FGF2': 10.0,
            'CytoSig_GCSF': 12.0,
            'CytoSig_GDF11': 12.0,
            'CytoSig_GMCSF': 9.0,
            'CytoSig_HGF': 12.0,
            'CytoSig_IFN1': 15.0,
            'CytoSig_IFNG': 12.0,
            'CytoSig_IFNL': 12.0,
            'CytoSig_IL10': 10.0,
            'CytoSig_IL12': 8.0,
            'CytoSig_IL13': 6.0,
            'CytoSig_IL15': 6.0,
            'CytoSig_IL17A': 10.0,
            'CytoSig_IL1A': 7.0,
            'CytoSig_IL1B': 9.0,
            'CytoSig_IL2': 14.0,
            'CytoSig_IL21': 10.0,
            'CytoSig_IL22': 16.0,
            'CytoSig_IL27': 13.0,
            'CytoSig_IL3': 12.0,
            'CytoSig_IL36': 10.0,
            'CytoSig_IL4': 13.0,
            'CytoSig_IL6': 8.0,
            'CytoSig_LIF': 10.0,
            'CytoSig_LTA': 6.0,
            'CytoSig_MCSF': 12.0,
            'CytoSig_NO': 11.0,
            'CytoSig_OSM': 10.0,
            'CytoSig_TGFB1': 16.0,
            'CytoSig_TGFB3': 16.0,
            'CytoSig_TNFA': 11.0,
            'CytoSig_TRAIL': 10.0,
            'CytoSig_TWEAK': 8.0,
            'CytoSig_VEGFA': 15.0,
            'CytoSig_WNT3A': 6.0
        }
}

PAN_CANCER_SITES = ['BRCA',
                    'PAAD',
                    'CHOL',
                    'UCS',
                    'DLBC',
                    'UVM',
                    'UCEC',
                    'MESO',
                    'ACC',
                    'KICH',
                    'THYM',
                    'TGCT',
                    'PCPG',
                    'ESCA',
                    'SARC',
                    'CESC',
                    'PRAD',
                    'THCA',
                    'OV',
                    'KIRC',
                    'BLCA',
                    'STAD',
                    'SKCM',
                    'READ',
                    'LUSC',
                    'LUAD',
                    'LIHC',
                    'LGG',
                    'KIRP',
                    'HNSC',
                    'COAD',
                    'GBM']


def clean_csv(slide_data, mask_root, model_name,
              classification_dict=None, regression_list=None, ignore_index_dict=None,
              only_step2=False):
    if not only_step2:
        # remove invalid files
        invalid_CLIDs = {'TCGA-5P-A9KA', 'TCGA-5P-A9KC', 'TCGA-HT-7483', 'TCGA-UZ-A9PQ'}
        if 'patch' in model_name:
            existed_prefixes = set([os.path.basename(f).replace('.txt', '')
                                    for f in glob.glob(os.path.join(mask_root, '*.txt'))])
        elif 'pretrain' in model_name:
            existed_prefixes = set([os.path.basename(f).replace('.pt', '')
                                    for f in glob.glob(os.path.join(mask_root, '*.pt'))])
        else:
            existed_prefixes = set()
        invalid_inds = []
        for ind, DX_filename in enumerate(slide_data['DX_filename'].values):
            fileprefix = os.path.basename(DX_filename).replace('.svs', '')
            if fileprefix[:12] in invalid_CLIDs or fileprefix not in existed_prefixes:
                invalid_inds.append(ind)
        if len(invalid_inds) > 0:
            slide_data = slide_data.drop(invalid_inds).reset_index(drop=True)

        if len(classification_dict) > 0:
            for k, v in classification_dict.items():
                if k in slide_data:
                    slide_data[k] = slide_data[k].fillna(ignore_index_dict[k])
        if len(regression_list) > 0:
            for k in regression_list:
                if k in slide_data:
                    slide_data = slide_data[slide_data[k].isnull() == False]

    slide_data = slide_data.reset_index(drop=True)

    if 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype' in slide_data:
        SUBTYPES_DICT = {
            'LumA': 0,
            'LumB': 1,
            'Basal': 2,
            'HER2E': 3,
            'normal-like': 4,
            'normal': 4,
            'CLOW': 4
        }
        slide_data['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = \
            slide_data['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].fillna('normal')
        slide_data['CLS_Molecular_Subtype'] = \
            slide_data['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].map(SUBTYPES_DICT)
        slide_data['CLS_Molecular_Subtype'] = slide_data['CLS_Molecular_Subtype'].fillna(4)

        slide_data['HER2.newly.derived'] = \
            slide_data['HER2.newly.derived'].fillna('Other')
        IHC_HER2_dict = {k: 2 for k in slide_data['HER2.newly.derived'].value_counts().index}
        IHC_HER2_dict['Positive'] = 1
        IHC_HER2_dict['Negative'] = 0
        slide_data['CLS_IHC_HER2'] = slide_data['HER2.newly.derived'].map(IHC_HER2_dict)
        slide_data['CLS_IHC_HER2'] = slide_data['CLS_IHC_HER2'].fillna(2)

        slide_data['2016_Histology_Annotations'] = \
            slide_data['2016_Histology_Annotations'].fillna('Other')
        HistoAnno_dict = {k: 2 for k in slide_data['2016_Histology_Annotations'].value_counts().index}
        HistoAnno_dict['Invasive ductal carcinoma'] = 1
        HistoAnno_dict['Invasive lobular carcinoma'] = 0
        slide_data['CLS_HistoAnno'] = slide_data['2016_Histology_Annotations'].map(HistoAnno_dict)
        slide_data['CLS_HistoAnno'] = slide_data['CLS_HistoAnno'].fillna(2)

    if 'Response' in slide_data:
        ResponseDict = {k: 4 for k in slide_data['Response'].value_counts().index}
        ResponseDict.update({'PD': 0, 'SD': 1, 'PR': 2, 'CR': 3})
        slide_data['ResponseGroup'] = slide_data['Response'].map(ResponseDict)
        slide_data['ResponseGroup'] = slide_data['ResponseGroup'].fillna(4)

    return slide_data


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


def silhouette_score_modified(features, labels, metric='cosine'):
    unique_labels = np.unique(labels)
    scores, auc_scores, auc_scores2 = [], [], []
    for label in unique_labels:
        inds0 = np.where(labels != label)[0]  # between-class
        inds1 = np.where(labels == label)[0]  # within-class
        X0, X1 = features[inds0], features[inds1]
        dist0 = pairwise_distances(X1, X0, n_jobs=4, metric=metric)  # between distances
        dist1 = pairwise_distances(X1, X1, n_jobs=4, metric=metric)  # within distances
        a, b = np.mean(dist1), np.mean(dist0)
        score = (b - a) / max(a, b)
        scores.append(score)

        X0 = dist0.flatten()
        X1 = dist1.flatten()
        del dist0, dist1
        Y0 = np.zeros_like(X1, dtype=np.int32)
        Y1 = np.ones_like(X0, dtype=np.int32)
        X = np.concatenate([X1, X0], axis=0)
        Y = np.concatenate([Y0, Y1], axis=0)
        del X0, X1, Y0, Y1
        auc_score = roc_auc_score(Y, X)
        auc_scores.append(auc_score)

        gc.collect()
    return np.mean(scores), np.mean(auc_scores)


def silhouette_score_modified_v2(features, labels, metric='cosine'):
    unique_labels = np.unique(labels)
    scores = []
    for label in unique_labels:
        inds0 = np.where(labels != label)[0]  # between-class
        inds1 = np.where(labels == label)[0]  # within-class
        X0, X1 = features[inds0], features[inds1]
        dist0 = pairwise_distances(X1, X0, n_jobs=4, metric=metric)  # between distances
        dist1 = pairwise_distances(X1, X1, n_jobs=4, metric=metric)  # within distances
        a, b = np.mean(dist1), np.mean(dist0)
        score = (b - a) / max(a, b)

        X0 = dist0.flatten()
        X1 = dist1.flatten()
        del dist0, dist1
        Y0 = np.zeros_like(X1, dtype=np.int32)
        Y1 = np.ones_like(X0, dtype=np.int32)
        X = np.concatenate([X1, X0], axis=0)
        Y = np.concatenate([Y0, Y1], axis=0)
        del X0, X1, Y0, Y1
        auc_score = roc_auc_score(Y, X)

        scores.append([score, auc_score])

        gc.collect()

        if len(unique_labels) <= 2:
            break

    return np.array(scores)


def draw_kde_plots(feats_embedding_2D, labels, labels_dict, save_root, save_postfix):
    # draw KDE plot
    valid_labels = np.unique(labels)
    label_index = 0
    nrows = int(np.ceil(np.sqrt(len(valid_labels))))
    fig, axes = plt.subplots(nrows, nrows, figsize=(15, 15))
    for i in range(nrows):
        for j in range(nrows):
            if label_index < len(valid_labels):
                label = valid_labels[label_index]
                label_name = labels_dict[label]
                ind = np.where(labels == label)[0]
                if len(ind) < 3:
                    axes[i, j].scatter(feats_embedding_2D[ind, 0], feats_embedding_2D[ind, 1])
                else:
                    sns.kdeplot(ax=axes[i, j], x=feats_embedding_2D[ind, 0], y=feats_embedding_2D[ind, 1],
                                fill=True)
                axes[i, j].set_title('{}({})'.format(label_name, len(ind)))

                label_index += 1
            else:
                axes[i, j].axis('off')
    plt.savefig('{}/{}_kdeplot.png'.format(save_root, save_postfix),
                bbox_inches='tight', dpi=600)
    plt.close(fig)

    # random control
    labels = np.copy(labels)
    np.random.shuffle(labels)
    valid_labels = np.unique(labels)
    nrows = int(np.ceil(np.sqrt(len(valid_labels))))
    fig, axes = plt.subplots(nrows, nrows, figsize=(15, 15))
    label_index = 0
    for i in range(nrows):
        for j in range(nrows):
            if label_index < len(valid_labels):
                label = valid_labels[label_index]
                label_name = labels_dict[label]
                ind = np.where(labels == label)[0]
                if len(ind) < 3:
                    axes[i, j].scatter(feats_embedding_2D[ind, 0], feats_embedding_2D[ind, 1])
                else:
                    sns.kdeplot(ax=axes[i, j], x=feats_embedding_2D[ind, 0], y=feats_embedding_2D[ind, 1],
                                fill=True)
                axes[i, j].set_title('{}({})'.format(label_name, len(ind)))
                label_index += 1
            else:
                axes[i, j].axis('off')
    plt.savefig('{}/{}_kdeplot_rc.png'.format(save_root, save_postfix),
                bbox_inches='tight', dpi=600)
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='v1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--project_name', type=str, default='')
    parser.add_argument('--csv_filename', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--split_label_name', type=str, default='vital_status')
    parser.add_argument('--combine_train_val_splits', action='store_true', default=False)
    parser.add_argument('--splits_dir', type=str, default='')

    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--fixed_backbone', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='patch')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_root', type=str, default='.')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--optim_type', type=str, default='adam')

    parser.add_argument('--num_patches', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--norm_type', type=str, default='mean_std', choices=['mean_std', 'zero_one', 'patch_mean_std'])

    parser.add_argument('--vae_loss_type', type=str, default='')
    parser.add_argument('--bce_loss_coeff', type=float, default=1.0)
    parser.add_argument('--kl_loss_coeff', type=float, default=0)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--surv_loss_type', type=str, default='')
    parser.add_argument('--surv_alpha', type=float, default=0.4)
    parser.add_argument('--surv_loss_coeff', type=float, default=1.0)
    parser.add_argument('--cls_task_keys', type=str, default='None')
    parser.add_argument('--cls_loss_type', type=str, default='')
    parser.add_argument('--cls_loss_coeff', type=str, default='')  # split by comma
    parser.add_argument('--reg_task_keys', type=str, default='None')
    parser.add_argument('--reg_loss_type', type=str, default='')
    parser.add_argument('--reg_loss_coeff', type=str, default='')  # split by comma
    parser.add_argument('--regu_loss_type', type=str, default='')
    parser.add_argument('--regu_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--use_weighted_sampling', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--cluster_strategy', type=str, default='density_log')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--cluster_task_index', type=int, default=0)
    parser.add_argument('--cluster_task_name', type=str, default='HistoAnno')
    parser.add_argument('--cluster_subset', type=str, default='test')
    parser.add_argument('--cluster_feat_name', type=str, default='feat_before_attention_feat')
    parser.add_argument('--attention_arch', type=str, default='shared_attention',
                        choices=['shared_attention', 'multiple_attention'])

    parser.add_argument('--split_num', type=int, default=0)
    parser.add_argument('--encoder_type', type=str, default='None')
    parser.add_argument('--encoder_ckpt_path', type=str, default="")

    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    parser.add_argument('--cache_root', default='None', type=str,
                        help='cache_root for offline data')
    parser.add_argument('--masks_dir', default='./', type=str,
                        help='directory for masks')
    parser.add_argument('--feats_dir', default='./', type=str,
                        help='directory for feats')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='accumulate iterations for training')

    # MoE
    parser.add_argument('--moe_type', default='None', type=str)
    parser.add_argument('--expert_count_per_node', default=1, type=int)
    parser.add_argument('--hidden_size_per_expert', default=512, type=int)

    return parser.parse_args()


def get_args_multiple_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='v1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--project_name', type=str, default='')  # separated by |, e.g., TCGA-BRCA|TCGA-PAAD
    parser.add_argument('--csv_filename', type=str, default='')  # separated by |, e.g., aa.csv|bb.csv
    parser.add_argument('--dataset_root', type=str, default='')  # separated by |
    parser.add_argument('--dataset_name', type=str, default='')  # separated by |
    parser.add_argument('--split_label_name', type=str, default='vital_status')  # separated by |
    parser.add_argument('--combine_train_val_splits', type=str, default="False")  # separated by |
    parser.add_argument('--combine_train_val_test_splits', type=str, default="False")  # separated by |
    parser.add_argument('--splits_dir', type=str, default='')  # separated by |
    parser.add_argument('--batch_size', type=str, default='')  # separated by |
    parser.add_argument('--split_num', type=str, default='')  # separated by |
    parser.add_argument('--cache_root', default='None', type=str, help='cache_root for offline data')  # separated by |
    parser.add_argument('--masks_dir', default='./', type=str, help='directory for masks')  # separated by |
    parser.add_argument('--feats_dir', default='./', type=str, help='directory for feats')  # separated by |
    parser.add_argument('--accum_iter', default='1', type=str,
                        help='accumulate iterations for training')  # separated by |

    parser.add_argument('--num_patches', type=str, default=64)  # separated by |
    parser.add_argument('--num_channels', type=str, default=3)  # separated by |
    parser.add_argument('--image_size', type=str, default=128)  # separated by |
    parser.add_argument('--norm_type', type=str, default='mean_std')  # separated by |
    parser.add_argument('--reg_normalization_type', type=str, default='None')  # separated by |
    parser.add_argument('--use_weighted_sampling', type=str, default="False")  # separated by |
    parser.add_argument('--focal_gamma', type=str, default="2.0")  # separated by |

    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--fixed_backbone', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='patch')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_root', type=str, default='.')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--regu_loss_type', type=str, default='')
    parser.add_argument('--regu_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--training_strategy', type=str, default="type0")
    parser.add_argument('--training_sub_epochs', type=int, default=1)
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
    parser.add_argument('--use_early_stopping', default='False', type=str)

    parser.add_argument('--vae_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--bce_loss_coeff', type=str, default='1.0')  # separated by |
    parser.add_argument('--kl_loss_coeff', type=str, default='0')  # separated by |
    parser.add_argument('--z_dim', type=str, default='512')  # separated by |

    parser.add_argument('--surv_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--surv_alpha', type=str, default='0.4')  # separated by |
    parser.add_argument('--surv_loss_coeff', type=str, default='1.0')  # separated by |

    parser.add_argument('--cls_task_keys', type=str, default='None')  # separated by |
    parser.add_argument('--cls_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--cls_loss_coeff', type=str, default='')  # separated by |, then split by comma

    parser.add_argument('--reg_task_keys', type=str, default='None')  # separated by |
    parser.add_argument('--reg_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--reg_loss_coeff', type=str, default='')  # separated by |, then split by comma

    parser.add_argument('--cluster_strategy', type=str, default='density_log')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--cluster_task_index', type=int, default=0)
    parser.add_argument('--cluster_task_name', type=str, default='HistoAnno')
    parser.add_argument('--cluster_subset', type=str, default='test')
    parser.add_argument('--cluster_feat_name', type=str, default='feat_before_attention_feat')
    parser.add_argument('--attention_arch', type=str, default='shared_attention',
                        choices=['shared_attention', 'multiple_attention'])

    parser.add_argument('--encoder_type', type=str, default='None')
    parser.add_argument('--encoder_ckpt_path', type=str, default="")

    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    # MoE
    parser.add_argument('--moe_type', default='None', type=str)
    parser.add_argument('--expert_count_per_node', default=1, type=int)
    parser.add_argument('--hidden_size_per_expert', default=512, type=int)

    return parser.parse_args()


HF_MODELS_DICT = {
    'resnet18': ['ResNetModel', 'microsoft/resnet-18', 'pooler_output'],
    'resnet50': ['ResNetModel', 'microsoft/resnet-50', 'pooler_output'],
    'beit': ['BeitModel', 'microsoft/beit-base-patch16-224-pt22k', 'pooler_output'],
    'bit': ['BitModel', 'google/bit-50', 'pooler_output', 'pooler_output'],
    'convnext': ['ConvNextModel', 'facebook/convnext-tiny-224', 'pooler_output'],
    # 'cvt': ['CvtModel', 'microsoft/cvt-13', 'cls_token_value'],
    # 'deit': ['DeiTModel', 'facebook/deit-base-distilled-patch16-224', 'pooler_output'],
    # 'dinat': ['DinatModel', 'shi-labs/dinat-mini-in1k-224'],
    # 'dpt': ['DPTModel', 'Intel/dpt-large', 'pooler_output'],
    # 'efficientformer': ['EfficientFormerModel', 'snap-research/efficientformer-l1-300'],
    # 'efficientnet': ['EfficientNetModel', 'google/efficientnet-b7'],
    # 'glpn': ['GLPNModel', 'vinvino02/glpn-kitti'],
    # 'imagegpt': ['ImageGPTModel', 'openai/imagegpt-small'],
    # 'levit': ['LevitModel', 'facebook/levit-128S', 'pooler_output'],
    'mobilenetv1': ['MobileNetV1Model', 'google/mobilenet_v1_1.0_224', 'pooler_output'],
    'mobilenetv2': ['MobileNetV2Model', 'google/mobilenet_v2_1.0_224', 'pooler_output'],
    'mobilevit': ['MobileViTModel', 'apple/mobilevit-small', 'pooler_output'],
    # 'nat': ['NatModel', 'shi-labs/nat-mini-in1k-224'],
    # 'poolformer': ['PoolFormerModel', 'sail/poolformer_s12'],
    # 'swin': ['SwinModel', 'microsoft/swin-tiny-patch4-window7-224', 'pooler_output'],
    'swinv2': ['Swinv2Model', 'microsoft/swinv2-tiny-patch4-window8-256', 'pooler_output'],
    'vit': ['ViTModel', 'google/vit-base-patch16-224-in21k', 'pooler_output'],
    'vithybrid': ['ViTHybridModel', 'google/vit-hybrid-base-bit-384', 'pooler_output'],
    # 'vitmae': ['ViTMAEModel', 'facebook/vit-mae-base'],
    # 'vitmsn': ['ViTMSNModel', 'facebook/vit-msn-small']
    'mobilenetv3': ['timm_mobilenetv3', '', 'flatten']
}


# 20230305
def get_args_v2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='v1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only_vis', action='store_true')  # only do tensorboard visualization
    parser.add_argument('--image_ext', type=str, default='.svs')

    parser.add_argument('--project_name', type=str, default='')  # separated by |, e.g., TCGA-BRCA|TCGA-PAAD
    parser.add_argument('--csv_filename', type=str, default='')  # separated by |, e.g., aa.csv|bb.csv
    parser.add_argument('--dataset_root', type=str, default='')  # separated by |
    parser.add_argument('--dataset_name', type=str, default='')  # separated by |
    parser.add_argument('--data_version', type=str, default='')
    parser.add_argument('--split_label_name', type=str, default='vital_status')  # separated by |
    parser.add_argument('--combine_train_val_splits', type=str, default="False")  # separated by |
    parser.add_argument('--combine_train_val_test_splits', type=str, default="False")  # separated by |
    parser.add_argument('--splits_dir', type=str, default='')  # separated by |
    parser.add_argument('--batch_size', type=int, default=1)  # separated by |
    parser.add_argument('--split_num', type=int, default=0)  # separated by |
    parser.add_argument('--cache_root', default='None', type=str, help='cache_root for offline data')  # separated by |
    parser.add_argument('--masks_dir', default='./', type=str, help='directory for masks')  # separated by |
    parser.add_argument('--feats_dir', default='./', type=str, help='directory for feats')  # separated by |
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='accumulate iterations for training')  # separated by |

    parser.add_argument('--num_patches', type=int, default=64)  # separated by |
    parser.add_argument('--num_channels', type=int, default=3)  # separated by |
    parser.add_argument('--image_size', type=int, default=224)  # separated by |
    parser.add_argument('--norm_type', type=str, default='mean_std')  # separated by |
    parser.add_argument('--reg_normalization_type', type=str, default='None')  # separated by |
    parser.add_argument('--use_weighted_sampling', type=str, default="False")  # separated by |
    parser.add_argument('--focal_gamma', type=float, default=3.0)  # separated by |

    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--fixed_backbone', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='patch')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_root', type=str, default='.')
    parser.add_argument('--final_save_root', type=str, default='None')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--regu_loss_type', type=str, default='None')
    parser.add_argument('--regu_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--vae_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--bce_loss_coeff', type=float, default=1.0)  # separated by |
    parser.add_argument('--kl_loss_coeff', type=float, default=0.0)  # separated by |
    parser.add_argument('--z_dim', type=int, default=512)  # separated by |

    parser.add_argument('--surv_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--surv_alpha', type=float, default=0.3)  # separated by |
    parser.add_argument('--surv_loss_coeff', type=float, default=1.0)  # separated by |

    parser.add_argument('--cls_task_keys', type=str, default='None')  # separated by |
    parser.add_argument('--cls_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--cls_loss_coeff', type=float, default=1.0)  # separated by |, then split by comma

    parser.add_argument('--reg_task_keys', type=str, default='None')  # separated by |
    parser.add_argument('--reg_loss_type', type=str, default='')  # separated by |
    parser.add_argument('--reg_loss_coeff', type=float, default=1.0)  # separated by |, then split by comma

    parser.add_argument('--cluster_strategy', type=str, default='density_log')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--cluster_task_index', type=int, default=0)
    parser.add_argument('--cluster_task_name', type=str, default='HistoAnno')
    parser.add_argument('--cluster_subset', type=str, default='test')
    parser.add_argument('--cluster_feat_name', type=str, default='feat_before_attention_feat')
    parser.add_argument('--attention_arch', type=str, default='shared_attention',
                        choices=['shared_attention', 'multiple_attention'])
    parser.add_argument('--patch_mtl_saveprefix', type=str, default='')

    parser.add_argument('--encoder_type', type=str, default='None')
    parser.add_argument('--encoder_ckpt_path', type=str, default="")

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--val_subset', default='val', type=str)
    parser.add_argument('--outside_test_filenames', default='', type=str)
    parser.add_argument('--bcss_patches_filename', default='', type=str)
    parser.add_argument('--bcss_feats_filename', default='', type=str)
    parser.add_argument('--bcss_ratio', type=float, default=0.5,
                        help='if the area for one label is greater than ratio, consider the patch as the label')
    parser.add_argument('--bcss_min', type=int, default=50,
                        help='if the number of patches is greater than this value, consider the label')
    parser.add_argument('--bcss_crop_size', type=int, default=256, help='crop_size')
    parser.add_argument('--bcss_target_size', type=int, default=256, help='target_size')
    parser.add_argument('--bcss_overlap', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--bcss_version', type=str, default='')
    parser.add_argument('--num_parallel_jobs', default=1, type=int)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug',
                        action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    return parser.parse_args()


# Seed for repeatability
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def prepare_histo_splits(args):
    # csv_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/BigData/TCGA-BRCA/all_with_fpkm_clean_with_clinical_fixed_withTIDE_withMPP_CLAM_IHC.csv'
    save_dir = '{}/splits_{}'.format(args.dataset_root, args.split_label_name)
    os.makedirs(save_dir, exist_ok=True)
    train_filename = os.path.join(save_dir, 'train-%d.csv' % args.split_num)
    val_filename = os.path.join(save_dir, 'val-%d.csv' % args.split_num)
    test_filename = os.path.join(save_dir, 'test-%d.csv' % args.split_num)

    if not os.path.exists(train_filename):

        invalid_CLIDs = [
            # 'TCGA-A8-A09B-01A',
            # 'TCGA-A8-A0A4-01A',
            # 'TCGA-A8-A07U-01A',
            # 'TCGA-A8-A0A2-01A',
            # 'TCGA-A8-A08Z-01A'

            # 'TCGA-E9-A1N6-01A',
            # 'TCGA-D8-A1JC-01A'

        ]
        df = pd.read_csv(args.csv_filename, delimiter=',', index_col=0)

        removed_columns = [col for col in df.columns if '_cnv' in col or '_rnaseq' in col or '_mut' in col]
        df = df.drop(columns=removed_columns)

        if 'DX_filename' not in df.columns:
            dx_filenames = []
            for slideid in df['slide_id'].values:
                dx_filenames.append(os.path.join(args.dataset_root, 'svs', slideid))
            df['DX_filename'] = dx_filenames

        df = df[df.DX_filename.isnull() == False]
        if len(invalid_CLIDs) > 0:
            df = df.drop(invalid_CLIDs)
        df = df[df[df.columns[10]] != 'CLOW']  # PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype
        # df = df.dropna()

        # remove those rows with `X` values 20220213 from Peng
        # df = df[df['ajcc_pathologic_stage'].isin(STAGES_DICT.keys())]
        # df['ajcc_pathologic_stage'] = df['ajcc_pathologic_stage'].map(STAGES_DICT)

        SUBTYPES_DICT = {
            'LumA': 0,
            'LumB': 1,
            'Basal': 2,
            'HER2E': 3,
            'normal-like': 4
        }
        if 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype' in df.columns:
            df = df[df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].isin(SUBTYPES_DICT.keys())]
            df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = \
                df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].map(SUBTYPES_DICT).fillna(4)
        else:
            df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = [0 for _ in range(len(df))]

        if 'HER2.newly.derived' in df.columns:
            # df = df[df['IHC_HER2'].isin(['Positive', 'Negative'])]
            IHC_HER2_dict = {k: 2 for k in df['HER2.newly.derived'].value_counts().index}
            IHC_HER2_dict['Positive'] = 1
            IHC_HER2_dict['Negative'] = 0
            df['IHC_HER2'] = df['HER2.newly.derived'].map(IHC_HER2_dict).fillna(2)
        else:
            df['IHC_HER2'] = [0 for _ in range(len(df))]

        if '2016_Histology_Annotations' in df.columns:
            # try oncotree_code
            HistoAnno_dict = {k: 2 for k in df['2016_Histology_Annotations'].value_counts().index}
            HistoAnno_dict['Invasive ductal carcinoma'] = 1
            HistoAnno_dict['Invasive lobular carcinoma'] = 0
            df['HistoAnno'] = df['2016_Histology_Annotations'].map(HistoAnno_dict).fillna(2)
        else:
            df['HistoAnno'] = [0 for _ in range(len(df))]

        if 'ajcc_pathologic_stage' in df.columns:
            # STAGES_DICT = {
            #     'Stage I': 0,
            #     'Stage IA': 1,
            #     'Stage IB': 2,
            #     'Stage II': 3,
            #     'Stage IIA': 4,
            #     'Stage IIB': 5,
            #     'Stage III': 6,
            #     'Stage IIIA': 7,
            #     'Stage IIIB': 8,
            #     'Stage IIIC': 9,
            #     'Stage IV': 10
            # }
            Stage_dict = {k: 1.0 for k in df['ajcc_pathologic_stage'].value_counts().index}
            Stage_dict['Stage I'] = 0
            Stage_dict['Stage IA'] = 0.1
            Stage_dict['Stage IB'] = 0.2
            Stage_dict['Stage II'] = 0.3
            Stage_dict['Stage IIA'] = 0.4
            Stage_dict['Stage IIB'] = 0.5
            Stage_dict['Stage III'] = 0.6
            Stage_dict['Stage IIIA'] = 0.7
            Stage_dict['Stage IIIB'] = 0.8
            Stage_dict['Stage IIIC'] = 0.9
            df['Stage'] = df['ajcc_pathologic_stage'].map(Stage_dict).fillna(1.0)
        else:
            df['Stage'] = [0.0 for _ in range(len(df))]

        # gene classification tasks
        gene_cls_columns = ['TP53_cls', 'PIK3CA_cls', 'KRAS_cls']
        for col in gene_cls_columns:
            if col not in df.columns:
                df[col] = [0 for _ in range(len(df))]

        # 43 CytoSig regression tasks
        CytoSig_columns = {col: col.replace(' ', '_') for col in df.columns if 'CytoSig_' in col}
        df = df.rename(columns=CytoSig_columns)

        # dropna for some columns
        columns = [
            'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype', 'Stage', 'IHC_HER2', 'HistoAnno',
            'censorship', 'survival_months', 'vital_status',
            'Cytotoxic_T_Lymphocyte', 'TIDE_M2', 'TIDE_CAF', 'TIDE_MDSC', 'TIDE_Dys'
        ]
        for col in columns:
            df = df[df[col].notna()]

        counts = {}
        if 'brca' in args.project_name.lower():
            counts.update({
                'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype': 5,
                'HistoAnno': 3,
                'IHC_HER2': 3
            })
            counts['TP53_cls'] = 3
            counts['PIK3CA_cls'] = 3
        if 'paad' in args.project_name.lower():
            counts['TP53_cls'] = 3
            counts['KRAS_cls'] = 3

        while True:
            targets = df[args.split_label_name]
            train_idx, remaining_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True,
                                                        stratify=targets)
            trainset = df.iloc[train_idx].copy().reset_index()  # 80% as training
            remainingset = df.iloc[remaining_idx].copy().reset_index()

            targets = remainingset[args.split_label_name]
            val_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.5, shuffle=True, stratify=targets)
            valset = remainingset.iloc[val_idx].copy().reset_index()  # 10% as validation
            valset = valset.drop(['index'], axis=1)

            testset = remainingset.iloc[test_idx].copy().reset_index()  # 10% as testing
            testset = testset.drop(['index'], axis=1)

            good = []
            for key, count in counts.items():
                good += [len(trainset[key].value_counts()) == count,
                         len(valset[key].value_counts()) == count,
                         len(testset[key].value_counts()) == count]
            if np.all(good):
                break

        trainset.to_csv(train_filename)
        valset.to_csv(val_filename)
        testset.to_csv(test_filename)
    else:
        trainset = pd.read_csv(train_filename)
        valset = pd.read_csv(val_filename)
        testset = pd.read_csv(test_filename)

    # return trainset, valset, testset
    return train_filename, val_filename, test_filename


def prepare_histo_splits_for_one_dataset(dataset_root, split_label_name, split_num, csv_filename):
    # csv_filename = '/data/Jiang_Lab/Data/Zisha_Zhong/BigData/TCGA-BRCA/all_with_fpkm_clean_with_clinical_fixed_withTIDE_withMPP_CLAM_IHC.csv'
    if split_label_name == 'None':
        save_dir = '{}/splits'.format(dataset_root)
        os.makedirs(save_dir, exist_ok=True)
        train_filename = os.path.join(save_dir, 'train-%d.csv' % split_num)
        val_filename = os.path.join(save_dir, 'val-%d.csv' % split_num)
        test_filename = os.path.join(save_dir, 'test-%d.csv' % split_num)
        return train_filename, val_filename, test_filename

    save_dir = '{}/splits_{}'.format(dataset_root, split_label_name)
    os.makedirs(save_dir, exist_ok=True)
    train_filename = os.path.join(save_dir, 'train-%d.csv' % split_num)
    val_filename = os.path.join(save_dir, 'val-%d.csv' % split_num)
    test_filename = os.path.join(save_dir, 'test-%d.csv' % split_num)

    if not os.path.exists(train_filename):

        invalid_CLIDs = [
            # 'TCGA-A8-A09B-01A',
            # 'TCGA-A8-A0A4-01A',
            # 'TCGA-A8-A07U-01A',
            # 'TCGA-A8-A0A2-01A',
            # 'TCGA-A8-A08Z-01A'

            # 'TCGA-E9-A1N6-01A',
            # 'TCGA-D8-A1JC-01A'

        ]
        df = pd.read_csv(csv_filename, delimiter=',', index_col=0)

        removed_columns = [col for col in df.columns if '_cnv' in col or '_rnaseq' in col]
        df = df.drop(columns=removed_columns)

        if 'DX_filename' not in df.columns:
            dx_filenames = []
            for slideid in df['slide_id'].values:
                dx_filenames.append(os.path.join(dataset_root, 'svs', slideid))
            df['DX_filename'] = dx_filenames

        df = df[df.DX_filename.isnull() == False]
        if len(invalid_CLIDs) > 0:
            df = df.drop(invalid_CLIDs)
        df = df[df[df.columns[10]] != 'CLOW']  # PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype
        # df = df.dropna()

        # remove those rows with `X` values 20220213 from Peng
        # df = df[df['ajcc_pathologic_stage'].isin(STAGES_DICT.keys())]
        # df['ajcc_pathologic_stage'] = df['ajcc_pathologic_stage'].map(STAGES_DICT)

        SUBTYPES_DICT = {
            'LumA': 0,
            'LumB': 1,
            'Basal': 2,
            'HER2E': 3,
            'normal-like': 4
        }
        if 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype' in df.columns:
            df = df[df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].isin(SUBTYPES_DICT.keys())]
            df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = \
                df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].map(SUBTYPES_DICT).fillna(4)
        else:
            df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'] = [0 for _ in range(len(df))]

        if 'HER2.newly.derived' in df.columns:
            # df = df[df['IHC_HER2'].isin(['Positive', 'Negative'])]
            IHC_HER2_dict = {k: 2 for k in df['HER2.newly.derived'].value_counts().index}
            IHC_HER2_dict['Positive'] = 1
            IHC_HER2_dict['Negative'] = 0
            df['IHC_HER2'] = df['HER2.newly.derived'].map(IHC_HER2_dict).fillna(2)
        else:
            df['IHC_HER2'] = [0 for _ in range(len(df))]

        if '2016_Histology_Annotations' in df.columns:
            # try oncotree_code
            HistoAnno_dict = {k: 2 for k in df['2016_Histology_Annotations'].value_counts().index}
            HistoAnno_dict['Invasive ductal carcinoma'] = 1
            HistoAnno_dict['Invasive lobular carcinoma'] = 0
            df['HistoAnno'] = df['2016_Histology_Annotations'].map(HistoAnno_dict).fillna(2)
        else:
            df['HistoAnno'] = [0 for _ in range(len(df))]

        if 'ajcc_pathologic_stage' in df.columns:
            # STAGES_DICT = {
            #     'Stage I': 0,
            #     'Stage IA': 1,
            #     'Stage IB': 2,
            #     'Stage II': 3,
            #     'Stage IIA': 4,
            #     'Stage IIB': 5,
            #     'Stage III': 6,
            #     'Stage IIIA': 7,
            #     'Stage IIIB': 8,
            #     'Stage IIIC': 9,
            #     'Stage IV': 10
            # }
            Stage_dict = {k: 1.0 for k in df['ajcc_pathologic_stage'].value_counts().index}
            Stage_dict['Stage I'] = 0
            Stage_dict['Stage IA'] = 0.1
            Stage_dict['Stage IB'] = 0.2
            Stage_dict['Stage II'] = 0.3
            Stage_dict['Stage IIA'] = 0.4
            Stage_dict['Stage IIB'] = 0.5
            Stage_dict['Stage III'] = 0.6
            Stage_dict['Stage IIIA'] = 0.7
            Stage_dict['Stage IIIB'] = 0.8
            Stage_dict['Stage IIIC'] = 0.9
            df['Stage'] = df['ajcc_pathologic_stage'].map(Stage_dict).fillna(1.0)
        else:
            df['Stage'] = [0.0 for _ in range(len(df))]

        # dropna for some columns
        columns = [
            'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype', 'Stage', 'IHC_HER2', 'HistoAnno',
            'censorship', 'survival_months', 'vital_status',
            'Cytotoxic_T_Lymphocyte', 'M2', 'CAF', 'MDSC', 'Dys'
        ]
        for col in columns:
            df = df[df[col].notna()]

        targets = df[split_label_name]
        train_idx, remaining_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True,
                                                    stratify=targets)
        trainset = df.iloc[train_idx].copy().reset_index()  # 80% as training

        remainingset = df.iloc[remaining_idx].copy().reset_index()

        targets = remainingset[split_label_name]
        val_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.5, shuffle=True, stratify=targets)
        valset = remainingset.iloc[val_idx].copy().reset_index()  # 10% as validation
        valset = valset.drop(['index'], axis=1)

        testset = remainingset.iloc[test_idx].copy().reset_index()  # 10% as testing
        testset = testset.drop(['index'], axis=1)

        trainset.to_csv(train_filename)
        valset.to_csv(val_filename)
        testset.to_csv(test_filename)
    else:
        trainset = pd.read_csv(train_filename)
        valset = pd.read_csv(val_filename)
        testset = pd.read_csv(test_filename)

    # return trainset, valset, testset
    return train_filename, val_filename, test_filename


# def get_histo_datasets(args):
#     mask_root = '%s/masks/im%d' % (args.dataset_root, args.image_size)
#     print('\nSetup Data Loader ...')
#     if args.norm_type == 'mean_std':
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#     elif args.norm_type == 'zero_one':
#         mean = [0, 0, 0]
#         std = [1, 1, 1]
#     else:
#         raise ValueError('wrong norm type')
#     if args.num_channels == 1:
#         data_transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             # transforms.CenterCrop((im_size, im_size)),
#             # transforms.RandomCrop((im_size, im_size)),
#             # transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#         ])
#     else:
#         data_transform = transforms.Compose([
#             # transforms.CenterCrop((im_size, im_size)),
#             # transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#         ])
#
#     trainset = pd.read_csv(args.trainset_filename)
#     valset = pd.read_csv(args.valset_filename)
#
#     train_dataset = HistoDataset(df=trainset, mask_root=mask_root, transform=data_transform,
#                                  num_patches=args.num_patches, debug=args.debug, prefix='train')
#     val_dataset = HistoDataset(df=valset, mask_root=mask_root, transform=data_transform,
#                                num_patches=args.num_patches, debug=args.debug, prefix='val')
#
#     return train_dataset, val_dataset


class Regression_Logger(object):
    def __init__(self):
        super(Regression_Logger, self).__init__()

        self.Y_hat = []
        self.Y = []

    def log(self, Y_hat, Y):
        if isinstance(Y_hat, torch.Tensor):
            Y_hat_list = Y_hat.detach().cpu().numpy().tolist()
            if isinstance(Y, torch.Tensor):
                Y_list = Y.detach().cpu().numpy().tolist()
            elif isinstance(Y, np.ndarray):
                Y_list = Y.tolist()
            if not isinstance(Y_hat_list, list):
                Y_hat_list = [Y_hat_list]
            if not isinstance(Y_list, list):
                Y_list = [Y_list]
            self.Y_hat += Y_hat_list
            self.Y += Y_list
        else:
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.Y_hat.append(Y_hat)
            self.Y.append(Y)

    def mean_squared_error(self):
        if len(self.Y) == 0:
            return 0.
        else:
            Y_hat = np.array(self.Y_hat)
            Y = np.array(self.Y)
            return ((Y_hat - Y) ** 2).mean(axis=None)

    def compute_metrics(self):
        if len(self.Y) == 0:
            return {'r2score': 0.0,
                    'pearsonr_corr': 0.0,
                    'pearsonr_pvalue': 1.0,
                    'spearmanr_corr': 0.0,
                    'spearmanr_pvalue': 1.0}
        else:
            Y = np.array(self.Y).reshape(-1)
            Y_hat = np.array(self.Y_hat).reshape(-1)
            pearson_corr, pearsonr_pvalue = pearsonr(Y, Y_hat)
            spearmanr_corr, spearmanr_pvalue = spearmanr(Y, Y_hat)
            return {'r2score': r2_score(Y, Y_hat),
                    'pearsonr_corr': pearson_corr,
                    'pearsonr_pvalue': pearsonr_pvalue,
                    'spearmanr_corr': spearmanr_corr,
                    'spearmanr_pvalue': spearmanr_pvalue
                    }


def generate_roc_curve(Y, Y_prob, task_name=None, label_names=None, save_filename=None):
    # check epoch 0, using original data
    # skip null value
    # Y:[0, 1, 2] 3
    # Y_pred_prob:
    # [[0.6, 0.3, 0.1],
    # [0.5, 0.4, 0.1],
    # [0.4, 0.2, 0.4]]
    # Y->Y_true_prob: [] N x 3
    # [[1, 0, 0],
    #  [0, 1, 0],
    #  [0, 0, 1]
    # Y.ravale() --> Nx3 --> N1 x 1: [1, 0, 0, 0, 1, 0]
    # micro: Y_true: [1, 0, 0, 0, 1, 0, 0, 0, 1] 9 x 1, Y_pred_prob: [0.6, 0.3, 0.1, ...]

    # resnet18 from ImageNet
    # resnet18 from patch-based MTL
    # resnet18 form DINO (no label, self-supervised)
    # resnet18 form VAE

    # fine-tuning on decoder (multi-task)
    # svs 10000 x (256 x 256)
    # 600 svs, 128 patches -->
    # 1.svs, 128 patches, 32 parts, --> 4 patches / part
    # 2.svs, 256 pathces, 32parts, -->

    Y0 = np.copy(Y)
    Y = label_binarize(Y, classes=[i for i in range(len(label_names))])
    n_classes = Y.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label_counts = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        label_counts.append(len(np.where(Y0 == i)[0]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            # color=color,
            lw=lw,
            label="ROC {0}:{1}({2}) ({3:0.2f})".format(i, label_names[i], label_counts[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task_name)
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_filename)
    plt.close()


def generate_roc_curve_fixed(Y, Y_prob, task_name='HistoAnno',
                             label_names=['lob', 'duct', 'other'],
                             save_filename=None):
    # check epoch 0, using original data
    # skip null value
    # Y:[0, 1, 2] 3
    # Y_pred_prob:
    # [[0.6, 0.3, 0.1],
    # [0.5, 0.4, 0.1],
    # [0.4, 0.2, 0.4]]
    # Y->Y_true_prob: [] N x 3
    # [[1, 0, 0],
    #  [0, 1, 0],
    #  [0, 0, 1]
    # Y.raval() --> Nx3 --> N1 x 1: [1, 0, 0, 0, 1, 0]
    # micro: Y_true: [1, 0, 0, 0, 1, 0, 0, 0, 1] 9 x 1, Y_pred_prob: [0.6, 0.3, 0.1, ...]

    # resnet18 from ImageNet
    # resnet18 from patch-based MTL
    # resnet18 form DINO (no label, self-supervised)
    # resnet18 form VAE

    # fine-tuning on decoder (multi-task)
    # svs 10000 x (256 x 256)
    # 600 svs, 128 patches -->
    # 1.svs, 128 patches, 32 parts, --> 4 patches / part
    # 2.svs, 256 pathces, 32parts, -->

    Y0 = np.copy(Y)
    # Y = label_binarize(Y, classes=[i for i in range(len(label_names))])
    # n_classes = Y.shape[1]

    Y = np.eye(len(label_names))[Y]
    n_classes = Y.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label_counts = []
    invalid_i = []

    thresholds = dict()
    optim_threshold_indices = []
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(Y[:, i], Y_prob[:, i])
        # tpr: TP / (TP + FN) = N / (N + 0)
        # fpr: FP / (FP + TN) = 0 / (0 + 0)
        # print(i, tpr[i], fpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        count = len(np.where(Y0 == i)[0])
        label_counts.append(count)
        if count == 0:
            invalid_i.append(i)
            optim_threshold_indices.append(-1)
        else:
            temp = tpr[i] - fpr[i]
            maxindex = temp.tolist().index(max(temp))
            optim_threshold_indices.append(maxindex)
    # print('debug done\n\n')
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i not in invalid_i]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in invalid_i:
            continue
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= (n_classes - len(invalid_i))

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "olive", "fuchsia", "seagreen", "indigo",
                    "deepskyblue"])
    # color_indices = np.array([87, 73, 126, 92, 125, 96, 24, 83, 112, 127, 29, 64, 46,
    #        78, 147, 49, 6, 19, 74, 9, 93, 14, 102, 39, 34, 146,
    #        51, 41, 60, 25, 142, 100, 42, 117, 82, 137, 133, 97, 15,
    #        7, 85, 99, 45, 114, 30, 38, 58, 59, 84, 120, 77, 22,
    #        80, 53, 128, 36, 20, 57, 3, 23, 10, 31, 48, 66, 12,
    #        143, 140, 139, 4, 5, 121, 55, 134, 50, 91, 131, 109, 65,
    #        32, 124, 26, 119, 0, 122, 141, 11, 98, 8, 68, 129, 106,
    #        56, 69, 37, 116, 75, 28, 138, 2, 54, 35, 111, 110, 123,
    #        105, 104, 16, 135, 61, 107, 18, 81, 130, 88, 94, 1, 89,
    #        101, 62, 63, 21, 71, 43, 118, 52, 90, 95, 144, 79, 44,
    #        27, 113, 72, 40, 17, 76, 33, 13, 86, 136, 132, 103, 67,
    #        145, 115, 47, 70, 108])
    # colors = np.array(list(mcolors.CSS4_COLORS))
    # colors = colors[color_indices].tolist()
    # colors = cycle([colors[i] for i in range(n_classes)])
    for i, color in zip(range(n_classes), colors):
        if i in invalid_i:
            continue
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC {0}:{1}({2}) ({3:0.2f})".format(i, label_names[i], label_counts[i], roc_auc[i]),
        )

        maxindex = optim_threshold_indices[i]
        optim_threshold = thresholds[i][maxindex]
        plt.text(fpr[i][maxindex],
                 tpr[i][maxindex],
                 '{:.3f}'.format(optim_threshold),
                 color=color,
                 fontsize='large')

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task_name)
    plt.legend(loc="lower right")
    plt.grid()
    if save_filename is not None:
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()


def softmax_stable(x):  # only 2-D
    x = np.exp(x - np.max(x, axis=1)[:, None])
    return x / x.sum(axis=1)[:, None]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, save_dir=None):

        score = val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_dir=None):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation score increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if save_dir is not None:
            if not isinstance(save_dir, list):
                save_dir = [save_dir]
            for i in range(len(save_dir)):
                torch.save(model.state_dict(), os.path.join(save_dir[i], 'best_checkpoint.pth'))
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), ckpt_name)


def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


def survival_AUC(df1, df2, times=None, method='hc_sksurv'):
    df1['censorship'] = 1 - df1['censorship']
    df2['censorship'] = 1 - df2['censorship']
    times = np.percentile(df2['survival'], np.linspace(20, 81, 15))

    surv1 = np.array(df1, dtype=int)
    risk2 = np.array(df2['risk'])
    surv2 = np.array(df2.drop(['risk'], axis=1), dtype=int)

    surv1 = np.core.records.fromarrays(surv1[:, [1, 0]].transpose(), names='obs, survival_months', formats='?, i8')
    surv2 = np.core.records.fromarrays(surv2[:, [1, 0]].transpose(), names='obs, survival_months', formats='?, i8')
    _, iauc = cumulative_dynamic_auc(surv1, surv2, risk2, times)
    return iauc


def draw_survival_curves(survival_df, save_filename=None):
    risk_percentiles = [50]
    p = np.percentile(survival_df['risk'], risk_percentiles)
    survival_df.insert(len(survival_df.columns), 'strat', [hazard2grade(risk, p) for risk in survival_df['risk']])
    T_low, T_high = survival_df['survival'][survival_df['strat'] == 0], survival_df['survival'][
        survival_df['strat'] == 1]
    E_low, E_high = 1 - survival_df['censorship'][survival_df['strat'] == 0], 1 - survival_df['censorship'][
        survival_df['strat'] == 1]

    low_vs_high_p_value = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low,
                                       event_observed_B=E_high).p_value

    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)

    kmf.fit(T_low, event_observed=E_low, label="low risk ({})".format(len(T_low)))
    kmf.plot_survival_function(ax=ax)

    kmf.fit(T_high, event_observed=E_high, label="high risk ({})".format(len(T_high)))
    kmf.plot_survival_function(ax=ax)

    plt.title("Kaplan-Meier with logrank pvalue = {:0.4f}".format(low_vs_high_p_value))
    plt.savefig(save_filename, dpi=600)
    plt.close()


def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def _assertLevelDownsamples(slide):
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]

    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples


def block_blending(slide, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
    print('\ncomputing blend')
    level_downsamples = _assertLevelDownsamples(slide)
    downsample = level_downsamples[vis_level]
    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)
    print('using block size: {} x {}'.format(block_size_x, block_size_y))

    shift = top_left  # amount shifted w.r.t. (0,0)
    for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
            # print(x_start, y_start)

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]) / int(downsample[0]))
            y_start_img = int((y_start - shift[1]) / int(downsample[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue
            # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            if not blank_canvas:
                # 4. read actual wsi block as canvas block
                pt = (x_start, y_start)
                canvas = np.array(slide.read_region(pt, vis_level, blend_block_size).convert("RGB"))
            else:
                # 4. OR create blank canvas block
                canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

            # 5. blend color block and canvas block
            img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                1 - alpha, 0, canvas)
    return img


"""
  # downsample at which to visualize heatmap (-1 refers to downsample closest to 32x downsample)
  vis_level: 1
  # transparency for overlaying heatmap on background (0: background only, 1: foreground only)
  alpha: 0.4
  # whether to use a blank canvas instead of original slide
  blank_canvas: false
  # whether to also save the original H&E image
  save_orig: true
  # file extension for saving heatmap/original image
  save_ext: jpg
  # whether to calculate percentile scores in reference to the set of non-overlapping patches
  use_ref_scores: true
  # whether to use gaussian blur for further smoothing
  blur: false
  # whether to shift the 4 default corner points for checking if a patch is inside a foreground contour
  use_center_shift: true
  # whether to only compute heatmap for ROI specified by x1, x2, y1, y2
  use_roi: false 
  # whether to calculate heatmap with specified overlap (by default, coarse heatmap without overlap is always calculated)
  calc_heatmap: true
  # whether to binarize attention scores
  binarize: false
  # binarization threshold: (0, 1)
  binary_thresh: -1
  # factor for downscaling the heatmap before final dispaly
  custom_downsample: 1
  cmap: jet
"""


def visHeatmap(wsi, scores, coords, vis_level=1,
               top_left=None, bot_right=None,
               patch_size=(256, 256),
               blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
               blur=False, overlap=0.0,
               segment=True, use_holes=True,
               convert_to_percentiles=False,
               binarize=False, thresh=0.5,
               max_size=None,
               custom_downsample=1,
               cmap='jet'):
    """
    Args:
        scores (numpy array of float): Attention scores
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                        self.contours_tissue and self.holes_tissue are not None
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        threshold (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
    """

    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    # downsample = self.level_downsamples[vis_level]
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level

    if len(scores.shape) == 2:
        scores = scores.flatten()

    if binarize:
        if thresh < 0:
            threshold = 1.0 / len(scores)

        else:
            threshold = thresh

    else:
        threshold = 0.0

    ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)

    else:
        # region_size = self.level_dim[vis_level]
        region_size = wsi.level_dimensions[vis_level]
        top_left = (0, 0)
        # bot_right = self.level_dim[0]
        bot_right = wsi.level_dimensions[0]
        w, h = region_size

    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    print('\ncreating heatmap for: ')
    print('top_left: ', top_left, 'bot_right: ', bot_right)
    print('w: {}, h: {}'.format(w, h))
    print('scaled patch size: ', patch_size)

    ###### normalize filtered scores ######
    if convert_to_percentiles:
        scores = to_percentiles(scores)

    scores /= 100

    ######## calculate the heatmap of raw attention scores (before colormap)
    # by accumulating scores over overlapped regions ######

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    overlay = np.full(np.flip(region_size), 0).astype(float)
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)
    count = 0
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0
        # accumulate attention
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

    if binarize:
        print('\nbinarized tiles based on cutoff of {}'.format(threshold))
        print('identified {}/{} patches as positive'.format(count, len(coords)))

    # fetch attended region and average accumulated attention
    zero_mask = counter == 0

    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter
    if blur:
        overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    # if segment:
    #     tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
    #     # return Image.fromarray(tissue_mask) # tissue mask

    if not blank_canvas:
        # downsample original image and use as canvas
        img = np.array(wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
    else:
        # use blank canvas
        img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # return Image.fromarray(img) #raw image

    print('\ncomputing heatmap image')
    print('total of {} patches'.format(len(coords)))
    twenty_percent_chunk = max(1, int(len(coords) * 0.2))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for idx in range(len(coords)):
        if (idx + 1) % twenty_percent_chunk == 0:
            print('progress: {}/{}'.format(idx, len(coords)))

        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            # attention block
            raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

            # # image block (either blank canvas or orig image)
            # img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()
            #
            # # color block (cmap applied to attention block)
            # color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)
            #
            # if segment:
            #     # tissue mask block
            #     mask_block = tissue_mask[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
            #     # copy over only tissue masked portion of color block
            #     img_block[mask_block] = color_block[mask_block]
            # else:
            #     # copy over entire color block
            #     img_block = color_block

            img_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

            # rewrite image block
            img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

    # return Image.fromarray(img) #overlay
    print('Done')
    del overlay

    if blur:
        img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if alpha < 1.0:
        img = block_blending(wsi, img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas,
                             block_size=1024)

    img = Image.fromarray(img)
    w, h = img.size

    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img


def visWSI(wsi, vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
           line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
           view_slide_only=False,
           number_contours=False, seg_display=True, annot_display=True):
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]

    if top_left is not None and bot_right is not None:
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
    else:
        top_left = (0, 0)
        region_size = wsi.level_dimensions[vis_level]

    img = wsi.read_region(top_left, vis_level, region_size).convert("RGB")
    w, h = img.size
    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img


def get_coords_by_level(wsi, coords, level):
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level
    coords_level = np.ceil(coords * np.array(scale)).astype(int)
    return coords_level


def get_coords(patch_filename):
    data_dict = {}
    with h5py.File(patch_filename, 'r') as h5file:  # the mask_root is the CLAM patches dir
        data_dict['locations'] = h5file['coords'][()]
        data_dict['patch_level'] = h5file['coords'].attrs['patch_level']
        data_dict['patch_size'] = h5file['coords'].attrs['patch_size']
    return data_dict


def new_web_annotation(cluster_label, min_dist, x, y, w, h, annoid_str):
    anno = {
        "type": "Annotation",
        "body": [{
            "type": "TextualBody",
            "value": "{:d},({:.3f})".format(cluster_label, min_dist),
            "purpose": "tagging"
        }],
        "target": {
            "source": "http://localhost:3000/",
            "selector": {
                "type": "FragmentSelector",
                "conformsTo": "http://www.w3.org/TR/media-frags/",
                "value": f"xywh=pixel:{x},{y},{w},{h}"
            }
        },
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": annoid_str
    }
    return anno
