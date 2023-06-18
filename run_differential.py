import sys, os, shutil, glob, json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from histo_utils import score2percentile, _assertLevelDownsamples, \
    visHeatmap, visWSI, new_web_annotation, get_coords, clean_csv, get_args_v2
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import openslide
import cv2
from PIL import Image, ImageDraw
import time
from kde import KDE
from scipy.spatial import distance_matrix
from skimage.feature import blob_dog, blob_log, blob_doh
import math
import pyvips
from sklearn import cluster, mixture
from torchvision import transforms
from torchvision.utils import save_image
from PIL import ImageFont
import webcolors
import io, tarfile


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        dropout = 0.25
        size = [1280, 256, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attention_only=False):

        x_path = x.squeeze(0)
        A, h = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return {'A_raw': A}

        A = F.softmax(A, dim=1)
        h = torch.mm(A, h)
        h = self.rho(h)
        return h


args = get_args_v2()
args.cluster_task_name = 'response_groups'

debug_postfix = '_debug' if args.debug else ''
save_root = '{}/{}_{}{}/' \
            'split{}_e{}_h{}_p{}_density_vis/' \
            '{}_{}/{}_top_{}_clustering/{}/{}{}'.format(
    args.save_root,
    args.attention_arch,
    args.encoder_type,
    args.backbone,
    args.split_num,
    args.max_epochs - 1,
    args.image_size,
    args.num_patches,
    args.cluster_strategy,
    args.num_clusters,
    args.cluster_task_name,
    args.num_patches,
    args.cluster_feat_name,
    args.cluster_subset,
    debug_postfix)

os.makedirs(save_root, exist_ok=True)
with open(os.path.join(save_root, 'args.txt'), 'w') as fp:
    json.dump(args.__dict__, fp, indent=2)

# load colormap consistent with Frontend
cmap = []
colornames = []
with open('web/mycolors.js', 'r') as fp:
    lines = fp.readlines()
    for ci in range(1, len(lines) - 1):
        if lines[ci][0] == ']':
            break
        colorname = lines[ci].strip().replace(',', '').replace('"', '')
        colornames.append(colorname)
        cmap.append(list(webcolors.name_to_rgb(colorname)))
cmap = np.array(cmap, dtype=np.uint8)

labels_dict = {
    'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype':
        {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'HER2E', 4: 'normal'},
    'CLS_HistoAnno':
        {0: 'Lobular', 1: 'Ductal', 2: 'Other'},
    'CLS_IHC_HER2':
        {0: 'Negative', 1: 'Positive', 2: 'Other'},
    # for PAAD and Sheba
    'survival_groups':
        {0: 'LowRisk', 1: 'HighRisk'},
    'Stage':
        {0: 'StageI_II', 1: 'StageM'},
    # for mouse
    'treatment_group':
        {1: '1 (control Ab)', 2: '2 (Anti-TGFb)'},
    'met_weight_label':
        {0: 'group0', 1: 'group1', 2: 'group2'},
    # for gene mutation
    'TP53_cls':
        {0: 'Gain_Or_Unknown_Or_NaN', 1: 'Loss', 2: 'other'},
    # for Adoptive_TIL_Breast, WiemData (Deprecated)
    'ResponseGroup':
        {0: 'PD', 1: 'SD', 2: 'PR', 3: 'CR'},  # here should be two-group, TODO: discussion with Dr. Jiang
    # for METABRIC
    'molecular_subtype_groups':
        {0: 'LumA', 1: 'LumB', 2: 'Basal', 3: 'Her2', 4: 'Normal'},
    # for TIL, WiemData
    'response_groups':
        {0: 'BadResponse', 1: 'GoodResponse'},
    'group_label':
        {0: 'group0', 1: 'group1'}
}

labels_dict_reverse = {k: {v: k for k, v in labels_dict[k].items()} for k in labels_dict.keys()}

model = AttentionModel()
state_dict = torch.load(args.ckpt_path, map_location='cpu')  # encoder + attention except final
model.load_state_dict(state_dict['MODEL_STATE'], strict=False)

if torch.cuda.device_count() > 0:
    model.cuda()
model.eval()

feature_tensors = {}


def get_activation(name):
    def hook(model, input, output):
        feature_tensors[name + '_feat'] = output.detach()

    return hook


feature_names = [
    'feat_after_encoder',  # 10000x512
    'feat_before_attention',  # 10000x256
    'feat_after_attention_0_linear',
    'feat_after_attention_1_relu',
    'feat_after_attention_2_dropout'
]
model.attention_net[0].register_forward_hook(get_activation('feat_before_attention')),
model.rho[0].register_forward_hook(get_activation('feat_after_attention_0_linear')),
model.rho[1].register_forward_hook(get_activation('feat_after_attention_1_relu')),
model.rho[2].register_forward_hook(get_activation('feat_after_attention_2_dropout')),
# model.classifiers['HistoAnno'].register_forward_hook(get_activation('outputs_HistoAnno')),
# model.classifiers['IHC_HER2'].register_forward_hook(get_activation('outputs_IHC_HER2')),
# model.classifiers['subtype'].register_forward_hook(get_activation('outputs_subtype'))

print('feature_tensors keys:', feature_tensors.keys())

csv_filename = args.csv_filename

if 'xlsx' in csv_filename:
    df = pd.read_excel(csv_filename)
else:
    df = pd.read_csv(csv_filename)

df = clean_csv(df, mask_root=None, model_name=None,
               classification_dict=None, regression_list=None, ignore_index_dict=None,
               only_step2=True)

svs_filenames = df['HEfiles'] if 'HEfiles' in df else df['DX_filename'].values
label_title = 'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype' \
    if args.cluster_task_name == 'CLS_Molecular_Subtype' else args.cluster_task_name
labels = df[label_title].values.astype(np.int32)

X = []
Y = []
all_patches = []
all_positions = []  # svs_ind, x, y, label
for ind in np.arange(len(svs_filenames)):
    label = labels[ind]
    label_text = labels_dict[label_title][label]
    As = []
    jjs = []
    svs_filename_list = svs_filenames[ind].split(',')  # 2 slides
    print(ind, len(svs_filename_list))
    for j, filename in enumerate(svs_filename_list):
        svs_filename = os.path.join(args.dataset_root, 'svs', filename.replace(' ', '').replace('&', ''))
        slide = openslide.OpenSlide(svs_filename)
        svs_prefix = os.path.basename(svs_filename).replace(args.image_ext, '')
        patch_info_dict = get_coords(
            os.path.join('{}/patches/'.format(args.dataset_root), svs_prefix + '.h5'))
        patch_level = patch_info_dict['patch_level']
        dimension = slide.level_dimensions[patch_level]
        patch_size = patch_info_dict['patch_size']

        coords = patch_info_dict['locations']

        print(svs_prefix)
        feats = torch.load(os.path.join(args.feats_dir, svs_prefix + '.pt'))
        if torch.cuda.device_count() > 0:
            feats = feats.cuda()
        results_dict = model(feats, attention_only=True)
        A_raw = results_dict['A_raw'].detach().cpu().numpy()

        A = A_raw[args.cluster_task_index]
        print(A.shape)
        As.append(A)
        jjs.append(j * np.ones_like(A, dtype=np.int32))

    ccs = [len(A) for A in As]
    cc = [0] + np.cumsum(ccs).tolist()
    A = np.concatenate(As)
    jj = np.concatenate(jjs)
    assert len(A) == len(jj), 'wrong, check it'
    ref_scores = np.copy(A)
    for ind1 in range(len(A)):
        A[ind1] = score2percentile(A[ind1], ref_scores)

    sort_ind = np.argsort(A)
    num_ps = args.num_patches if len(sort_ind) > args.num_patches else len(sort_ind)
    if True:
        selected_indices = sort_ind[-int(num_ps):]
    else:
        selected_indices = sort_ind
    jj_selected = jj[selected_indices]

    for j, filename in enumerate(svs_filename_list):
        svs_filename = os.path.join(args.dataset_root, 'svs', filename.replace(' ', '').replace('&', ''))
        slide = openslide.OpenSlide(svs_filename)
        svs_prefix = os.path.basename(svs_filename).replace(args.image_ext, '')
        patch_info_dict = get_coords(
            os.path.join('{}/patches/'.format(args.dataset_root), svs_prefix + '.h5'))
        patch_level = patch_info_dict['patch_level']
        dimension = slide.level_dimensions[patch_level]
        patch_size = patch_info_dict['patch_size']

        coords = patch_info_dict['locations']

        selected_indices_j = selected_indices[jj_selected == j] - cc[j]

        print(svs_prefix)
        feats = torch.load(os.path.join(args.feats_dir, svs_prefix + '.pt'))
        if torch.cuda.device_count() > 0:
            feats = feats.cuda()
        if 'after_encoder' in args.cluster_feat_name:
            x = feats.cpu().numpy()[selected_indices_j]
        else:
            results_dict = model(feats, attention_only=True)
            x = feature_tensors[args.cluster_feat_name].cpu().numpy()[selected_indices_j]
        X += x.tolist()  # 16 x 256
        Y.append(label * np.ones((x.shape[0],), dtype=np.int32))

        # extract the image patches
        for ind1, coord in enumerate(coords[selected_indices_j]):
            patch = slide.read_region(location=(int(coord[0]), int(coord[1])), level=patch_level,
                                      size=(patch_size, patch_size)).convert('RGB')
            all_patches.append(patch)
            all_positions.append((svs_prefix, int(coord[0]), int(coord[1]), label))

print('collect data for UMAP')
# X = np.concatenate(X, axis=0)  # (num_cases x num_patches) * feat_dim
X = np.array(X)
Y = np.concatenate(Y, axis=0)
print(X.shape)
print(Y.shape)

metric = 'euclidean'
# norm_type = 'standard'
# if norm_type == 'standard':
#     X_scaled = StandardScaler().fit_transform(X)
# elif norm_type == 'minmax':
#     X_scaled = MinMaxScaler().fit_transform(X)
# else:
#     X_scaled = np.copy(X)
X_scaled = X

if args.cluster_strategy == 'density_log_2d':
    print('do UMAP')
    reducer = umap.UMAP(random_state=42, metric=metric)
    feats_embedding = reducer.fit_transform(X_scaled)

    feats_embedding_hist0, feats_embedding_bin_edges0 = np.histogram(feats_embedding[:, 0], bins=100)
    feats_embedding_hist1, feats_embedding_bin_edges1 = np.histogram(feats_embedding[:, 1], bins=100)
    feats_embedding_inds0 = np.digitize(feats_embedding[:, 0], feats_embedding_bin_edges0)
    feats_embedding_inds1 = np.digitize(feats_embedding[:, 1], feats_embedding_bin_edges1)

    feats_embedding1_orig = np.stack([feats_embedding_inds0, feats_embedding_inds1]).T
    feats_embedding1 = [tuple(row) for row in feats_embedding1_orig]
    feats_embedding1 = np.unique(feats_embedding1, axis=0)

    fig = plt.figure()
    kdeplot = sns.kdeplot(x=feats_embedding[:, 0],
                          y=feats_embedding[:, 1],
                          fill=True)
    plt.savefig(
        '{}/{}_{}_{}_umap2d_kdeplot_no_category.png'.format(save_root,
                                                            args.cluster_task_name,
                                                            args.cluster_subset,
                                                            args.cluster_feat_name),
        bbox_inches='tight', dpi=600)
    plt.close()

    fig = plt.figure()
    kdeplot = sns.kdeplot(x=feats_embedding1[:, 0],
                          y=feats_embedding1[:, 1],
                          fill=True)
    plt.savefig('{}/{}_{}_{}_umap2d_kdeplot_no_category_quantized.png'.format(save_root,
                                                                              args.cluster_task_name,
                                                                              args.cluster_subset,
                                                                              args.cluster_feat_name),
                bbox_inches='tight', dpi=600)
    plt.close()

    fig, axes = plt.subplots(1, len(labels_dict[label_title].keys()), figsize=(15, 5))
    fig.suptitle('{}_{}'.format(args.cluster_task_name, args.cluster_feat_name))
    for label_index, label in enumerate(labels_dict[label_title].keys()):
        label_name = labels_dict[label_title][label]
        ind = np.where(Y == label)[0]
        sns.kdeplot(ax=axes[label_index], x=feats_embedding[ind, 0], y=feats_embedding[ind, 1],
                    fill=True)
        axes[label_index].set_title(label_name)
    plt.savefig(
        '{}/{}_{}_{}_umap2d_kdeplot.png'.format(save_root,
                                                args.cluster_task_name,
                                                args.cluster_subset,
                                                args.cluster_feat_name),
        bbox_inches='tight', dpi=600)
    plt.close(fig)

    print('do clustering')
    bw_method = 'scott'
    bw_adjust = 1
    gridsize = 100
    cut = 0
    clip = None
    legend = True
    cumulative = False
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )
    estimator = KDE(**estimate_kws)
    observations = (feats_embedding1[:, 0], feats_embedding1[:, 1])
    weights = None
    density, support = estimator(*observations, weights=weights)

    density_new = 255 * ((density - density.min()) / (density.max() - density.min()))

    image_gray = density_new.astype(np.uint8)
    image = np.repeat(image_gray[:, :, None], repeats=3, axis=2)

    blobs_log = blob_log(image_gray, min_sigma=1, max_sigma=5, num_sigma=20, threshold=.01)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)

    blobs_dog = blob_dog(image_gray, min_sigma=1, max_sigma=5, threshold=.01)
    blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

    blobs_doh = blob_doh(image_gray, min_sigma=1, max_sigma=5, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    _, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.savefig('{}/kde_local_peak.png'.format(save_root))
    plt.close()

    final_centroids = blobs_log[:, [1, 0]]  # using LoG
    # distances = distance_matrix(feats_embedding1_orig, final_centroids)
    # nn_labels = np.argmin(distances, axis=1)

    rs = blobs_log[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    kdeplot = sns.kdeplot(x=feats_embedding1[:, 0],
                          y=feats_embedding1[:, 1],
                          fill=True)
    for c_ind, center in enumerate(final_centroids):
        circle = plt.Circle(center, radius=rs[c_ind], color=colornames[c_ind], fill=False)
        ax.add_patch(circle)
        ax.text(center[0], center[1],
                s='c{}({:.1f},{:.1f})'.format(c_ind, center[0], center[1]),
                c=colornames[c_ind])
    plt.savefig(
        '{}/{}_{}_{}_umap2d_kdeplot_no_category_with_centers.png'.format(save_root,
                                                                         args.cluster_task_name,
                                                                         args.cluster_subset,
                                                                         args.cluster_feat_name),
        bbox_inches='tight', dpi=600)
    plt.close()

    print('do remap')
    # density1 = np.copy(density_new)
    src = np.copy(density_new).astype(np.uint8)
    src = np.repeat(src[:, :, None], repeats=3, axis=2)
    scale = 4
    src = cv2.resize(src, dsize=None, fx=float(scale), fy=float(scale))
    padleft, padtop = 128, 128
    src = np.pad(src, ((padleft, padleft), (padtop, padtop), (0, 0)))
    lines = []
    for c_ind, center in enumerate(final_centroids):
        center_scale = center * scale
        # cv2.circle(src, (int(center_scale[0]), int(center_scale[1])), radius=3, color=(255, 0, 255), thickness=2)
        color = cmap[c_ind].tolist()
        cv2.circle(src, (int(center_scale[0] + padleft), int(center_scale[1] + padtop)), radius=1, color=color,
                   thickness=2)
        cv2.putText(src, text='c{}({:.1f},{:.1f})'.format(c_ind, center[0], center[1]),
                    org=(int(center_scale[0] + padleft), int(center_scale[1] + padtop)),
                    fontFace=1, fontScale=1, color=color, thickness=1)
        lines.append('c{},{},{}\n'.format(c_ind, center[0], center[1]))
    cv2.imwrite('{}/{}_{}_final_centroids.png'.format(save_root, args.cluster_task_name, args.cluster_subset), src)

elif args.cluster_strategy == 'density_log_3d':
    print('do UMAP 3D')
    reducer = umap.UMAP(random_state=42, metric=metric, n_components=3)
    feats_embedding = reducer.fit_transform(X_scaled)
    kmeans = cluster.MiniBatchKMeans(n_clusters=args.num_clusters)
    kmeans.fit(feats_embedding)
    final_centroids = kmeans.cluster_centers_
elif args.cluster_strategy == 'feature_kmeans':
    print('do kmeans')
    kmeans = cluster.MiniBatchKMeans(n_clusters=args.num_clusters)
    kmeans.fit(X_scaled)
    final_centroids = kmeans.cluster_centers_

print('do remap')

font = ImageFont.truetype("Gidole-Regular.ttf", size=42)
save_root3 = os.path.join(save_root, 'big_images')
# if os.path.exists(save_root3):
#     shutil.rmtree(save_root3, ignore_errors=True)
#     time.sleep(2)
os.makedirs(save_root3, exist_ok=True)

cluster_counts_per_case = []
# all_distances_per_category = {label: {cluster_index: [] for cluster_index in range(len(final_centroids))}
#                               for label in labels_dict[label_title].keys()}
limited_count = 2 if args.debug else 200000

current_label_counts = {label: 0 for label in labels_dict[label_title].keys()}

all_data_for_umap3dvis = []
all_distances_for_each_cluster = {k: 0 for k in range(len(final_centroids))}
for ind in np.arange(len(svs_filenames)):
    label = labels[ind]
    if current_label_counts[label] >= limited_count:
        continue
    else:
        current_label_counts[label] += 1

    label_text = labels_dict[label_title][label]
    As = []
    jjs = []
    svs_filename_list = svs_filenames[ind].split(',')
    print(ind, len(svs_filename_list))
    for j, filename in enumerate(svs_filename_list):
        svs_filename = os.path.join(args.dataset_root, 'svs', filename.replace(' ', '').replace('&', ''))
        slide = openslide.OpenSlide(svs_filename)
        svs_prefix = os.path.basename(svs_filename).replace(args.image_ext, '')
        patch_info_dict = get_coords(
            os.path.join('{}/patches/'.format(args.dataset_root), svs_prefix + '.h5'))
        patch_level = patch_info_dict['patch_level']
        dimension = slide.level_dimensions[patch_level]
        patch_size = patch_info_dict['patch_size']

        coords = patch_info_dict['locations']

        print(svs_prefix)
        feats = torch.load(os.path.join(args.feats_dir, svs_prefix + '.pt'))
        if torch.cuda.device_count() > 0:
            feats = feats.cuda()
        results_dict = model(feats, attention_only=True)
        A_raw = results_dict['A_raw'].detach().cpu().numpy()

        A = A_raw[args.cluster_task_index]
        print(A.shape)
        As.append(A)
        jjs.append(j * np.ones_like(A, dtype=np.int32))

    ccs = [len(A) for A in As]
    cc = [0] + np.cumsum(ccs).tolist()
    A = np.concatenate(As)  # 2000 + 3000
    jj = np.concatenate(jjs)
    assert len(A) == len(jj), 'wrong, check it'
    ref_scores = np.copy(A)
    for ind1 in range(len(A)):
        A[ind1] = score2percentile(A[ind1], ref_scores)

    sort_ind = np.argsort(A)
    num_ps = args.num_patches if len(sort_ind) > args.num_patches else len(sort_ind)
    if True:
        selected_indices = sort_ind[-int(num_ps):]
    else:
        selected_indices = sort_ind
    jj_selected = jj[selected_indices]

    for j, filename in enumerate(svs_filename_list):
        svs_filename = os.path.join(args.dataset_root, 'svs', filename.replace(' ', '').replace('&', ''))
        slide = openslide.OpenSlide(svs_filename)
        svs_prefix = os.path.basename(svs_filename).replace(args.image_ext, '')
        print(svs_prefix)

        dimension = slide.level_dimensions[1]
        if dimension[0] > 100000 or dimension[1] > 100000:
            vis_level = 2
        else:
            vis_level = 1
        print('dimension', dimension)

        patch_info_dict = get_coords(
            os.path.join('{}/patches/'.format(args.dataset_root), svs_prefix + '.h5'))
        patch_level = patch_info_dict['patch_level']
        dimension = slide.level_dimensions[patch_level]
        dimension_vis_level = slide.level_dimensions[vis_level]
        patch_size = patch_info_dict['patch_size']
        coords_original = patch_info_dict['locations']
        print('dimension_vis_level', dimension_vis_level)

        level_downsamples = _assertLevelDownsamples(slide)
        downsample0 = level_downsamples[patch_level]
        downsample = level_downsamples[vis_level]
        scale = [downsample0[0] / downsample[0], downsample0[1] / downsample[1]]  # Scaling from 0 to desired level
        # scale = [1 / downsample[0], 1 / downsample[1]]
        patch_size_vis_level = np.ceil(patch_size * scale[0]).astype(int)

        print('level_downsamples', level_downsamples)
        print('downsample0', downsample0)
        print('downsample', downsample)
        print('scale', scale)

        feats = torch.load(os.path.join(args.feats_dir, svs_prefix + '.pt'))
        if torch.cuda.device_count() > 0:
            feats = feats.cuda()
        results_dict = model(feats, attention_only=True)
        # A_raw = results_dict['A_raw'].detach().cpu().numpy()
        #
        # A = A_raw[args.cluster_task_index]
        # ref_scores = np.copy(A)
        # for ind1 in range(len(A)):
        #     A[ind1] = score2percentile(A[ind1], ref_scores)
        #
        # sort_ind = np.argsort(A)
        # num_ps = args.num_patches if len(sort_ind) > args.num_patches else len(sort_ind)
        # if True:
        #     selected_indices = sort_ind[-int(num_ps):]
        # else:
        #     selected_indices = sort_ind
        selected_indices_j = selected_indices[jj_selected == j] - cc[j]

        if len(selected_indices_j) == 0:
            print('no patches in this image')
            continue

        if 'after_encoder' in args.cluster_feat_name:
            xx = feats.cpu().numpy()[selected_indices_j]
        else:
            xx = feature_tensors[args.cluster_feat_name].cpu().numpy()[selected_indices_j]
        # xx = feature_tensors[args.cluster_feat_name].cpu().numpy()[selected_indices]
        reduced_feats = None
        if args.cluster_strategy == 'density_log_2d':
            xx_reduced = reducer.transform(xx)
            feats_embedding_inds0 = np.digitize(xx_reduced[:, 0], feats_embedding_bin_edges0)
            feats_embedding_inds1 = np.digitize(xx_reduced[:, 1], feats_embedding_bin_edges1)
            feats_embedding1_orig = np.stack([feats_embedding_inds0, feats_embedding_inds1]).T
            reduced_feats = feats_embedding1_orig
        elif args.cluster_strategy == 'density_log_3d':
            reduced_feats = reducer.transform(xx)
        elif args.cluster_strategy == 'feature_kmeans':
            reduced_feats = xx

        distances = distance_matrix(reduced_feats, final_centroids)
        nn_labels = np.argmin(distances, axis=1)

        unique1, counts1 = np.unique(nn_labels, return_counts=True)
        aaa = np.zeros_like(np.arange(len(final_centroids)))
        aaa[unique1] = counts1
        counts2 = counts1.astype(np.float32)
        cluster_counts_per_case.append([svs_prefix, label, label_text,
                                        patch_level, vis_level,
                                        dimension[0], dimension[1],
                                        dimension_vis_level[0], dimension_vis_level[1]] + aaa.tolist())

        coords = coords_original[selected_indices_j]
        coords1 = np.copy(coords)
        coords2 = np.floor(coords / patch_size).astype(np.int32)
        coords_in_vis_level = np.ceil(coords_original * np.array(scale)).astype(int)[selected_indices_j]

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        if True:
            thumbnail = slide.get_thumbnail(size=(np.ceil(dimension[0] / patch_size),
                                                  np.ceil(dimension[1] / patch_size)))
            thumbnail = np.array(thumbnail)[:, :, :3]
            thumbnail = thumbnail[:, :, ::-1]
            thumbnail = np.pad(thumbnail, ((0, 5), (0, 5), (0, 0)), constant_values=255)

            scores = A[cc[j]:(cc[j] + len(feats))] * 255. / 100
            scores = scores.astype(np.uint8)
            scores = cv2.applyColorMap(scores, cv2.COLORMAP_JET)

            heatmap = 255 * np.ones_like(thumbnail)
            coords_original1 = np.floor(coords_original / patch_size).astype(np.int32)
            for ind1, coord in enumerate(coords_original1):
                heatmap[coord[1], coord[0], :] = scores[ind1, 0, :]

            thumbnail_patches = np.copy(thumbnail)
            for coord in coords2:
                cv2.circle(thumbnail_patches, center=tuple(coord), radius=3, color=(255, 255, 0))

            thumbnail_selected_patches = np.copy(thumbnail)
            for coord in coords2:
                cv2.circle(thumbnail_selected_patches,
                           center=(coord[0], coord[1]),
                           radius=5, color=(30, 128, 255),
                           thickness=2)

            cv2.putText(thumbnail, text='{}'.format(label_text), org=(10, 10),
                        fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(thumbnail_patches, text='patches', org=(10, 10),
                        fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(heatmap, text='attention map', org=(10, 10),
                        fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(thumbnail_selected_patches, text='selected patches', org=(10, 10),
                        fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)

            cluster_heatmap = np.copy(thumbnail)  # 255 * np.ones_like(thumbnail)
            cv2.putText(cluster_heatmap, text='{}'.format(label_text), org=(10, 10),
                        fontFace=1, fontScale=1, color=(0, 0, 255), thickness=1)
            if thumbnail.shape[0] > thumbnail.shape[1]:
                final_img = np.concatenate([
                    np.concatenate([thumbnail, thumbnail_patches], axis=1),
                    np.concatenate([heatmap, cluster_heatmap], axis=1)
                ], axis=0)
            else:
                final_img = np.concatenate([
                    np.concatenate([thumbnail, thumbnail_patches], axis=0),
                    np.concatenate([heatmap, cluster_heatmap], axis=0)
                ], axis=1)
            # cv2.imwrite(save_filename, final_img)
            im_buffer = io.BytesIO()
            patch = Image.fromarray(final_img)
            patch.save(im_buffer, format='JPEG')
            patch_filename = 'prediction_thumbnails/{}.png'.format(svs_prefix)
            info = tarfile.TarInfo(name=patch_filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)
            del thumbnail, thumbnail_patches, heatmap, cluster_heatmap, thumbnail_selected_patches

        all_patches_json_dict = []
        all_patches = []
        all_patch_filenames = {iii: [] for iii in range(len(final_centroids))}
        for ind1, coord in enumerate(coords2):
            reduced_feat = reduced_feats[ind1]
            nn_label = nn_labels[ind1]
            min_distance = distances[ind1][nn_label]
            all_distances_for_each_cluster[nn_label] += min_distance

            # all_distances_per_category[label][nn_label].append(min_distance)

            coord_in_vis_level = coords_in_vis_level[ind1]
            annoid_str = "annoid-{:d}-{:d}-{:.3f}".format(nn_label, ind1, min_distance)
            all_patches_json_dict.append(
                new_web_annotation(nn_label, min_distance,
                                   coord_in_vis_level[0], coord_in_vis_level[1],
                                   patch_size_vis_level, patch_size_vis_level,
                                   annoid_str)
            )
            coord1 = coords1[ind1]
            patch = slide.read_region(location=(int(coord1[0]), int(coord1[1])), level=patch_level,
                                      size=(patch_size, patch_size)).convert('RGB')
            all_patches.append(transforms.ToTensor()(patch))

            I1 = ImageDraw.Draw(patch)
            # Add Text to an image
            I1.text((5, 5), "{:d},{:.3f}".format(nn_label, min_distance), fill=(0, 255, 255), font=font)
            patch_filename = 'selected_patches/{}/c{}/{}.jpg'.format(svs_prefix, nn_label, annoid_str)
            # patch.save(patch_filename)
            all_patch_filenames[nn_label].append(
                '{:d},{:d},{}.jpg,{},{:.6f},{:.6f},{:.6f},{:d},{:.6f}\n'.format(
                    coord_in_vis_level[0], coord_in_vis_level[1], annoid_str,
                    patch_filename,
                    reduced_feat[0],
                    reduced_feat[1],
                    reduced_feat[2] if len(reduced_feat) == 3 else 0,
                    nn_label,
                    min_distance))

            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=patch_filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)

            # add data for umap3d visualization
            all_data_for_umap3dvis.append(
                '{},{:.6f},{:.6f},{:.6f},{:d},{:.6f},{:d},{:d},{}.jpg\n'.format(patch_filename,
                                                                                reduced_feat[0],
                                                                                reduced_feat[1],
                                                                                reduced_feat[2] if len(
                                                                                    reduced_feat) == 3 else 0,
                                                                                nn_label,
                                                                                min_distance,
                                                                                coord_in_vis_level[0],
                                                                                coord_in_vis_level[1],
                                                                                annoid_str)
            )

        for iii in range(len(final_centroids)):
            txt_buffer = io.StringIO(''.join(all_patch_filenames[iii]))
            btxt_buffer = io.BytesIO(txt_buffer.read().encode())
            txt_filename = 'selected_patches/{}/c{}.txt'.format(svs_prefix, iii)
            info = tarfile.TarInfo(name=txt_filename)
            info.size = btxt_buffer.getbuffer().nbytes
            info.mtime = time.time()
            btxt_buffer.seek(0)
            tar_fp.addfile(info, btxt_buffer)

        if len(all_patches) > 0:
            im_buffer = io.BytesIO()
            save_image(all_patches, im_buffer, format='JPEG')
            all_patches_filename = 'selected_patches/{}_all_patches_image.jpg'.format(svs_prefix)
            info = tarfile.TarInfo(name=all_patches_filename)
            info.size = im_buffer.getbuffer().nbytes
            # info.size = im_buffer.__sizeof__()
            # info.size = len(im_buffer.getvalue())
            # info.size = im_buffer.tell()
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)
            del all_patches

        tar_fp.close()
        with open('{}/{}.tar.gz'.format(save_root, svs_prefix), 'wb') as fp:
            fp.write(fh.getvalue())

        with open('{}/{}_patches_annotations.json'.format(save_root3, svs_prefix), 'w') as fp:
            json.dump(all_patches_json_dict, fp, indent=2)

        print("do heatmap big")
        save_filename = '{}/{}_big_orig.zip'.format(save_root3, svs_prefix)
        if not os.path.exists(save_filename):
            img = visWSI(slide, vis_level=vis_level)
            print(type(img), img.size)
            img_vips = pyvips.Image.new_from_array(img)
            img_vips.dzsave(save_filename)
            # img.save('{}/{}_big_orig.jpg'.format(save_root, svs_prefix))
            time.sleep(1)
            del img, img_vips

        save_filename = '{}/{}_big_attention_map.zip'.format(save_root3, svs_prefix)
        if not os.path.exists(save_filename):
            img = visHeatmap(slide, scores=A[cc[j]:(cc[j] + len(feats))], coords=coords_original,
                             vis_level=vis_level, patch_size=(patch_size, patch_size),
                             convert_to_percentiles=False)
            img_vips = pyvips.Image.new_from_array(img)
            img_vips.dzsave(save_filename)
            time.sleep(1)
            del img, img_vips

if len(all_data_for_umap3dvis) > 0:
    with open('{}/all_data_for_umap3dvis.txt'.format(save_root), 'w') as fp:
        fp.writelines(all_data_for_umap3dvis)

lines = []
for c_ind, center in enumerate(final_centroids):
    lines.append('c{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(c_ind,
                                                                                 center[0],
                                                                                 center[1],
                                                                                 center[2] if len(center) > 2 else 0,
                                                                                 np.min(all_distances_for_each_cluster[
                                                                                            c_ind]),
                                                                                 np.max(all_distances_for_each_cluster[
                                                                                            c_ind]),
                                                                                 np.mean(all_distances_for_each_cluster[
                                                                                             c_ind]),
                                                                                 np.median(
                                                                                     all_distances_for_each_cluster[
                                                                                         c_ind]))
                 )
with open('{}/{}_{}_final_centroids.txt'.format(save_root, args.cluster_task_name, args.cluster_subset), 'w') as fp:
    fp.writelines(lines)

cluster_cols = ['c{}'.format(i) for i in range(len(final_centroids))]
counts_df = pd.DataFrame(cluster_counts_per_case,
                         columns=['filename', args.cluster_task_name, '{}Name'.format(args.cluster_task_name),
                                  'patch_level', 'vis_level',
                                  'patch_level_width', 'patch_level_height',
                                  'vis_level_width', 'vis_level_height'] + cluster_cols)
counts_df = counts_df.sort_values(args.cluster_task_name)
counts_df.to_csv('{}/all_info.csv'.format(save_root))

counts_df1 = counts_df[['{}Name'.format(args.cluster_task_name)] + cluster_cols].copy()
cluster_counts_by_category = counts_df[['{}Name'.format(args.cluster_task_name)] + cluster_cols].copy().groupby(
    '{}Name'.format(args.cluster_task_name)).sum()

for label, label_text in labels_dict[label_title].items():
    counts_df1 = counts_df[counts_df['{}Name'.format(args.cluster_task_name)] == label_text].reset_index(drop=True)[
        'filename']
    counts_df1.to_csv('{}/counts_in_for_label_{}.csv'.format(save_root, label_text))

cluster_counts_per_case = counts_df[
    ['filename', args.cluster_task_name, '{}Name'.format(args.cluster_task_name)] + cluster_cols].copy().reset_index(
    drop=True)
cluster_counts_per_case.to_csv('{}/cluster_counts_per_case.csv'.format(save_root))

# get fractions
fractions_df = cluster_counts_by_category.copy()
fractions_df = fractions_df.div(fractions_df.sum(axis=1), axis=0)
fractions_df.to_csv('{}/cluster_counts_by_category_fraction.csv'.format(save_root), float_format='%.3f')

# plot fraction bar plots
save_filename = os.path.join('{}/cluster_counts_by_category_fraction.png'.format(save_root))
df = fractions_df.copy()
cluster_names = df.columns.values
label_name = df.index.name
label_names = df.index.values

if True:
    fig, axes = plt.subplots(1, len(label_names), figsize=(15, 5))
    fig.suptitle('{}'.format(label_name))
    for i, label in enumerate(label_names):
        values = df.iloc[i].loc[cluster_names].values
        axes[i].bar(cluster_names, values, width=0.4)
        axes[i].set_title(label_names[i])
    plt.savefig(
        save_filename,
        bbox_inches='tight', dpi=600)
    plt.close(fig)

if True:
    bar = np.arange(len(cluster_names))
    barWidth = 0.25
    fig, ax = plt.subplots()
    for i, label in enumerate(label_names):
        values = df.iloc[i].loc[cluster_names].values
        plt.bar(bar, values, width=barWidth, label=label)
        bar = [x + barWidth for x in bar]
    plt.xticks([r + (len(label_names) / 2.) * barWidth for r in range(len(cluster_names))],
               cluster_names)
    # plt.grid()
    fig.suptitle('{}'.format(label_name.replace('Name', '')))
    plt.xlabel('Clusters', fontweight='bold', fontsize=15)
    plt.ylabel('Fractions', fontweight='bold', fontsize=15)
    plt.legend()
    plt.savefig(
        save_filename.replace('.png', '_bargroup.png'),
        bbox_inches='tight', dpi=600)
    plt.close(fig)

# box plots and
from scipy.stats import ranksums


# bar = np.arange(len(cluster_names))
# barWidth = 0.25
# fig, ax = plt.subplots()
# for i, group_id in enumerate(group_ids):
#     values = list(all_distances_per_category[group_id].values())
#     plt.boxplot(values, positions=bar, widths=barWidth)
#     bar = [x + barWidth for x in bar]
# plt.legend()
# plt.savefig(
#     os.path.join('{}/cluster_counts_by_category_fraction.png'.format(save_root)),
#     bbox_inches='tight', dpi=600)
# plt.close(fig)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# colors = ['#D7191C', '#2C7BB6']
# data_a = list(all_distances_per_category[group_ids[0]].values())
# data_b = list(all_distances_per_category[group_ids[1]].values())
# ticks = cluster_names
# plt.figure()
# bpl = plt.boxplot(data_a, positions=np.arange(len(data_a)) * 2.0 - 0.4, sym='', widths=0.6)
# bpr = plt.boxplot(data_b, positions=np.arange(len(data_b)) * 2.0 + 0.4, sym='', widths=0.6)
# set_box_color(bpl, colors[0])  # colors are from http://colorbrewer2.org/
# set_box_color(bpr, colors[1])
#
# # draw circles
# xs = []
# ys = []
# rs = []
# for x1, y1, x2, y2 in zip(np.arange(len(data_a)) * 2.0 - 0.4, data_a,
#                           np.arange(len(data_b)) * 2.0 + 0.4, data_b):
#     xs.append((x1 + x2) / 2)
#     rs.append(ranksums(y1, y2)[1])
#     yy = y1 + y2
#     if len(yy) > 0:
#         ys.append(max(yy))
#     x1 = np.random.normal(x1, 0.04, size=len(y1))
#     plt.scatter(x1, y1, c=colors[0], alpha=0.2)
#     x2 = np.random.normal(x2, 0.04, size=len(y2))
#     plt.scatter(x2, y2, c=colors[1], alpha=0.2)
#
# # draw ranksum result
# # y = max(ys)
# # for x, r in zip(xs, rs):
# #     plt.text(x, y, '{:.3f}'.format(r) if isinstance(r, float) else '')
#
# # draw temporary red and blue lines and use them to create a legend
# plt.plot([], c=colors[0], label=labels_dict[args.cluster_task_name][group_ids[0]])
# plt.plot([], c=colors[1], label=labels_dict[args.cluster_task_name][group_ids[1]])
# plt.legend()
#
# ticks1 = ['{}\n{:.3f}'.format(xx, pp) for xx, pp in zip(ticks, rs)]
# plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks1)
# plt.xlim(-2, len(ticks) * 2)
# # plt.ylim(0, 8)
# # plt.tight_layout()
# plt.title(args.cluster_task_name)
# plt.xlabel('Clusters & P-value', fontweight='bold', fontsize=15)
# plt.ylabel('Distances to cluster center', fontweight='bold', fontsize=15)
# plt.savefig(
#     os.path.join('{}/cluster_distance_boxplot_by_category.png'.format(save_root)),
#     bbox_inches='tight', dpi=600)
# plt.close(fig)

keys = list(labels_dict[args.cluster_task_name].keys())
for ii, key0 in enumerate(keys):
    for jj, key1 in enumerate(keys):
        if ii <= jj:
            continue

        #################### plot fraction box plots
        df0 = cluster_counts_per_case[
            cluster_counts_per_case['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][
                key0]][cluster_names]
        df1 = cluster_counts_per_case[
            cluster_counts_per_case['{}Name'.format(args.cluster_task_name)] == labels_dict[args.cluster_task_name][
                key1]][cluster_names]
        if len(df0) == 0 or len(df1) == 0:
            continue
        df0 = df0.div(df0.sum(axis=1), axis=0)
        df1 = df1.div(df1.sum(axis=1), axis=0)

        colors = ['#D7191C', '#2C7BB6']
        data_a = df0.values.T.tolist()
        data_b = df1.values.T.tolist()
        ticks = cluster_names
        plt.figure()
        bpl = plt.boxplot(data_a, positions=np.arange(len(data_a)) * 2.0 - 0.4, sym='', widths=0.6)
        bpr = plt.boxplot(data_b, positions=np.arange(len(data_b)) * 2.0 + 0.4, sym='', widths=0.6)
        set_box_color(bpl, colors[0])  # colors are from http://colorbrewer2.org/
        set_box_color(bpr, colors[1])

        # draw circles
        xs = []
        ys = []
        zs = []
        rs = []
        for x1, y1, x2, y2 in zip(np.arange(len(data_a)) * 2.0 - 0.4, data_a,
                                  np.arange(len(data_b)) * 2.0 + 0.4, data_b):
            xs.append((x1 + x2) / 2)
            z, r = ranksums(y1, y2)
            zs.append(z)
            rs.append(r)
            yy = y1 + y2
            if len(yy) > 0:
                ys.append(max(yy))
            x1 = np.random.normal(x1, 0.04, size=len(y1))
            plt.scatter(x1, y1, c=colors[0], alpha=0.2)
            x2 = np.random.normal(x2, 0.04, size=len(y2))
            plt.scatter(x2, y2, c=colors[1], alpha=0.2)

        # draw ranksum result
        # y = max(ys)
        # for x, r in zip(xs, rs):
        #     plt.text(x, y, '{:.3f}'.format(r) if isinstance(r, float) else '')

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=colors[0], label=labels_dict[args.cluster_task_name][key0])
        plt.plot([], c=colors[1], label=labels_dict[args.cluster_task_name][key1])
        plt.legend()

        ticks1 = ['{}\n{:.3f}\n{:.3f}'.format(xx, zz, pp) for xx, zz, pp in zip(ticks, zs, rs)]
        plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks1)
        plt.xlim(-2, len(ticks) * 2)
        # plt.ylim(0, 8)
        # plt.tight_layout()
        plt.title(args.cluster_task_name)
        plt.xlabel('Clusters & P-value', fontweight='bold', fontsize=15)
        plt.ylabel('Patch fractions', fontweight='bold', fontsize=15)
        plt.savefig(
            os.path.join('{}/cluster_fractions_boxplot_between_{}_and_{}.png'.format(save_root, key0, key1)),
            bbox_inches='tight', dpi=600)
        plt.savefig(
            os.path.join('{}/cluster_fractions_boxplot_between_{}_and_{}.png'.format(save_root, key1, key0)),
            bbox_inches='tight', dpi=600)
        plt.close(fig)
