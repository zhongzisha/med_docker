import os
import numpy as np
import pandas as pd
import time
import h5py
import openslide
from wsi_core.wsi_utils import save_hdf5, HF_MODELS_DICT
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import ResNetModel, BeitModel, BitModel, ConvNextModel, CvtModel, DeiTModel, \
    DinatModel, DPTModel, EfficientFormerModel, GLPNModel, MobileNetV1Model, ImageGPTModel, \
    LevitModel, MobileNetV1Model, MobileNetV2Model, MobileViTModel, NatModel, PoolFormerModel, \
    SwinModel, Swinv2Model, ViTModel, ViTHybridModel, ViTMAEModel, ViTMSNModel
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import gc


class PatchDatasetV2(Dataset):
    def __init__(self, slide, coords, patch_level, patch_size):
        self.slide = slide
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        coord = self.coords[index]
        patch = self.slide.read_region(location=(int(coord[0]), int(coord[1])), level=self.patch_level,
                                       size=(self.patch_size, self.patch_size)).convert('RGB')  # BGR
        return {
            'pixel_values': patch,
            'coords': coord
        }


def mainV2():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default='./data/temp/')
    parser.add_argument('--data_slide_dir', type=str, default='./data/temp/svs')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str,
                        default='./data/temp/all.xlsx')
    parser.add_argument('--feat_dir', type=str, default='./data/temp/feats/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='mobilenetv3')
    parser.add_argument('--feature_extraction_ckpt_path', type=str, default="None")
    parser.add_argument('--crop_patch_size', type=int, default=224)
    parser.add_argument('--pretrain_type', type=str, default='imagenet')
    args = parser.parse_args()

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    if 'xlsx' in csv_path:
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    if 'slide_id' not in df:
        slide_ids = [os.path.basename(f).replace(args.slide_ext, '') for f in df['DX_filename'].values]
        df['slide_id'] = slide_ids
    else:
        df['slide_id'] = df['slide_id'].astype(str)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
    existed_prefixes = set([os.path.basename(f).replace('.pt', '') for f in dest_files])

    drop_ids = []
    for ind, f in enumerate(df['DX_filename'].values):
        svs_prefix = os.path.basename(f).replace(args.slide_ext, '')
        if svs_prefix in existed_prefixes:
            drop_ids.append(ind)
    if len(drop_ids) > 0:
        df = df.drop(drop_ids)

    sub_df = df

    # models_dict = {
    #     'resnet18': ['ResNetModel', 'microsoft/resnet-18', 'pooler_output'],
    #     'resnet50': ['ResNetModel', 'microsoft/resnet-50', 'pooler_output'],
    #     'beit': ['BeitModel', 'microsoft/beit-base-patch16-224-pt22k', 'pooler_output'],
    #     'bit': ['BitModel', 'google/bit-50', 'pooler_output', 'pooler_output'],
    #     'convnext': ['ConvNextModel', 'facebook/convnext-tiny-224', 'pooler_output'],
    #     # 'cvt': ['CvtModel', 'microsoft/cvt-13', 'cls_token_value'],
    #     # 'deit': ['DeiTModel', 'facebook/deit-base-distilled-patch16-224', 'pooler_output'],
    #     # 'dinat': ['DinatModel', 'shi-labs/dinat-mini-in1k-224'],
    #     # 'dpt': ['DPTModel', 'Intel/dpt-large', 'pooler_output'],
    #     # 'efficientformer': ['EfficientFormerModel', 'snap-research/efficientformer-l1-300'],
    #     # 'efficientnet': ['EfficientNetModel', 'google/efficientnet-b7'],
    #     # 'glpn': ['GLPNModel', 'vinvino02/glpn-kitti'],
    #     # 'imagegpt': ['ImageGPTModel', 'openai/imagegpt-small'],
    #     # 'levit': ['LevitModel', 'facebook/levit-128S', 'pooler_output'],
    #     'mobilenetv1': ['MobileNetV1Model', 'google/mobilenet_v1_1.0_224', 'pooler_output'],
    #     'mobilenetv2': ['MobileNetV2Model', 'google/mobilenet_v2_1.0_224', 'pooler_output'],
    #     'mobilevit': ['MobileViTModel', 'apple/mobilevit-small', 'pooler_output'],
    #     # 'nat': ['NatModel', 'shi-labs/nat-mini-in1k-224'],
    #     # 'poolformer': ['PoolFormerModel', 'sail/poolformer_s12'],
    #     # 'swin': ['SwinModel', 'microsoft/swin-tiny-patch4-window7-224', 'pooler_output'],
    #     'swinv2': ['Swinv2Model', 'microsoft/swinv2-tiny-patch4-window8-256', 'pooler_output'],
    #     'vit': ['ViTModel', 'google/vit-base-patch16-224-in21k', 'pooler_output'],
    #     'vithybrid': ['ViTHybridModel', 'google/vit-hybrid-base-bit-384', 'pooler_output'],
    #     # 'vitmae': ['ViTMAEModel', 'facebook/vit-mae-base'],
    #     # 'vitmsn': ['ViTMSNModel', 'facebook/vit-msn-small']
    # }
    batch_size = args.batch_size

    model_name = args.model_name
    model_params = HF_MODELS_DICT[model_name]
    # for model_name, model_params in HF_MODELS_DICT.items():

    feature_tensors = {}
    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()

        return hook

    time.sleep(1)
    for index, slide_file_path in enumerate(sub_df['DX_filename'].values):
        svs_prefix = os.path.basename(slide_file_path).replace(args.slide_ext, '')

        h5_file_path = os.path.join(args.data_h5_dir, 'patches', svs_prefix + '.h5')
        if not os.path.exists(h5_file_path):
            continue

        local_svs_filename = slide_file_path

        h5file = h5py.File(h5_file_path, 'r')
        dset = h5file['coords']
        coords = dset[:]
        patch_level = h5file['coords'].attrs['patch_level']
        patch_size = h5file['coords'].attrs['patch_size']

        # filter the coords
        slide = openslide.OpenSlide(local_svs_filename)
        time.sleep(1)
        print('extract features')

        if model_name == 'mobilenetv3':
            model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            model.flatten.register_forward_hook(get_activation('after_flatten'))
            image_processor = None
        else:
            model = globals()[model_params[0]].from_pretrained(model_params[1])
            config = None
            transform = None
            image_processor = AutoImageProcessor.from_pretrained(model_params[1])
        if torch.cuda.device_count()>0:
            model = model.cuda()
        model.eval()

        # print('image_processor')
        # print(image_processor)

        def collate_fn2(examples):
            pixel_values = [transform(example["pixel_values"]) for example in examples]
            pixel_values = torch.stack(pixel_values)
            coords = np.vstack([example["coords"] for example in examples])
            return pixel_values, coords


        def collate_fn(examples):
            pixel_values = image_processor([example["pixel_values"] for example in examples], return_tensors='pt')
            coords = np.vstack([example["coords"] for example in examples])
            return pixel_values['pixel_values'], coords


        dataset = PatchDatasetV2(slide, coords, patch_level, patch_size)
        kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': False}
        kwargs = {}
        if model_name == 'mobilenetv3':
            loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_fn2)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_fn)

        output_path = os.path.join(args.feat_dir, 'h5_files', svs_prefix + '.h5')
        time_start = time.time()
        mode = 'w'
        for count, (images, coords) in enumerate(loader):
            with torch.no_grad():
                if count % 20 == 0:
                    print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
                if torch.cuda.device_count() >0:
                    images = images.cuda()

                if model_name == 'mobilenetv3':
                    output = model(images)
                    features = feature_tensors.get('after_flatten_feat').cpu().numpy().reshape((len(coords), -1))
                else:
                    features = model(images, return_dict=True)
                    features = features.get(model_params[2]).cpu().numpy().reshape((len(coords), -1))

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))
        del model
        gc.collect()

        with h5py.File(output_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            features = torch.from_numpy(features)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', svs_prefix + '.pt'))

        slide.close()


if __name__ == '__main__':
    mainV2()