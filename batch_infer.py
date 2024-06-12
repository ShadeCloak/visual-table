import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

# from torchvision.datasets import CocoDetection
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from groundingdino.util.utils import get_phrases_from_posmap
from PIL import Image
import json
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model
    

class CocoDetection(torchvision.datasets.CocoDetection, Dataset):
    def __init__(self, image_dir, anno_path, transforms=None):

        self.image_dir = image_dir 
        self.transform = transforms
        self.image_data = [ json.loads(item) for item in open(anno_path, 'r', encoding='utf-8')]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_path = self.image_data[idx]['image_path']
        image = Image.open(self.image_dir + '\\' + image_path).convert('RGB')
        w, h = image.size

        image_attribute_new = {}
        image_attribute_new['image_path'] = self.image_data[idx]['image_path']
        image_attribute_new['tag'] = self.image_data[idx]['key_words']
        image_attribute_new['orig_size'] = torch.as_tensor([int(h), int(w)])


        if self.transform:
            image, image_attribute = self.transform(image, image_attribute_new)

        return image, image_attribute


def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    # model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    
    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    # tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    # postprocessor = PostProcessCocoGrounding(
    #     coco_api=dataset.coco, tokenlizer=tokenlizer)

    out_list = []
    # run inference
    for i, (images, images_attribute) in enumerate(data_loader):
        # get images and captions
        # images = images.tensors.to(args.device)
        images = images.tensors
        bs, _, H, W = images.shape

        # build captions
        input_captions = [item['tag'] for item in images_attribute]

        # feed to the model
        with torch.no_grad():
            outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in images_attribute], dim=0)

        for i in range(bs):
            
            prediction_logits = outputs["pred_logits"].cpu().sigmoid()[i]  # prediction_logits.shape = (nq, 256)
            prediction_boxes = outputs["pred_boxes"].cpu()[i]  # prediction_boxes.shape = (nq, 4)

            mask = prediction_logits.max(dim=1)[0] > 0.35
            logits = prediction_logits[mask]  # logits.shape = (n, 256)
            boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

            tokenizer = model.tokenizer
            tokenized = tokenizer(input_captions[i])

            phrases = [
                get_phrases_from_posmap(logit > 0.10, tokenized, tokenizer).replace('.', '')
                for logit
                in logits
            ]

            h, w = images_attribute[i]['size']
            images_attribute[i]['bbox'] = (boxes * torch.Tensor([W, H, W, H]) / torch.Tensor([w, h, w, h])).tolist()
            images_attribute[i]['phrases'] = phrases
            images_attribute[i]['orig_size'] = images_attribute[i]['orig_size'].tolist()
            images_attribute[i]['size'] = images_attribute[i]['size'].tolist()
            images_attribute[i]['logits'] = logits.max(dim=1)[0].tolist()
            out_list.append(images_attribute[i])

        with open('out.jsonl', 'w', encoding='utf-8') as f:
            for item in out_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO batch infer", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        help="path to config file",
                        default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, help="path to checkpoint file", default="weights/groundingdino_swint_ogc.pth"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        help="coco root", default="C:\\Users\\曾煜\\OneDrive\\桌面\\Lvis\\train2017_entity_bbox.jsonl")
    parser.add_argument("--image_dir", type=str,
                        help="coco image dir", default="C:\\Users\\曾煜\\OneDrive\\桌面\\Lvis\\train2017")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()

    main(args)
