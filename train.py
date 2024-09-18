# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import sys
sys.path.append("..")
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything_training import sam_model_registry
from network import MaskDecoderHQ, MaskDecoderHQ1
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from pycocotools import mask as coco_mask
import transform as T
import random
import utils.misc as misc
from skimage import io


# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", type=str, default= "/home/bme001/20225531/CATCH_dataset/CATCH/TRAIN/",
                    help="path to training slide files")  # "/home/bme001/20225531/CATCH_dataset/CATCH/TRAIN/", "/home/bme001/20225531/DCIS/TRAIN/"
parser.add_argument("-train_annotation_file", type=str, default= "/home/bme001/20225531/CATCH_dataset/CATCH_separate/train.json",
                    help="path to training annotation file")  # "/home/bme001/20225531/CATCH_dataset/CATCH_separate/train.json", "/home/bme001/20225531/DCIS/DCIS_train.json"
parser.add_argument("-val_annotation_file", type=str, default= "/home/bme001/20225531/CATCH_dataset/CATCH_separate/val.json",
                    help="path to validation annotation file")  # "/home/bme001/20225531/CATCH_dataset/CATCH_separate/val.json", "/home/bme001/20225531/DCIS/DCIS_val.json"
parser.add_argument("-task_name", type=str, default="ours_low5_twonets_noise_20epo_vitb_bs2_lr-3")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default='/home/bme001/20225531/segment-anything/checkpoint/sam_vit_b_01ec64.pth')  #"/home/bme001/20225531/segment-anything/work_dir/SAM/sam_vit_b_01ec64.pth", '/home/bme001/20225531/sam-hq/pretrained_checkpoint/mobile_sam.pt'
parser.add_argument("--load_pretrain", type=bool, default=True, help="use wandb to monitor training")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-patch_num", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("-use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--visualize", type=str, default=False)
args = parser.parse_args()

if args.use_wandb:
    import wandb
    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,})
# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = int(box[0]), int(box[1])
    w, h = int(box[2] - box[0]), int(box[3] - box[1])
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects([polygons], height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def make_transforms(if_train):
    if if_train:
        return T.Compose([          
            T.RandomHorizontalFlip(),
            T.RandomColor(),
        ])
    else:
        return None

class NpyDataset(Dataset):
    def __init__(self, args, if_train, train_list):
        self.if_train = if_train
        self.Transform = make_transforms(self.if_train)
        self.data_dir = args.data_dir
        self.train_list = train_list
        self.image_low_folder = '/home/bme001/20225531/CATCH_dataset/train_patches/train_patches_low/images'
        self.patch_size = 1024

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        image_path = self.train_list[idx]
        mask_path = image_path.replace('images', 'masks')
        slide_name = image_path.split('/')[-2]
        patch_count = image_path.split('/')[-1].split('_[')[0]
        pattern = os.path.join(self.image_low_folder, slide_name, '{}_*'.format(patch_count))
        file= glob.glob(pattern)
        image_low_path = os.path.join(self.image_low_folder, slide_name, file[0])
        mask_low_path = image_low_path.replace('images', 'masks')

        image = io.imread(image_path)  
        mask = io.imread(mask_path) 
        image_low = io.imread(image_low_path)
        mask_low = io.imread(mask_low_path)

        mask = torch.as_tensor(mask, dtype=torch.float32, device=args.device).unsqueeze(0)
        mask = mask/255.0  # torch.Size([1, 1024, 1024])
        mask_low = torch.as_tensor(mask_low, dtype=torch.float32, device=args.device).unsqueeze(0)
        mask_low = mask_low/255.0  # torch.Size([1, 1024, 1024])
        if mask.any() > 0:
            image = torch.as_tensor(image, dtype=torch.uint8, device=args.device).permute(2, 0, 1)  # image (2431, 2614, 3)
            patch = torch.as_tensor(image_low, dtype=torch.uint8, device=args.device).permute(2, 0, 1)
            target = {}
            target['patch'] = patch
            target['masks'] = mask
            target['patch_mask'] = mask_low
            # data augmentation
            if self.if_train:
                img, target = self.Transform(image, target)  # image torch.Size([3,1024,1024])
            else:
                img = image
            # img = (img - self.pixel_mean) / self.pixel_std
            mask = target['masks']
            patch = target['patch']
            patch_mask = target['patch_mask']

            dict_input_high = {}
            dict_input_low = {}
            dict_input_high['image'] = img
            dict_input_high['masks'] = mask
            dict_input_low['image'] = patch
            dict_input_low['masks'] = patch_mask
            mask_prompt = mask * 255.0
            patch_mask_prompt = patch_mask * 255.0
            input_keys = ['box','point','noise_mask']
            try:
                labels_points = misc.masks_sample_points(mask_prompt)
                labels_points_low = misc.masks_sample_points(patch_mask_prompt)
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            input_type = random.choice(input_keys)
            if input_type == 'box':
                boxes = misc.masks_to_boxes(mask_prompt)
                boxes = torch.as_tensor(boxes, device=args.device)
                boxes = misc.box_noise(boxes)
                boxes_low_noised = boxes / 2 + 256
                dict_input_high['boxes'] = boxes
                dict_input_low['boxes'] = boxes_low_noised
            elif input_type == 'point':
                point_coords = labels_points
                dict_input_high['point_coords'] = point_coords
                dict_input_high['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                ##################################################
                point_coords_low = labels_points_low
                dict_input_low['point_coords'] = point_coords_low
                dict_input_low['point_labels'] = torch.ones(point_coords_low.shape[1], device=point_coords.device)[None,:]
            elif input_type == 'noise_mask':
                labels_256 = F.interpolate(mask_prompt.unsqueeze(0), size=(256, 256), mode='bilinear')
                labels_noisemask = misc.masks_noise(labels_256)
                dict_input_high['mask_inputs'] = labels_noisemask
                ##################################################
                labels_256_low = F.interpolate(patch_mask_prompt.unsqueeze(0), size=(256, 256), mode='bilinear')
                labels_noisemask_low = misc.masks_noise(labels_256_low)
                dict_input_low['mask_inputs'] = labels_noisemask_low
            else:
                raise NotImplementedError
            dict_input_high['original_size'] = img.shape[-2:]
            dict_input_low['original_size'] = patch.shape[-2:]
            dict_input_high['slide_name'] = slide_name
            patch_info_high = [dict_input_high]
            patch_info_low = [dict_input_low]
                   
        return {"high": patch_info_high, "low": patch_info_low}
    

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def validate(args, sam_model, net, net1):
    val_dataset = NpyDataset(args, if_train=False)
    print("Number of validating samples: ", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=trivial_batch_collator,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    sam_model = sam_model.to(device)
    sam_model.eval()
    net = net.to(device)
    net.eval()
    net1 = net1.to(device)
    net1.eval()
    epoch_loss = 0
    val_dice = 0
    val_ce = 0
    for step, patch_infos in enumerate(tqdm(val_dataloader)):
        batched_input_high = patch_infos[0]["high"]
        batched_input_low = patch_infos[0]["low"]
        masks = torch.stack([x["masks"] for x in batched_input_high], dim=0)
        patch_masks = torch.stack([x["masks"] for x in batched_input_low], dim=0)

        with torch.no_grad():
            batched_output_high, interm_embeddings_high = sam_model(batched_input_high, multimask_output=False)
            batched_output_low, interm_embeddings_low = sam_model(batched_input_low, multimask_output=False)

        batch_len_high = len(batched_output_high)
        encoder_embedding_high = torch.cat([batched_output_high[i_l]['encoder_embedding'] for i_l in range(batch_len_high)], dim=0)
        image_pe_high = [batched_output_high[i_l]['image_pe'] for i_l in range(batch_len_high)]
        sparse_embeddings_high = [batched_output_high[i_l]['sparse_embeddings'] for i_l in range(batch_len_high)]
        dense_embeddings_high = [batched_output_high[i_l]['dense_embeddings'] for i_l in range(batch_len_high)]

        batch_len_low = len(batched_output_low)
        encoder_embedding_low = torch.cat([batched_output_low[i_l]['encoder_embedding'] for i_l in range(batch_len_low)], dim=0)
        image_pe_low = [batched_output_low[i_l]['image_pe'] for i_l in range(batch_len_low)]
        sparse_embeddings_low = [batched_output_low[i_l]['sparse_embeddings'] for i_l in range(batch_len_low)]
        dense_embeddings_low = [batched_output_low[i_l]['dense_embeddings'] for i_l in range(batch_len_low)]
        
        with torch.no_grad():
            _, _, masks_hq_token_high, hq_image_embeddings_high = net(
                image_embeddings=encoder_embedding_high,
                image_pe=image_pe_high,
                sparse_prompt_embeddings=sparse_embeddings_high,
                dense_prompt_embeddings=dense_embeddings_high,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings_high,
            )
            _, _, masks_hq_token_low, hq_image_embeddings_low = net1(
                image_embeddings=encoder_embedding_low,
                image_pe=image_pe_low,
                sparse_prompt_embeddings=sparse_embeddings_low,
                dense_prompt_embeddings=dense_embeddings_low,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings_low,
            )

            masks_hq_token = masks_hq_token_high + masks_hq_token_low
            masks_hq_high = (masks_hq_token @ hq_image_embeddings_high).view(10, -1, 256, 256)
            masks_hq_low = (masks_hq_token @ hq_image_embeddings_low).view(10, -1, 256, 256)

            masks_hq_high = F.interpolate(
            masks_hq_high,
            (1024,1024),
            mode="bilinear",
            align_corners=False,
            )

            masks_hq_low = F.interpolate(
            masks_hq_low,
            (1024,1024),
            mode="bilinear",
            align_corners=False,
            )
            
        dice_loss_high = seg_loss(masks_hq_high, masks)
        cross_loss_high = ce_loss(masks_hq_high, masks.float())
        dice_loss_low = seg_loss(masks_hq_low, patch_masks)
        cross_loss_low = ce_loss(masks_hq_low, patch_masks.float())
        dice_loss = dice_loss_high + dice_loss_low
        cross_loss = cross_loss_high + cross_loss_low
        loss = dice_loss + cross_loss
        epoch_loss += loss.item()
        val_dice += dice_loss.item()
        val_ce += cross_loss.item()
    epoch_loss /= step
    val_dice /= step
    val_ce /= step
    net.train()
    net1.train()
    return val_dice, val_ce, epoch_loss

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    for param in sam_model.parameters():
        param.requires_grad = False

    net = MaskDecoderHQ(args.model_type).to(device).train()
    net1 = MaskDecoderHQ1(args.model_type).to(device).train()
    print("Number of total parameters: ",
        sum(p.numel() for p in net.parameters())) 
    print("Number of trainable parameters: ",
        sum(p.numel() for p in net.parameters() if p.requires_grad),)  

    net_params = list(net.parameters())
    net_params.extend(list(net1.parameters()))
    optimizer = torch.optim.AdamW(
        net_params, lr=args.lr, weight_decay=args.weight_decay)
    # print("Number of image encoder and mask decoder parameters: ",
    #     sum(p.numel() for p in img_mask_encdec_params if p.requires_grad)) 
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    train_losses_dice = []
    train_losses_ce = []
    val_losses = []
    val_losses_dice = []
    val_losses_ce = []
    best_loss = 1e10
    high_patch_path = "/home/bme001/20225531/CATCH_dataset/train_patches/train_patches_high/images"
    patch_per_slide = 10
    random.seed(2024)
    train_patches = []
    slides = os.listdir(high_patch_path)
    for slide in slides:
        patches_path = os.path.join(high_patch_path, slide)
        all_patches = [file for file in os.listdir(patches_path) if file.endswith('.png')]
        selected_patches = []
        selected_patches.extend(random.sample(all_patches, min(patch_per_slide, len(all_patches))))
        selected_patches = [os.path.join(patches_path, item) for item in selected_patches]
        train_patches.extend(selected_patches)
    random.shuffle(train_patches)
    train_dataset = NpyDataset(args, if_train=True, train_list=train_patches)
    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=trivial_batch_collator,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False)
    start_epoch = 0
    # if args.resume is not None:
    #     if os.path.isfile(args.resume):
    #         ## Map model to be loaded to specified single GPU
    #         checkpoint = torch.load(args.resume, map_location=device)
    #         start_epoch = checkpoint["epoch"] + 1
    #         medsam_model.load_state_dict(checkpoint["model"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dice = 0
        train_ce = 0
        for step, patch_infos in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # for i in range(len(batch_inputs)):
            batched_input_high = patch_infos[0]["high"]
            batched_input_low = patch_infos[0]["low"]
            masks = torch.stack([x["masks"] for x in batched_input_high], dim=0)
            slide_name = batched_input_high[0]["slide_name"]

            # image = torch.stack([x["image"] for x in batched_input_high], dim=0)
            # patch = torch.stack([x["image"] for x in batched_input_low], dim=0)
            # boxes = torch.stack([x["boxes"] for x in batched_input_high], dim=0).to(torch.int)  #torch.Size([1, 1, 4])
            patch_masks = torch.stack([x["masks"] for x in batched_input_low], dim=0)
            # boxes_low = torch.stack([x["boxes"] for x in batched_input_low], dim=0).to(torch.int)

            with torch.no_grad():
                batched_output_high, interm_embeddings_high = sam_model(batched_input_high, multimask_output=False)
                batched_output_low, interm_embeddings_low = sam_model(batched_input_low, multimask_output=False)

            
            batch_len_high = len(batched_output_high)
            encoder_embedding_high = torch.cat([batched_output_high[i_l]['encoder_embedding'] for i_l in range(batch_len_high)], dim=0)
            image_pe_high = [batched_output_high[i_l]['image_pe'] for i_l in range(batch_len_high)]
            sparse_embeddings_high = [batched_output_high[i_l]['sparse_embeddings'] for i_l in range(batch_len_high)]
            dense_embeddings_high = [batched_output_high[i_l]['dense_embeddings'] for i_l in range(batch_len_high)]

            batch_len_low = len(batched_output_low)
            encoder_embedding_low = torch.cat([batched_output_low[i_l]['encoder_embedding'] for i_l in range(batch_len_low)], dim=0)
            image_pe_low = [batched_output_low[i_l]['image_pe'] for i_l in range(batch_len_low)]
            sparse_embeddings_low = [batched_output_low[i_l]['sparse_embeddings'] for i_l in range(batch_len_low)]
            dense_embeddings_low = [batched_output_low[i_l]['dense_embeddings'] for i_l in range(batch_len_low)]
            

            _, _, masks_hq_token_high, hq_image_embeddings_high = net(
                image_embeddings=encoder_embedding_high,
                image_pe=image_pe_high,
                sparse_prompt_embeddings=sparse_embeddings_high,
                dense_prompt_embeddings=dense_embeddings_high,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings_high,
            )
            _, _, masks_hq_token_low, hq_image_embeddings_low = net1(
                image_embeddings=encoder_embedding_low,
                image_pe=image_pe_low,
                sparse_prompt_embeddings=sparse_embeddings_low,
                dense_prompt_embeddings=dense_embeddings_low,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings_low,
            )

            masks_hq_token = masks_hq_token_high + masks_hq_token_low  #torch.Size([10, 1, 32])
            masks_hq_high = (masks_hq_token @ hq_image_embeddings_high).view(1, -1, 256, 256)
            masks_hq_low = (masks_hq_token @ hq_image_embeddings_low).view(1, -1, 256, 256)

            masks_hq_high = F.interpolate(
            masks_hq_high,
            (1024,1024),
            mode="bilinear",
            align_corners=False,
            )

            masks_hq_low = F.interpolate(
            masks_hq_low,
            (1024,1024),
            mode="bilinear",
            align_corners=False,
            )

            if args.visualize == True:
                count=0
                for idx in range(masks_hq_high.shape[0]):
                    medsam_p = (masks_hq_high[idx] > 0.5)
                    medsam_p = torch.as_tensor(medsam_p, dtype=torch.uint8)                           
                    fig, ax = plt.subplots(1, 2, figsize=(25, 25))
                    ax[0].imshow(image[idx].permute(1,2,0).cpu().numpy())
                    show_mask(masks[idx].cpu().numpy(), ax[0])
                    show_box(boxes[idx,0,...].cpu().numpy(), ax[0])
                    ax[0].set_title("Input Image and Bounding Box")
                    ax[1].imshow(patch[idx].permute(1,2,0).cpu().numpy())
                    show_mask(patch_masks[idx].cpu().detach().numpy(), ax[1])
                    show_box(boxes_low[idx,0,...].cpu().numpy(), ax[1])
                    ax[1].set_title("MedSAM Segmentation")
                    # plt.show()
                    plt.subplots_adjust(wspace=0.01, hspace=0)
                    plt.savefig('/home/bme001/20225531/sam-hq/results/check_patch/{}_{}.png'.format(slide_name, count), bbox_inches="tight", dpi=300)
                    count+=1
            
            dice_loss_high = seg_loss(masks_hq_high, masks)
            cross_loss_high = ce_loss(masks_hq_high, masks.float())
            dice_loss_low = seg_loss(masks_hq_low, patch_masks)
            cross_loss_low = ce_loss(masks_hq_low, patch_masks.float())
            dice_loss = dice_loss_high + dice_loss_low
            cross_loss = cross_loss_high + cross_loss_low
            loss = dice_loss + cross_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            train_dice += dice_loss.item()
            train_ce += cross_loss.item()
            iter_num += 1
        epoch_loss /= step
        train_dice /= step
        train_ce /= step
        losses.append(epoch_loss)
        train_losses_dice.append(train_dice)
        train_losses_ce.append(train_ce)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # validation after each epoch
        # if epoch % 10 == 0 and epoch > 0:
        #     val_dice, val_ce, val_epoch_loss = validate(args, sam_model, net, net1)
        #     val_losses.append(val_epoch_loss)
        #     val_losses_dice.append(val_dice)
        #     val_losses_ce.append(val_ce)
            ## save the best model
            # if val_epoch_loss < best_loss:
            #     best_loss = val_epoch_loss
            #     checkpoint = {
            #         "model": net.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "epoch": epoch}
            #     torch.save(checkpoint, join(model_save_path, "medsam_hq_model_best.pth"))
        ## save the latest model
        checkpoint = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}
        torch.save(checkpoint, join(model_save_path, "medsam_hq_model_latest_net.pth"))
        checkpoint1 = {
            "model": net1.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch}
        torch.save(checkpoint1, join(model_save_path, "medsam_hq_model_latest_net1.pth"))
        # ## save the best model
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     checkpoint = {
        #         "model": net.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch}
        #     torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))     
        if epoch % 10 == 0:
            torch.save(checkpoint, join(model_save_path, "medsam_hq_model_net_{}.pth".format(epoch)))
        if epoch % 10 == 0:
            torch.save(checkpoint1, join(model_save_path, "medsam_hq_model_net1_{}.pth".format(epoch)))

        # %% plot loss
        plt.plot(losses)
        plt.title("Training Dice & Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

        plt.plot(train_losses_dice)
        plt.title("Training Dice Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_dice_loss.png"))
        plt.close()

        plt.plot(train_losses_ce)
        plt.title("Training Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_ce_loss.png"))
        plt.close()

        plt.plot(val_losses)
        plt.title("Validation Dice & Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "val_loss.png"))
        plt.close()

        plt.plot(val_losses_dice)
        plt.title("Validation Dice Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "val_dice_loss.png"))
        plt.close()

        plt.plot(val_losses_ce)
        plt.title("Validation Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "val_ce_loss.png"))
        plt.close()

if __name__ == "__main__":
    main()