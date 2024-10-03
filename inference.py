import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything_training import sam_model_registry
from network import MaskDecoderHigh, MaskDecoderLow
import torch.nn.functional as F
import os
from typing import Tuple

def parse_args():
    """
    Parses command-line arguments to retrieve image paths and bounding box input.
    """
    parser = argparse.ArgumentParser(description="Run inference with high-res and low-res images with a bounding box.")
    parser.add_argument("--high_res_image", type=str, default='examples/high.png', help="Path to the high-resolution image.")
    parser.add_argument("--low_res_image", type=str, default='examples/low.png', help="Path to the low-resolution image.")
    parser.add_argument("--bounding_box", type=str, default=[0, 0, 964, 697], help="Bounding box coordinates in the format 'xmin,ymin,xmax,ymax'.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model (default: cuda:0)")
    parser.add_argument("--visual", type=str, default=True, help="visualize the output")
    parser.add_argument("--model_type", type=str, default="vit_tiny", help="Model type for SAM")
    parser.add_argument("--checkpoint_SAM", type=str, default='pretrained_checkpoint/mobile_sam.pt', help="Path to the trained model checkpoint")
<<<<<<< HEAD
    parser.add_argument("--checkpoint_net_high", type=str, default='pretrained_checkpoint/net_high.pth', help="Path to net")
    parser.add_argument("--checkpoint_net_low", type=str, default='pretrained_checkpoint/net_low.pth', help="Path to net1")
=======
    parser.add_argument("--checkpoint_net", type=str, default='pretrained_checkpoint/net.pth', help="Path to net")
    parser.add_argument("--checkpoint_net1", type=str, default='pretrained_checkpoint/net1.pth', help="Path to net1")
>>>>>>> 6680bfd787b6bd02c3d671c656b5b844624af575
    args = parser.parse_args()
    return args

        

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([255, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def load_image(image_path: str) -> torch.Tensor:
    """
    Loads an image from a file path and returns it as a torch.Tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Loaded image as a (C, H, W) tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image_tensor = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)  # Convert to (C, H, W)
    return image_tensor

def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:

    masks = F.interpolate(
        masks,
        (1024,1024),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


def run_inference(high_res_image: torch.Tensor, low_res_image: torch.Tensor, box: list, args):
    """
    Runs inference on the provided high and low-resolution images with a bounding box.

    Args:
        high_res_image (torch.Tensor): High-resolution image tensor (C, H, W)
        low_res_image (torch.Tensor): Low-resolution image tensor (C, H, W)
        box (list): Bounding box [x_min, y_min, x_max, y_max]
        args: Additional arguments for model loading.
    """

    # Load model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint_SAM).to(args.device).eval()
    net_high = MaskDecoderHigh(args.model_type).to(args.device).eval()
    net_low = MaskDecoderLow(args.model_type).to(args.device).eval()
    
    # Load checkpoints
<<<<<<< HEAD
    net_ckpt_high = args.checkpoint_net_high
    checkpoint_high = torch.load(net_ckpt_high, map_location='cpu')['model']
    net_high.load_state_dict(checkpoint_high, strict=False)
    
    net_ckpt_low = args.checkpoint_net_low
    checkpoint_low = torch.load(net_ckpt_low, map_location='cpu')['model']
    net_low.load_state_dict(checkpoint_low, strict=False)
=======
    net_ckpt = args.checkpoint_net
    checkpoint = torch.load(net_ckpt, map_location='cpu')['model']
    net_high.load_state_dict(checkpoint, strict=False)
    
    net1_ckpt = args.checkpoint_net1
    checkpoint1 = torch.load(net1_ckpt, map_location='cpu')['model']
    net_low.load_state_dict(checkpoint1, strict=False)
>>>>>>> 6680bfd787b6bd02c3d671c656b5b844624af575
    
    # Convert bounding box to tensor
    box_tensor = torch.tensor(box, device=args.device).unsqueeze(0).to(args.device)



    # Prepare input for high-res and low-res images
    dict_input_high = {
        'image': high_res_image.to(args.device),
        'boxes': box_tensor   # [x_min, y_min, x_max, y_max]
    }

    dict_input_low = {
        'image': low_res_image.to(args.device),
        'boxes': box_tensor // 2 + 256 # Scale down the bounding box for low-res image
    }

    # Run inference using the SAM model
    with torch.no_grad():
        batched_output_high, interm_embeddings_high = sam_model([dict_input_high], multimask_output=False)
        batched_output_low, interm_embeddings_low = sam_model([dict_input_low], multimask_output=False)

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

        masks_sam_high, _, masks_wsi_token_high, wsi_image_embeddings_high = net_high(
            image_embeddings=encoder_embedding_high,
            image_pe=image_pe_high,
            sparse_prompt_embeddings=sparse_embeddings_high,
            dense_prompt_embeddings=dense_embeddings_high,
            multimask_output=False,
            wsi_token_only=False,
            interm_embeddings=interm_embeddings_high,
        )
        _, _, masks_wsi_token_low, _ = net_low(
            image_embeddings=encoder_embedding_low,
            image_pe=image_pe_low,
            sparse_prompt_embeddings=sparse_embeddings_low,
            dense_prompt_embeddings=dense_embeddings_low,
            multimask_output=False,
            wsi_token_only=False,
            interm_embeddings=interm_embeddings_low,
        )

        masks_wsi_token = masks_wsi_token_high + masks_wsi_token_low  
        masks_wsi_high = (masks_wsi_token @ wsi_image_embeddings_high).view(1, -1, 256, 256)
        # masks_wsi_low = (masks_wsi_token @ wsi_image_embeddings_low).view(1, -1, 256, 256)

        masks_wsi_high = masks_wsi_high + masks_sam_high
        masks_wsi_high = postprocess_masks(
                masks_wsi_high,
                input_size=high_res_image.shape[-2:],
                original_size=(1024,1024),
            )


    # Return final mask prediction
    return masks_wsi_high

def main():
    # Parse arguments from the command line
    args = parse_args()

    # Parse bounding box input
    bounding_box = args.bounding_box
    bounding_box_low = torch.tensor(bounding_box) // 2 + 256

    # Load high-resolution and low-resolution images from the given paths
    high_res_image = load_image(args.high_res_image)  # (C, H, W)
    low_res_image = load_image(args.low_res_image)  # (C, H, W)

    # Run inference
    mask_prediction = run_inference(high_res_image, low_res_image, bounding_box, args)


    if args.visual == True:
        fig, ax = plt.subplots(1, 3, figsize=(25, 25))
        ax[0].imshow(high_res_image.permute(1,2,0).cpu().numpy())
        show_box(bounding_box, ax[0])
        ax[0].set_title("Input High-res Image with Bounding Box")
        ax[1].imshow(low_res_image.permute(1,2,0).cpu().numpy())
        show_box(bounding_box_low, ax[1])
        ax[1].set_title("Input Low-res Image with Bounding Box")
        ax[2].imshow(high_res_image.permute(1,2,0).cpu().numpy())
        show_mask(mask_prediction[0].cpu().numpy(), ax[2])
        show_box(bounding_box, ax[2])
        ax[2].set_title("High-res Image with Prediction")
        # plt.show()
        # plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig('examples/high_pred.png', bbox_inches="tight")

if __name__ == "__main__":
    main()
    print('Finished!')
