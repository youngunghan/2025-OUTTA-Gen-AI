import argparse # Command-line argument parsing
import math # Mathematical operations

import torch
from torchvision import utils

from model import StyledGenerator 


@torch.no_grad()
def get_mean_style(generator, device):
    """
    Computes the average latent style vector by sampling random latent vectors.

    Args:
        generator (StyledGenerator): The generator model.
        device (str): The device to run the computations on (CPU or CUDA).

    Example:
        mean_style = get_mean_style(generator, 'cuda')

    Returns:
        torch.Tensor: The mean style vector.
    """

    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10 # Compute mean style vector
    return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    """
    Generates a batch of images from random latent vectors.

    Args:
        generator (StyledGenerator): The generator model.
        step (int): Current resolution step.
        mean_style (torch.Tensor): The mean style vector.
        n_sample (int): Number of images to generate.
        device (str): The device to run computations on.

    Example:
        images = sample(generator, step=6, mean_style, 25, 'cuda')

    Returns:
        torch.Tensor: A batch of generated images.
    """

    image = generator(
        torch.randn(n_sample, 512).to(device), # Random latent vectors
        step=step,
        alpha=1, # Blending factor for progressive growing
        mean_style=mean_style,
        style_weight=0.7, # Strength of the mean style influence
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    """
    Performs style mixing between source and target latent codes.

    Args:
        generator (StyledGenerator): The generator model.
        step (int): Current resolution step.
        mean_style (torch.Tensor): The mean style vector.
        n_source (int): Number of source images.
        n_target (int): Number of target images.
        device (str): The device to run computations on.

    Example:
        mixed_images = style_mixing(generator, step=6, mean_style, 5, 3, 'cuda')

    Returns:
        torch.Tensor: A batch of images showing style mixing.
    """

    source_code = torch.randn(n_source, 512).to(device) # Source latent code
    target_code = torch.randn(n_target, 512).to(device) # Target latent code
    
    shape = 4 * 2 ** step # Image resolution (4x4 -> 8x8 -> ... -> 1024x1024)
    alpha = 1 # Bleding factor for progressive growing

    images = [torch.ones(1, 3, shape, shape).to(device) * -1] # Placeholder for visualization

    # Generate source images
    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    # Generate target images
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image) # Append source images

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1), # Apply style mixing
        )
        images.append(target_image[i].unsqueeze(0)) # Append target image
        images.append(image) # Append mixed images

    images = torch.cat(images, 0) # Concatenate images into a batch
    
    return images


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()
    
    # Determine whether to use CUDA or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the generator model
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path, map_location=torch.device('cpu'))['g_running'])
    generator.eval()

    # Compute mean style vector
    mean_style = get_mean_style(generator, device)

    # Determine the step for progressive growing based on image size.
    step = int(math.log(args.size, 2)) - 2
    
    # Generate and save a sample batch of images
    img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
    utils.save_image(img, 'sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    # Generate and save 20 style-mixed images
    for j in range(20):
        img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
        utils.save_image(
            img, f'sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        )