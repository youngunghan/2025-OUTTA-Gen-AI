import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random


def init_linear(linear):
    """
    Initialize a linear layer with Xavier normal initialization.
    Args:
        linear (nn.Linear): Linear layer to initialize.
    Example:
        >>> layer = nn.Linear(128, 256)
        >>> init_linear(layer)
    """
    init.xavier_normal(linear.weight) # Xavier initialization
    linear.bias.data.zero_() # Set bias to 0


def init_conv(conv, glu=True):
    """
    Initialize a convolutional layer with Kaiming normal initialization.
    Args:
        conv (nn.Conv2d): Convolutional layer to initialize.
        glu (bool, optional): Whether to use GLU activation. Defaults to True.
    Example:
        >>> layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> init_conv(layer)
    """
    init.kaiming_normal(conv.weight) # Kaiming initialization
    if conv.bias is not None:
        conv.bias.data.zero_() # Set bias to 0


class EqualLR:
    """
    Apply Equalized Learning Rate to a module's weights.
    This ensures stable training by normalizing the learning rate.
    """
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    """
    Wrapper to apply EqualLR to a given module.
    Args:
        module (nn.Module): Module to apply EqualLR to.
        name (str, optional): Name of the parameter to modify. Defaults to 'weight'.
    Example:
        >>> layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> layer = equal_lr(layer)
    """
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    """
    Custom upsampling layer using transposed convolution with fused weight modification.
    This operation smoothens the upsampling process to reduce aliasing effects.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        padding (int, optional): Padding size. Default is 0.

    Example:
        >>> upsample = FusedUpsample(64, 128, 3)
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> output = upsample(input_tensor)

    Return:
        Tensor: Upsampled feature map.
    """
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        # Initialize weights and bias
        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        # Calculate normalization factor for weight scaling
        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        # Apply padding to the weight and smooth it to reduce aliasing
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        # Perform transposed convolution (upsampling)
        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    """
    Downsamples an image using a learned convolution filter.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        padding (int, optional): Padding for the convolution. Default is 0.
    
    Example:
        downsample = FusedDownsample(3, 3, 3)
        output = downsample(input_tensor)
    
    Return:
        torch.Tensor: Downsampled image tensor.
    """

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        # Initialize weights and biases
        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        # Calculate normalization factor for weight scaling
        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        # Apply padding to weight and smooth it to reduce aliasing
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        # Perform standard convolution (downsampling)
        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    """
    Pixelwise normalization layer to stabilize training by normalizing feature maps.
    
    Example:
        pixel_norm = PixelNorm()
        output = pixel_norm(input_tensor)
    
    Return:
        torch.Tensor: Normalized tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Normalize each pixel feature vector to have unit variance
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    """
    Backward function for blur operation to reduce aliasing.
    
    Example:
        grad_output = torch.randn(1, 3, 64, 64)
        kernel = torch.randn(1, 1, 3, 3)
        kernel_flip = torch.flip(kernel, [2, 3])
        output = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)
    
    Return:
        torch.Tensor: Gradient after blur operation.
    """

    # Blur를 통해 aliasing을 많이 줄일 수 있다. 
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        # Apply 2D convolution to blur the gradient
        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        # Apply 2D convolution again in the backward pass
        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    """
    Applies a blur filter to the input image.
    
    Example:
        input_tensor = torch.randn(1, 3, 64, 64)
        kernel = torch.randn(1, 1, 3, 3)
        kernel_flip = torch.flip(kernel, [2, 3])
        output = BlurFunction.apply(input_tensor, kernel, kernel_flip)
    
    Return:
        torch.Tensor: Blurred image tensor.
    """

    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        # Apply 2D convolution to blur the input
        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        # Apply custom backward pass for blur
        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    """
    Blur layer to apply Gaussian-like smoothing filter.
    
    Args:
        channel (int): Number of input channels.
    
    Example:
        blur = Blur(3)
        output = blur(input_tensor)
    
    Return:
        torch.Tensor: Blurred image tensor.
    """

    def __init__(self, channel):
        super().__init__()

        # Define a simple 3x3 Gaussian-like kernel
        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        # Store the weights as buffers
        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        # Apply the blur function
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    """
    Convolutional layer with equalized learning rate.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    """
    Linear layer with equalized learning rate.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    """
    A convolutional block used in the StyleGAN architecture.

    This block applies two convolutional layers, each followed by a LeakyReLU activation.
    If downsampling is required, it applies either an average pooling layer or a fused downsample operation.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size for the first convolutional layer.
        padding (int): Padding for the first convolutional layer.
        kernel_size2 (int, optional): Kernel size for the second convolutional layer.
        padding2 (int, optional): Padding for the second convolutional layer. 
        downsample (bool, optional): Wheter to downsample the input. Defaults to False.
        fused (bool, optional): Wheter to use fused downsampling. Defaults to False.

    Example:
        >>> block = ConvBlock(64, 128, 3, 1, downsample=True)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x) # Output shape: (1, 128, 16, 16)

    Returns:
        torch.Tensor: Transformed feature map.    
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        # First convolutional layer with Equalized learning rate and LeakyReLU activation.
        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        # Second convolutional layer with downsampling if required
        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        """
        Forward pass for ConvBlock

        Args:
            input (torch.Tensor): Input feature map.
        
        Returns:
            torch.Tensor: Transformed feature map.
        """
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.

    This normalization layer adjusts the feature statistics using a style code.

    Args:
        in_channel (int): Number of input channels.
        style_dim (int): Dimension of the style embedding.

    Example:
        >>> adain = AdaptiveInstanceNorm(64, 512)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> style = torch.randn(1, 512)
        >>> output = adain(x, style)  # Output shape: (1, 64, 32, 32)

    Returns:
        torch.Tensor: Normalized feature map with applied style transformation.
    """
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel) # Instance Normalization
        self.style = EqualLinear(style_dim, in_channel * 2) # Style embedding transformation

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        """
        Forward pass for Adaptive Instance Normalization.

        Args:
            input (torch.Tensor): Feature map to be normalized.
            style (torch.Tensor): Style vector.

        Returns:
            torch.Tensor: Feature map transformed based on the style vector.
        """
        style = self.style(style).unsqueeze(2).unsqueeze(3) # Reshape style vector
        gamma, beta = style.chunk(2, 1) # Split into scale and shift

        out = self.norm(input) # Apply instance normalization
        out = gamma * out + beta # Apply style transformation

        return out


class NoiseInjection(nn.Module):
    """
    Noise Injection layer for adding stochastic variation to feature maps.

    Args:
        channel (int): Number of channels in the feature map.

    Example:
        >>> noise_injection = NoiseInjection(64)
        >>> image = torch.randn(1, 64, 32, 32)
        >>> noise = torch.randn(1, 1, 32, 32)
        >>> output = noise_injection(image, noise)  # Output shape: (1, 64, 32, 32)

    Returns:
        torch.Tensor: Feature map with added noise.
    """

    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        """
        Forward pass for NoiseInjection.

        Args:
            image (torch.Tensor): Input feature map.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Noisy feature map.
        """
        return image + self.weight * noise


class ConstantInput(nn.Module):
    """
    Generates a constant learned tensor as the initial input for the StyleGAN generator.

    Args:
        channel (int): Number of output channels.
        size (int, optional): Initial tensor size. Defaults to 4.

    Example:
        >>> const_input = ConstantInput(512)
        >>> batch_size = 4
        >>> x = torch.randn(batch_size, 1)
        >>> output = const_input(x)  # Output shape: (4, 512, 4, 4)

    Returns:
        torch.Tensor: Constant input tensor with batch size.
    """

    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        """
        Forward pass for ConstantInput.

        Args:
            input (torch.Tensor): Input tensor used to determine batch size.

        Returns:
            torch.Tensor: Constant input tensor with adjusted batch size.
        """
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    """
    A convolutional block used in StyleGAN with adaptive instance normalization and noise injection.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int, optional): Kernel size of the convolution. Default=3.
        padding (int, optional): Padding size for convolution. Default=1.
        style_dim (int, optional): Dimension of the style vector. Default = 512.
        initial (bool, optional): Whether this is the first block.
        upsample (bool, optional): Whether to apply upsampling befor convolution.
        fused (bool, optional): Whether to use fused upsampling.

    Example:
        >>> styled_conv = StyledConvBlock(512, 256, upsample=True)
        >>> x = torch.randn(1, 512, 8, 8)
        >>> style = torch.randn(1, 512)
        >>> noise = torch.randn(1, 1, 16, 16)
        >>> output = styled_conv(x, style, noise) # output shape: (1, 256, 16, 16)

    Returns:
        torch.Tensor: Feature map after applying convolution, noise, and normalization.   
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            # If this is the first block, use a learned constant input
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    # Fused upsampling combines upsampling and convolution
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    # Separate upsampling followed by convolution.
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                # Standard convolution without upsampling
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )
        
        # Noise injection, AdaIN, and activation function for the first convolution
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        # Second convolution with noise and AdaIN
        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        """
        Forward pass of StyledConvBlock.

        Args:
            input (torch.Tensor): Input feature map.
            style (torch.Tensor): Style vector for normalization.
            noise (torch.Tensor): Noise tensor for stochastic variation.

        Returns:
            torch.Tensor: Transformed feature map.
        """

        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    """
    The Generator class for StyleGAN.

    This model progressively grows from a 4x4 image to a 1024x1024 image using a series of `StyledConvBlock`.

    Args:
        code_dim (int): Dimension of the latent code.
        fused (bool, optional): Whether to use fused upsampling in later layers. Defaults to True.

    Example:
        >>> generator = Generator(512)
        >>> style = [torch.randn(1, 512) for _ in range(9)]
        >>> noise = [torch.randn(1, 1, 2**(2+i), 2**(2+i)) for i in range(9)]
        >>> output = generator(style, noise, step=6, alpha=0.5)  # Output image at 128x128

    Returns:
        torch.Tensor: Synthesized image at the specified resolution.
    """

    def __init__(self, code_dim, fused=True):
        super().__init__()

        # Progressive layers from 4x4 to 1024x1024
        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4x4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8x8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16x16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32x32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64x64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128x128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),  # 256x256
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512x512
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024x1024
            ]
        )

        # RGB output layers for each resolution
        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        """
        Forward pass of the Generator.

        Args:
            style (list of torch.Tensor): List of style vectors for different resolutions.
            noise (list of torch.Tensor): List of noise tensors for different resolutions.
            step (int, optional): Current resolution step in progressive training. Defaults to 0.
            alpha (float, optional): Blending factor for smooth transition between resolutions. Defaults to -1.
            mixing_range (tuple, optional): Defines range of layers affected by style mixing. Defaults to (-1, -1).

        Returns:
            torch.Tensor: Generated image at the specified resolution.
        """

        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))
        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]
            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]
                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out  
            out = conv(out, style_step, noise[i])
            if i == step:
                out = to_rgb(out)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out
                break
        return out


class StyledGenerator(nn.Module):
    """
    StyledGenerator is a StyleGAN generator that takes a latent vector and 
    applies multiple transformations to control the style of the generated image.

    Args:
        code_dim (int): Dimension of the latent vector. Default = 512
        n_mlp (int): Number of MLP layers for the style transformation. Default = 8

    Example:
        >>> generator = StyledGenerator()
        >>> latent_vector = torch.randn(1, 512)
        >>> noise = [torch.randn(1, 1, 4, 4)]
        >>> output = generator(latent_vector, noise, step=3, alpha=0.5)
        >>> print(output.shape) # Output: torch.Size([1, 3, 32, 32])

    Result:
        Generates an image from a latent vector with progressive growing and adaptive instance normalization.
    """

    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        # The main generator that synthesizes the image
        self.generator = Generator(code_dim)

        # The style mapping network consists of multiple MLP layers.
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        # Style transformation pipeline
        self.style = nn.Sequential(*layers)

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):

        """
        Forward pass for the generator.

        Args:
            input (Tensor or list of Tensor): Latent vector(s).
            noise (list of Tensor, optional): List of noise tensors per layer.
            step (int, optional): Current progression step (default: 0).
            alpha (float, optional): Alpha blending factor for resolution transition (default: -1).
            mean_style (Tensor, optional): Mean style for normalization.
            style_weight (float, optional): Weight for mixing the mean style.
            mixing_range (tuple, optional): Range of layers for style mixing.

        Returns:
            Tensor: Generated image.

        Example:
            >>> generator = StyledGenerator()
            >>> latent_vector = torch.randn(1, 512)
            >>> output = generator(latent_vector, step=4, alpha=0.7)
            >>> print(output.shape)  # Output: torch.Size([1, 3, 64, 64])
        """

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        # Apply style transformation to each input latent vector
        for i in input:
            styles.append(self.style(i))
        batch = input[0].shape[0]

        # If noise is not provided, generate random noise for each step
        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        # Apply mean style normalization if provided
        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm

        # Pass transformed styles and noise to the generator
        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        """
        Computes the mean style vector from the given latent vectors.

        Args:
            input (Tensor): Latent vectors of shape (batch, code_dim).

        Returns:
            Tensor: Mean style vector of shape (1, code_dim).

        Example:
            >>> generator = StyledGenerator()
            >>> latent_vector = torch.randn(10, 512)
            >>> mean_style = generator.mean_style(latent_vector)
            >>> print(mean_style.shape)  # Output: torch.Size([1, 512])
        """

        style = self.style(input).mean(0, keepdim=True)
        return style


class Discriminator(nn.Module):
    """
    Discriminator for StyleGAN. This network progressively downsamples 
    an image while applying convolutional blocks to extract high-level features.

    Args:
        fused (bool, optional): Whether to use fused downsampling (default: True).
        from_rgb_activate (bool, optional): Whether to apply activation on RGB input (default: False).

    Example:
        >>> discriminator = Discriminator()
        >>> fake_image = torch.randn(1, 3, 128, 128)
        >>> output = discriminator(fake_image, step=4, alpha=0.7)
        >>> print(output.shape)  # Output: torch.Size([1, 1])

    Result:
        - Outputs a scalar score indicating the realism of the input image.
    """

    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        # Convolutional blocks for progressive downsampling
        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512x512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256x256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128x128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64x64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32x32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16x16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8x8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4x4
                ConvBlock(513, 512, 3, 1, 4, 0), 
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))
            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()
        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        """
        Forward pass of the discriminator.

        Args:
            input (Tensor): Input image tensor.
            step (int, optional): Current resolution step.
            alpha (float, optional): Alpha blending factor for progressive growing.

        Returns:
            Tensor: Real vs Fake score.

        Example:
            >>> discriminator = Discriminator()
            >>> fake_image = torch.randn(1, 3, 128, 128)
            >>> output = discriminator(fake_image, step=4, alpha=0.5)
            >>> print(output.shape)  # Output: torch.Size([1, 1])
        """

        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out