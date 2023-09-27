import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.fft import fftn, fftshift
from torch.linalg import vector_norm


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    :param image: Input image (can be 2D or 3D)
    :param center: The pixel coordinates used as the center. The default is
                None, which then uses the center of the image (including
                fractional pixels).
    :return: The azimuthally averaged radial profile.

    """
    dim = len(image.shape)

    # Calculate the indices from the image
    if dim == 2:
        y, x = torch.meshgrid(torch.arange(image.shape[0]),
                              torch.arange(image.shape[1]),
                              indexing='ij')
        if not center:
            center = torch.tensor([(x.max() - x.min()) / 2.0,
                                   (y.max() - y.min()) / 2.0])
        r = vector_norm(
            torch.stack(
                [x - center[0], y - center[1]]
            ),
            ord=2,
            dim=0
        )

    elif dim == 3:
        z, y, x = torch.meshgrid(torch.arange(image.shape[0]),
                                 torch.arange(image.shape[1]),
                                 torch.arange(image.shape[2]),
                                 indexing='ij')
        if not center:
            center = torch.tensor([(x.max() - x.min()) / 2.0,
                                   (y.max() - y.min()) / 2.0,
                                   (z.max() - z.min()) / 2.0])
        r = vector_norm(
            torch.stack(
                [x - center[0], y - center[1], z - center[2]]
            ),
            ord=2,
            dim=0
        )
    else:
        raise ValueError(f'Expecting input to be 2D or 3D. Got {dim}D instead.')

    # Get sorted radii
    r_flat = torch.flatten(r)
    ind = torch.argsort(r_flat)
    r_sorted = r_flat[ind]
    image_flat = torch.flatten(image)
    i_sorted = image_flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = torch.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = torch.cumsum(i_sorted, dim=-1, dtype=torch.float64)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def get_spectrum(img, epsilon=1e-8):
    f = fftn(img)
    f = fftshift(f)
    f += epsilon

    magnitude_spectrum = 20 * torch.log(torch.abs(f))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(magnitude_spectrum)
    print(psd1D.shape)

    return psd1D


if __name__ == '__main__':
    img = Image.open("./cat.jpg").convert('L')
    arr = np.array(img.getdata(), dtype=np.float32)
    arr = arr.reshape(img.size[1], img.size[0])
    arr = arr / 255.0
    arr = torch.tensor(arr)

    psd1D = get_spectrum(arr)

    plt.plot(psd1D)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Log Power Spectral Density')
    plt.title('1D Power Spectrum')
    plt.savefig('./psd1D.png', dpi=300, bbox_inches='tight')
    plt.show()
