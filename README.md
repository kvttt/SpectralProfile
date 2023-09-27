Spectral Profile
============

This little script can be easily used on your PyTorch project to visualize 
the spectral profile of a generated (or any) image (either 2D or 3D).

The program is an unofficial PyTorch implementation of 
[UpConv](https://github.com/cc-hpc-itwm/UpConv) which is originally written using Numpy.
In addition to the original implementation, this program can also be used on 3D images, 
e.g. MRI or CT volumes.

Also see [frequency_bias](https://github.com/autonomousvision/frequency_bias) which is another 
PyTorch implementation but with only 2D-support and with obsolete PyTorch syntax.

Dependencies
------------
* PyTorch
* PIL (only for the example data)
* Numpy (only for the example data)
* Matplotlib

Usage
-----
To get the spectral profile of an image loaded as a tensor `img`, 
simply call the function `get_spectrum` as follows:

    psd1D = get_spectrum(img)

which returns a 1D power spectral density (PSD) of the image.




