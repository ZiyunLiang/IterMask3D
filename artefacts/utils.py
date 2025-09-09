import numpy as np
import skimage
from numpy.fft import ifft2, ifftshift
from skimage.filters import unsharp_mask
from skimage.exposure import rescale_intensity


def rigid_warp(img, ang_rot, tr_x, tr_y):
    '''
    Applies a rotation of `ang_rot` degrees to the centre of the image, and
    translate the image by [tr_x, tr_y] pixels.
    '''
    shift_y, shift_x = np.array(img.shape) / 2.
    transform = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y]) + \
                skimage.transform.SimilarityTransform(rotation=np.deg2rad(ang_rot)) + \
                skimage.transform.SimilarityTransform(translation=[shift_x - tr_x, shift_y - tr_y])
    return rescale_intensity(skimage.transform.warp(img, transform), out_range=(img.min(), img.max()))
    

def reconstruct_from_kspace(k_space):
    '''
    Reconstruct the image with the artifact from the k-space.
    '''
    recon_image = ifft2(ifftshift(k_space))
    return recon_image.real        


def sharpen(im):
    ''' 
    Applies shaperning to img 
    '''
    return unsharp_mask(im, radius=1, amount=1, preserve_range=1)


def powerlaw(a, b, g, size=1):
    '''
    Power-law gen for pdf(x)/propto x^{g-1} for a<=x<=b
    '''
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return 1-(ag + (bg - ag)*r)**(1./g)


def pad_to_shape(image, target_shape):
    '''
    Pad an image to a target shape
    '''
    h, w = np.shape(image)
    dest_h, dest_w = target_shape
    output_shape_placeholder = np.zeros(target_shape)
    top_left_x = int(np.round((dest_h - h) / 2))
    top_left_y = int(np.round((dest_w - w) / 2))
    output_shape_placeholder[top_left_x:(top_left_x + h), top_left_y:(top_left_y + w)] = image.copy()
    return output_shape_placeholder


def complex_rand(shape, rrange, irange):
    '''
    Generate a random complex vector of shape `shape`. Parameters `rrange`
     and `irange` represents the range of real and imaginary values respectively.
    '''
    min_value_r, max_value_r = rrange
    min_value_i, max_value_i = irange
    data = np.zeros(shape, dtype=complex)
    data.real = (np.random.rand(*shape) * (max_value_r - min_value_r) + min_value_r)
    data.imag = (np.random.rand(*shape) * (max_value_i - min_value_i) + min_value_i)
    return data

def complex_rand_from_reference(shape, reference):
    '''
    Same as complex_rand, but uses a reference k-space to 
    get the ranges.
    '''
    rrange = (reference.real.min(), reference.real.max())
    irange = (reference.imag.min(), reference.imag.max())
    return complex_rand(shape, rrange, irange)


def complex_uniform_ep(shape, k_space, error_percentage):
    '''
    Generates a complex matrix of uniform elements between [-eps, eps]
    '''
    eps_r = np.max(abs(k_space.real)) * error_percentage
    eps_i = np.max(abs(k_space.imag)) * error_percentage
    data = np.zeros(shape, dtype=complex)
    data.real = np.random.uniform(0 - eps_r, 0 + eps_r, shape)
    data.imag = np.random.uniform(0 - eps_i, 0 + eps_i, shape)
    return data