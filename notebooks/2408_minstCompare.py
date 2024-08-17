"""Demo: remove high-freq infos from MNIST.
"""
# %%
import jax
import numpy as np
import jax.numpy as jnp
import functools as ft

import matplotlib.pyplot as plt

# jax so I can vmap, import ffts from jax or expect errors:
# https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerArrayConversionError
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift
from jax import random

from tk.utils.data.fetch import load_mnist_gzip


def remove_high_freq(image: jnp.ndarray, cutoff_ratio=0.2):
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    h, w = image.shape
    cx, cy = w // 2, h // 2
    cutoff_x = int(cx * cutoff_ratio)
    cutoff_y = int(cy * cutoff_ratio)
    mask = jnp.zeros_like(image)
    mask = mask.at[cy-cutoff_y:cy+cutoff_y, cx-cutoff_x:cx+cutoff_x].set(1)
    f_shifted *= mask
    return jnp.abs(ifft2(ifftshift(f_shifted)))


train = load_mnist_gzip('train')
sample_image = jnp.array(train.x[0])
filtered_image = remove_high_freq(sample_image)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(sample_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Filtered Image")
plt.imshow(filtered_image, cmap='gray')
plt.show()
# %%
from collections import defaultdict

l2img = defaultdict(list)
for img, lbl in zip(train.x, train.y):
    l2img[lbl].append(img)

l2img = {k: np.stack(v) for k, v in l2img.items()}
print({k: v.shape for k, v in l2img.items()})

# %% "unit test" lol
def squared_error_matrix(A, B):
    diff = A[:, None, :, :] - B[None, :, :, :]
    return np.sum(diff**2, axis=(2, 3))


A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
B = np.array([[[1, 2], [3, 4]], [[9, 10], [11, 12]]])
result = squared_error_matrix(A, B)
print(result)

# %%
def sample(
    i1: jnp.ndarray, 
    rng: random.PRNGKey,
    k: int = 100
):
    ix1 = random.randint(rng, minval=0, maxval=len(i1), shape=k)
    return i1[ix1]


def compare(
    i1: np.ndarray, 
    i2: np.ndarray, 
):
    def squared_error_matrix(A, B):
        """diff cross-product of all elems of A and B."""
        diff = A[:, None, :, :] - B[None, :, :, :]
        return jnp.sum(diff**2, axis=(2, 3))

    distances = squared_error_matrix(i1, i2)
    return distances


rng = jax.random.PRNGKey(42)
rng, sampler_key = jax.random.split(rng)
sampler = ft.partial(sample, rng=sampler_key, k=100)
i0, i1 = map(sampler, (l2img[0], l2img[1]))
dists_before = compare(i0, i1)
print('Before', dists_before.mean(), dists_before.std())

lofreq0 = jax.vmap(remove_high_freq)(jnp.array(i0))
lofreq1 = jax.vmap(remove_high_freq)(jnp.array(i1))
dists_after = compare(lofreq0, lofreq1)
print('After', dists_after.mean(), dists_after.std())

# %%
dists = compare(i0, lofreq0)
print('Within 0 ::', dists.mean(), dists.std())
dists = compare(i1, lofreq1)
print('Within 1 ::', dists.mean(), dists.std())
# %%
