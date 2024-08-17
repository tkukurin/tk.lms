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
# %%
from tk.utils.data.fetch import load_mnist_gzip
train = load_mnist_gzip('train')
# %%
def rmfreqs(image: jnp.ndarray, cutoff=0.2, rm='lo'):
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    h, w = image.shape
    cx, cy = w // 2, h // 2
    cutoff_x = int(cx * cutoff)
    cutoff_y = int(cy * cutoff)
    if rm == 'hi':
        setval = 1
        mask = jnp.zeros_like(image)
    else:
        assert rm == 'lo', f'{rm=}'
        setval = 0
        mask = jnp.ones_like(image)
    mask = mask.at[cy-cutoff_y:cy+cutoff_y, cx-cutoff_x:cx+cutoff_x].set(setval)
    f_shifted *= mask
    return jnp.abs(ifft2(ifftshift(f_shifted)))


rm_hifreq = ft.partial(rmfreqs, rm='hi')
rm_lofreq = ft.partial(rmfreqs, rm='lo')


def side_by_side(
    imgs: jnp.ndarray | dict,
    nrows: int | None = None,
    titles: dict = None,
):
    """plot imgs side by side"""
    titles = titles or {}
    if isinstance(imgs, dict):
        titles = dict(enumerate(imgs.keys()))
        imgs = list(imgs.values())
    elif len(imgs) == 2:
        titles = {0: "Original", 1: "Filtered"}
    n = len(imgs)
    ncols = min(n, 5)
    fig, ax = plt.subplots(
        nrows=nrows or n // ncols, 
        ncols=(n // nrows) if nrows else ncols,
    )
    ax = ax.flatten()
    for i, img in enumerate(imgs):
        ax[i].set_title(f"{titles.get(i, i)}")
        ax[i].imshow(img, cmap='gray')
        ax[i].set_axis_off()
    fig.tight_layout()
    return fig, ax

sample_image = jnp.array(train.x[0])
# %%
filtered_images = {
    f'{cutoff=}': rmfreqs(sample_image, cutoff=cutoff, rm='hi')
    for cutoff in (1, .75, .5, .25, .1)
}
fig, ax = side_by_side(filtered_images)
# %%
filtered_images = {
    f'{cutoff=}': rmfreqs(sample_image, cutoff=cutoff, rm='lo')
    for cutoff in (0, .25, .5, .75, .9)
}
fig, ax = side_by_side(filtered_images)
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

lofreq0 = jax.vmap(rm_hifreq)(jnp.array(i0))
lofreq1 = jax.vmap(rm_hifreq)(jnp.array(i1))
dists_after = compare(lofreq0, lofreq1)
print('After', dists_after.mean(), dists_after.std())

# %%
dists = compare(i0, lofreq0)
print('Within 0 ::', dists.mean(), dists.std())
dists = compare(i1, lofreq1)
print('Within 1 ::', dists.mean(), dists.std())
# %%
_ = side_by_side([i0[0], lofreq0[0], i0[1], lofreq0[1]], nrows=2)
# %%
_ = side_by_side([i0[0], i1[0]])
# %%
_ = side_by_side([lofreq0[0], lofreq1[0]])

# %%
