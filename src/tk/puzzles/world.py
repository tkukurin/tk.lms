"""Based on GPT puzzles by Francois Fleuret <francois@fleuret.org>
"""
import tqdm

import torch

from torch import nn
from torch.nn import functional as F
from .utils import save_image

######################################################################


colors = torch.tensor(
    [
        [255, 255, 255],
        [255, 0, 0],
        [0, 192, 0],
        [0, 0, 255],
        [255, 192, 0],
        [0, 255, 255],
        [255, 0, 255],
        [192, 255, 192],
        [255, 192, 192],
        [192, 192, 255],
        [192, 192, 192],
    ]
)

token_background = 0
first_bird_token = 1
nb_bird_tokens = colors.size(0) - 1
token_forward = first_bird_token + nb_bird_tokens
token_backward = token_forward + 1

token2char = "_" + "".join([chr(ord("A") + n) for n in range(len(colors) - 1)]) + "><"


def generate_seq(
    nb, height, width, nb_birds=3, nb_iterations=2, return_iterations=False
):
    pairs = []
    kept_iterations = []

    for _ in tqdm.tqdm(range(nb), dynamic_ncols=True, desc="world generation"):
        while True:
            iterations = []

            f_start = torch.zeros(height, width, dtype=torch.int64)

            i, j, vi, vj = (
                torch.empty(nb_birds, dtype=torch.int64),
                torch.empty(nb_birds, dtype=torch.int64),
                torch.empty(nb_birds, dtype=torch.int64),
                torch.empty(nb_birds, dtype=torch.int64),
            )

            col = torch.randperm(colors.size(0) - 1)[:nb_birds].sort().values + 1

            for n in range(nb_birds):
                c = col[n]

                while True:
                    i[n], j[n] = (
                        torch.randint(height, (1,))[0],
                        torch.randint(width, (1,))[0],
                    )
                    vm = torch.randint(4, (1,))[0]
                    vi[n], vj[n] = (vm % 2) * 2 - 1, (vm // 2) * 2 - 1
                    if (
                        i[n] - vi[n] >= 0
                        and i[n] - vi[n] < height
                        and j[n] - vj[n] >= 0
                        and j[n] - vj[n] < width
                        and f_start[i[n], j[n]] == 0
                        and f_start[i[n] - vi[n], j[n]] == 0
                        and f_start[i[n], j[n] - vj[n]] == 0
                    ):
                        break

                f_start[i[n], j[n]] = c
                f_start[i[n] - vi[n], j[n]] = c
                f_start[i[n], j[n] - vj[n]] = c

            f_end = f_start.clone()

            for l in range(nb_iterations):
                iterations.append(f_end.clone())
                f_end[...] = 0
                nb_collisions = 0
                for n in range(nb_birds):
                    c = col[n]

                    pi, pj, pvi, pvj = (
                        i[n].item(),
                        j[n].item(),
                        vi[n].item(),
                        vj[n].item(),
                    )

                    if (i[n] == 0 and vi[n] == -1) or (
                        i[n] == height - 1 and vi[n] == 1
                    ):
                        vi[n] = -vi[n]
                    if (j[n] == 0 and vj[n] == -1) or (
                        j[n] == width - 1 and vj[n] == 1
                    ):
                        vj[n] = -vj[n]

                    i[n] += vi[n]
                    j[n] += vj[n]

                    if not (
                        f_end[i[n], j[n]] == 0
                        and f_end[i[n] - vi[n], j[n]] == 0
                        and f_end[i[n], j[n] - vj[n]] == 0
                    ):
                        nb_collisions += 1

                    f_end[i[n], j[n]] = c
                    f_end[i[n] - vi[n], j[n]] = c
                    f_end[i[n], j[n] - vj[n]] = c

            iterations.append(f_end.clone())

            if nb_collisions == 0:
                break

        kept_iterations.append(iterations)
        pairs.append((f_start, f_end))

    result = []
    for p in pairs:
        if torch.rand(1) < 0.5:
            result.append(
                torch.cat(
                    [p[0].flatten(), torch.tensor([token_forward]), p[1].flatten()],
                    dim=0,
                )[None, :]
            )
        else:
            result.append(
                torch.cat(
                    [p[1].flatten(), torch.tensor([token_backward]), p[0].flatten()],
                    dim=0,
                )[None, :]
            )

    if return_iterations:
        # iterations = torch.cat([ torch.cat([ x[None, None] for x in l], dim = 1) for l in kept_iterations ], dim=0)
        return torch.cat(result, dim=0), kept_iterations
    else:
        return torch.cat(result, dim=0)



def frame2img(x, height, width, upscale=15):
    x = x.reshape(-1, height, width)
    m = torch.logical_and(x >= 0, x < first_bird_token + nb_bird_tokens).long()
    x = colors[x * m].permute(0, 3, 1, 2)
    s = x.shape
    x = x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale)
    x = x.reshape(s[0], s[1], s[2] * upscale, s[3] * upscale)

    x[:, :, :, torch.arange(0, x.size(3), upscale)] = 0
    x[:, :, torch.arange(0, x.size(2), upscale), :] = 0
    x = x[:, :, 1:, 1:]

    for n in range(m.size(0)):
        for i in range(m.size(1)):
            for j in range(m.size(2)):
                if m[n, i, j] == 0:
                    for k in range(2, upscale - 2):
                        x[n, :, i * upscale + k, j * upscale + k] = 0
                        x[n, :, i * upscale + upscale - 1 - k, j * upscale + k] = 0

    return x


def seq2img(seq, height, width, upscale=15):
    f_first = seq[:, : height * width].reshape(-1, height, width)
    f_second = seq[:, height * width + 1 :].reshape(-1, height, width)
    direction = seq[:, height * width]

    direction_symbol = torch.full((direction.size(0), height * upscale - 1, upscale), 0)
    direction_symbol = colors[direction_symbol].permute(0, 3, 1, 2)
    separator = torch.full((direction.size(0), 3, height * upscale - 1, 1), 0)

    for n in range(direction_symbol.size(0)):
        if direction[n] == token_forward:
            for k in range(upscale):
                direction_symbol[
                    n,
                    :,
                    (height * upscale) // 2 - upscale // 2 + k,
                    3 + upscale // 2 - abs(k - upscale // 2),
                ] = 0
        elif direction[n] == token_backward:
            for k in range(upscale):
                direction_symbol[
                    n,
                    :,
                    (height * upscale) // 2 - upscale // 2 + k,
                    3 + abs(k - upscale // 2),
                ] = 0
        else:
            for k in range(2, upscale - 2):
                direction_symbol[
                    n, :, (height * upscale) // 2 - upscale // 2 + k, k
                ] = 0
                direction_symbol[
                    n, :, (height * upscale) // 2 - upscale // 2 + k, upscale - 1 - k
                ] = 0

    return torch.cat(
        [
            frame2img(f_first, height, width, upscale),
            separator,
            direction_symbol,
            separator,
            frame2img(f_second, height, width, upscale),
        ],
        dim=3,
    )


def seq2str(seq):
    result = []
    for s in seq:
        result.append("".join([token2char[v] for v in s]))
    return result


if __name__ == "__main__":
    import time

    height, width = 6, 8
    start_time = time.perf_counter()
    seq, it = generate_seq(
        nb=64, height=height, width=width, nb_iterations=100, return_iterations=True
    )
    delay = time.perf_counter() - start_time
    print(f"{seq.size(0)/delay:02f} samples/s")

    print(seq2str(seq[:4]))

    for t in range(len(it[0])):
        img = torch.cat([frame2img(f[t], height, width) for f in it], dim=0)
        save_image(
            img.float() / 255.0,
            f"/tmp/frame_{t:03d}.png",
            nrow=8,
            padding=6,
            pad_value=0,
        )

    img = seq2img(seq, height, width)
    print(img.size())

    save_image(
        img.float() / 255.0, "/tmp/world.png", nrow=6, padding=6, pad_value=0
    )
