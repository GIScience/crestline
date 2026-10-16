import numpy as np
from numba import cuda

if not hasattr(np, 'row_stack'):
    np.row_stack = np.vstack

THREADS = (32, 8)


@cuda.jit
def _grid(mask: np.ndarray, state: np.ndarray, level: np.ndarray):
    j, i = cuda.grid(2)
    if i < state.shape[0] and j < state.shape[1]:
        inside = 2 <= i < mask.shape[0] + 2 and 2 <= j < mask.shape[1] + 2 and mask[i - 2, j - 2]
        state[i, j] = 2 if inside else 1
        level[i, j] = 0


@cuda.jit
def _wave(state: np.ndarray, level: np.ndarray, step: int, flag: np.ndarray):
    j, i = cuda.grid(2)
    h, w = state.shape
    if i < 1 or j < 1 or i >= h - 1 or j >= w - 1 or state[i, j] != 2 or level[i, j] != 0:
        return

    for di in range(-1, 2):
        for dj in range(-1, 2):
            s, v = state[di + i, dj + j], level[di + i, dj + j]

            last = s == 1 if step == 0 else s == 2 and v == step
            before = s == 1 if step == 1 else s == 2 and v == step - 1

            side, odd = di == 0 or dj == 0, step % 2 == 1

            if (last and (side or odd)) or (before and odd and not side):
                level[i, j] = step + 1
                flag[0] = step + 1
                return


@cuda.jit
def _crests(depth: np.ndarray, crest: np.ndarray):
    j, i = cuda.grid(2)
    h, w = depth.shape
    if i >= h or j >= w:
        return

    d = depth[i, j]
    top = 2 <= i < h - 2 and 2 <= j < w - 2 and d >= 2
    if top:
        for di in range(-2, 3):
            for dj in range(-2, 3):
                top = top and depth[di + i, dj + j] <= d

    crest[i, j] = top


def flood(mask: np.ndarray):
    h, w = mask.shape[0] + 4, mask.shape[1] + 4
    blocks = ((w + THREADS[0] - 1) // THREADS[0], (h + THREADS[1] - 1) // THREADS[1])
    state = cuda.device_array((h, w), np.int8)
    level = cuda.device_array((h, w), np.int32)
    _grid[blocks, THREADS](cuda.to_device(np.ascontiguousarray(mask, bool)), state, level)

    flag = cuda.to_device(np.zeros(1, np.int32))
    step, empty = 0, 0
    while empty < 2:
        _wave[blocks, THREADS](state, level, step, flag)
        empty = 0 if flag.copy_to_host()[0] == step + 1 else empty + 1
        step += 1

    crest = cuda.device_array((h, w), np.bool_)
    _crests[blocks, THREADS](level, crest)

    return level.copy_to_host(), crest.copy_to_host()
