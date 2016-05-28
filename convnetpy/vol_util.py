from convnetpy.vol import Vol
from convnetpy.util import randi

def augment(V, crop, grayscale=False):
    # note assumes square outputs of size crop x crop
    # randomly sample a crop in the input volume
    if crop == V.sx: return V

    dx = randi(0, V.sx - crop)
    dy = randi(0, V.sy - crop)

    W = Vol(crop, crop, V.depth)
    for x in range(crop):
        for y in range(crop):
            if x + dx < 0 or x + dx >= V.sx or \
                y + dy < 0 or y + dy >= V.sy: continue
            for d in range(V.depth):
                W.set(x, y, d, V.get(x + dx, y + dy, d))

    if grayscale:
        #flatten into depth=1 array
        G = Vol(crop, crop, 1, 0.0)
        for i in range(crop):
            for j in range(crop):
                G.set(i, j, 0, W.get(i, j, 0))
        W = G

    return W