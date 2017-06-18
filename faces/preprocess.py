
from PIL import Image


def scale_range(size, min_size, img_size):
    scale_min = max(size[0] / img_size[0], size[1] / img_size[1])
    scale_max = max(size[0] / min_size[0], size[1] / min_size[1])
    return scale_min, scale_max


def rescale(size, scale):
    s_w = int(round(scale * size[0], 0))
    s_h = int(round(scale * size[1], 0))
    return (s_w, s_h)


def bound_range(lower, upper, step):
    v = lower
    while v < 1.:
        yield v
        v += step
    v = 1.
    while v < upper:
        yield v
        v += step
    yield upper


def sqr_bounds(size, max_size, step):
    m_w, m_h, w, h = *max_size, *size
    l, t, r, b = 0, 0, w, h
    while b <= m_h:
        while r <= m_w:
            yield (l, t, r, b)
            l, r = l + step, r + step
        l, r = 0, w
        t, b = t + step, b + step


def decompose(image_path, size=(70, 70), min_size=(50, 50),
              scale_step=0.1, step=20):
    img = Image.open(image_path).convert('L')
    scale_min, scale_max = scale_range(size, min_size, img.size)
    for scale in bound_range(scale_min, scale_max, scale_step):
        scaled_size = rescale(img.size, scale)
        scaled_img = img.resize(scaled_size)
        for bounds in sqr_bounds(size, scaled_img.size, step):
            region = scaled_img.crop(bounds)
            yield region, scale, bounds
