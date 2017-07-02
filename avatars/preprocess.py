
from PIL import Image

max_size = 2000


def scale_int(value, scale):
    return int(round(scale * value, 0))


def scale_range(size, min_size, img_size):
    scale_min = max(size[0] / img_size[0], size[1] / img_size[1])
    scale_max = max(size[0] / min_size[0], size[1] / min_size[1])
    return scale_min, scale_max


def scale_bounds(bounds, scale):
    l, t, r, b = bounds
    l = scale_int(l, scale)
    t = scale_int(t, scale)
    r = scale_int(r, scale)
    b = scale_int(b, scale)
    return (l, t, r, b)


def rescale(size, scale):
    w, h = size
    s_w = scale_int(w, scale)
    s_h = scale_int(h, scale)
    return (s_w, s_h)


def prescale(size):
    img_size = max(size)
    if img_size > max_size:
        scale = max_size / img_size
        return scale
    return 1.


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


def square_bounds(size, max_size, step):
    m_w, m_h, w, h = *max_size, *size
    l, t, r, b = 0, 0, w, h
    while b <= m_h:
        while r <= m_w:
            yield (l, t, r, b)
            l, r = l + step, r + step
        l, r = 0, w
        t, b = t + step, b + step


def decompose(img_size, size=(70, 70), min_size=(70, 70),
              scale_step=0.1, step=20):
    scale_min, scale_max = scale_range(size, min_size, img_size)
    for scale in bound_range(scale_min, scale_max, scale_step):
        scaled_size = rescale(img_size, scale)
        for bounds in square_bounds(size, scaled_size, step):
            yield scale, bounds


def bounds_overlap(b1, b2):
    x_o = (b1[0] >= b2[0] and b1[0] <= b2[2] or
           b2[0] >= b1[0] and b2[0] <= b1[2])
    y_o = (b1[1] >= b2[1] and b1[1] <= b2[3] or
           b2[1] >= b1[1] and b2[1] <= b1[3])
    return x_o and y_o
