"""正方格子的有关内容"""

import numpy
from basics import Square, Point, Segment


def brillouin():
    '''布里渊区'''
    return Square(Point(0., 0., 1), numpy.pi * 2.)


def dispersion(kxv, kyv):
    '''色散关系\n
    kxv是x方向的动量，kyv是y方向的动量'''
    return -2 * (numpy.cos(kxv) + numpy.cos(kyv))


def hole_disp(kxv, kyv):
    '''色散关系\n
    kxv是x方向的动量，kyv是y方向的动量'''
    return -2 * (numpy.cos(kxv) + numpy.cos(kyv)) + 0.2


def dispersion_gradient(kxv, kyv):
    '''色散关系的梯度切线\n
    返回一个角度，这个角度是从这个kx引出的线的角度\n
    '''
    slope_den = numpy.sin(kxv)
    slope_num = numpy.sin(kyv)
    if slope_den == 0:
        #这个时候是一个竖直的线
        return numpy.pi / 2
    slope = slope_num / slope_den
    return numpy.arctan(slope)


def shift_kv(kpt: Point, sft: Point) -> Point:
    '''将一个k点平移，然后重新对应到第一布里渊区'''
    dest = [kpt.coord[idx] + sft.coord[idx] for idx in range(2)]
    while numpy.abs(dest[0]) > numpy.pi:
        #如果大于第一布里渊区就向零靠拢2pi
        dest[0] -= numpy.sign(dest[0]) * numpy.pi * 2
    while numpy.abs(dest[1]) > numpy.pi:
        dest[1] -= numpy.sign(dest[1]) * numpy.pi * 2
    return Point(dest[0], dest[1], 1)
