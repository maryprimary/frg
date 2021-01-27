"""正方格子的布里渊区"""

import numpy
from basics import Square, Point, Segment


def brillouin():
    '''布里渊区'''
    return Square(Point(0., 0., 1), numpy.pi * 2.)


def dispersion(kxv, kyv):
    '''色散关系\n
    kxv是x方向的动量，kyv是y方向的动量'''
    return -2 * (numpy.cos(kxv) + numpy.cos(kyv))


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
