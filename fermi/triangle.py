r"""三角格子有关的内容
``````
初基矢量的选择
a1 = (-1/2, \sqrt{3}/2)
a2 = (1/2, \sqrt{3}/2)
倒格子的选择
b1 = (-2\pi, 2\pi/\sqrt{3})  #(-1, 1/\sqrt{3})
b2 = ( 2\pi, 2\pi/\sqrt{3})  #(1, 1/\sqrt{3})
"""

import numpy
from scipy import optimize
from basics import Hexagon, Point


def brillouin():
    '''布里渊区'''
    return Hexagon(Point(0, 0, 1), 2*numpy.pi/numpy.sqrt(3))


def dispersion(kxv, kyv):
    '''色散关系'''
    kyp = numpy.sqrt(3) * kyv / 2.0
    return -2*numpy.cos(kxv) - 4*numpy.cos(kyp)*numpy.cos(kxv/2) - 0.85


def get_patches(npat):
    '''获得三角格子的patches，最简单的版本，按角度切分'''
    deltaa = 2*numpy.pi / npat
    angs = [(idx+0.5) * deltaa for idx in range(npat)]
    #print(angs)
    pats = []
    for ang in angs:
        rrad = optimize.bisect(
            lambda rad: dispersion(rad*numpy.cos(ang), rad*numpy.sin(ang)),
            0, 2*numpy.pi/numpy.sqrt(3)
        )
        pats.append(Point(rrad*numpy.cos(ang), rrad*numpy.sin(ang), 1))
    return pats


def shift_kv(kpt: Point, sft: Point):
    '''将一个动量平移'''
    dest = [kpt.coord[idx] + sft.coord[idx] for idx in range(2)]
    #找到离目标最近的一个第一布里渊区的中心
    #如果这个点距离中心是最近的，那么他就在第一布里渊区里面了，如果
    #离其他的点近，就平移一次，再检查一遍
    b1x = -6.283185307179586#-2*numpy.pi
    b1y = 3.6275987284684357#2*numpy.pi/numpy.sqrt(3)
    brlu_cents = [(0, 0), (b1x, b1y), (0, 2*b1y), (-b1x, b1y),\
        (-b1x, -b1y), (0, -2*b1y), (b1x, -b1y)]
    #计算距离
    dis2cents = [numpy.square(dest[0]-cnt[0]) + numpy.square(dest[1]-cnt[1])\
        for cnt in brlu_cents]
    while numpy.argmin(dis2cents) != 0:
        ncnt = numpy.argmin(dis2cents)
        dest[0] = dest[0] - brlu_cents[ncnt][0]
        dest[1] = dest[1] - brlu_cents[ncnt][1]
        dis2cents = [numpy.square(dest[0]-cnt[0]) + numpy.square(dest[1]-cnt[1])\
            for cnt in brlu_cents]
    return Point(dest[0], dest[1], 1)


def get_exchange_gamma(jval, pinfos):
    '''获取交换相互作用的有效大小'''
    pnum = len(pinfos)
    result = numpy.zeros((pnum, pnum, pnum))
    nditer = numpy.nditer(result, flags=['multi_index'])
    co1 = 0.5
    co2 = numpy.sqrt(3) * 0.5
    while not nditer.finished:
        kidx1, kidx2, kidx3 = nditer.multi_index
        kv1, kv2, kv3 = pinfos[kidx1], pinfos[kidx2], pinfos[kidx3]
        #q1v = k_2 - k_4 = k_3 - k_1
        q1v = shift_kv(kv3, Point(-kv1.coord[0], -kv1.coord[1], 1))
        #q2v = k_1 - k_4 = k_3 - k_2
        q2v = shift_kv(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
        phase = numpy.cos(-co1*q1v.coord[0] + co2*q1v.coord[1])
        phase += numpy.cos(co1*q1v.coord[0] + co2*q1v.coord[1])
        phase += numpy.cos(q1v.coord[0])
        phase += 0.5*numpy.cos(-co1*q2v.coord[0] + co2*q2v.coord[1])
        phase += 0.5*numpy.cos(co1*q2v.coord[0] + co2*q2v.coord[1])
        phase += 0.5*numpy.cos(q2v.coord[0])
        result[kidx1, kidx2, kidx3] = -jval * phase
        nditer.iternext()
    return result
