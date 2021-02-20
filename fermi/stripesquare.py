"""带有stripe的正方格子"""

import numpy
from scipy import optimize
from basics import Square, Point, Segment

STRIPE = None

def brillouin():
    '''布里渊区'''
    return Square(Point(0., 0., 1), numpy.pi * 2.)


def set_stripe(sval):
    '''设置stripe，注意色散里用的stripe是负数的'''
    global STRIPE
    STRIPE = sval


def get_max_val():
    '''能带最大的数值'''
    stripe = STRIPE
    sq_max = numpy.square(stripe) + 4
    return numpy.sqrt(sq_max) + 2.

def s_band_disp(kxv, kyv):
    '''s带的色散关系'''
    stripe = STRIPE
    sq_ = numpy.square(stripe) + 2*numpy.cos(kxv) + 2
    #16*numpy.square(numpy.cos(kxv / 2.0))
    return -1 * (-numpy.sqrt(sq_) + 2. * numpy.cos(kyv))
    #return -1 * (stripe / 2. - 0.5 * numpy.sqrt(sq_) + 2. * numpy.cos(kyv))


def p_band_disp(kxv, kyv):
    '''p带的色散关系'''
    stripe = STRIPE
    sq_ = numpy.square(stripe) + 2*numpy.cos(kxv) + 2
    #16*numpy.square(numpy.cos(kxv / 2.0))
    return -1 * (numpy.sqrt(sq_) + 2. * numpy.cos(kyv))


def s_band_gd(kxv, kyv):
    '''s带向费米面的导数\n
    用类似正方格子的导数，交点在第一布里渊区外面，所以这里改成向点引直线\n
    '''
    target = (-numpy.pi, 0)
    if kxv > 0:
        target = (numpy.pi, 0)
    slope_den = kxv - target[0]
    slope_num = kyv - target[1]
    if slope_den == 0:
        #这个时候是一个竖直的线
        return numpy.pi / 2
    slope = slope_num / slope_den
    return numpy.arctan(slope)


def p_band_gd(kxv, kyv):
    '''p带向费米面的导数\n
    用类似s带的投影'''
    target = [numpy.pi, numpy.pi]
    if kxv < 0:
        target[0] = -numpy.pi
    if kyv < 0:
        target[1] = -numpy.pi
    slope_den = kxv - target[0]
    slope_num = kyv - target[1]
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


def get_s_band_patches(pnum):
    '''获得s能带的patch'''
    if (pnum % 4) != 0:
        raise ValueError('pnum必须是4的倍数')
    ppnum = pnum // 4
    patches = []
    #先算右边
    gap = numpy.pi / 2 / ppnum
    maxr = numpy.pi * 1.414
    for idx in range(ppnum * 2):
        angle = (idx + 0.5) * gap
        ycoff = numpy.cos(angle)
        xcoff = numpy.sin(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = numpy.pi - rad * xcoff
            ycord = ycoff * rad
            return s_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(numpy.pi - rrad * xcoff, ycoff * rrad, 1))
    #再算左边
    for idx in range(ppnum * 2):
        angle = (idx + 0.5) * gap
        ycoff = -numpy.cos(angle)
        xcoff = numpy.sin(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = -numpy.pi + rad * xcoff
            ycord = ycoff * rad
            return s_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(-numpy.pi + rrad * xcoff, ycoff * rrad, 1))
    return patches


def get_p_band_patches(pnum):
    '''获取p能带的pathces'''
    if (pnum % 4) != 0:
        raise ValueError('pnum必须是4的倍数')
    ppnum = pnum // 4
    patches = []
    gap = numpy.pi / 2 / ppnum
    maxr = numpy.pi * 1.414
    #先算左下角
    for idx in range(ppnum):
        angle = (idx + 0.5) * gap
        xcoff = numpy.sin(angle)
        ycoff = numpy.cos(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = xcoff * rad - numpy.pi
            ycord = ycoff * rad - numpy.pi
            return p_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(xcoff * rrad - numpy.pi, ycoff * rrad - numpy.pi, 1))
    #再算右下角
    for idx in range(ppnum):
        angle = (idx + 0.5) * gap
        xcoff = numpy.cos(angle)
        ycoff = numpy.sin(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = numpy.pi - xcoff * rad
            ycord = ycoff * rad - numpy.pi
            return p_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(numpy.pi - xcoff * rrad, ycoff * rrad - numpy.pi, 1))
    #再算右上角
    for idx in range(ppnum):
        angle = (idx + 0.5) * gap
        xcoff = numpy.sin(angle)
        ycoff = numpy.cos(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = numpy.pi - xcoff * rad
            ycord = numpy.pi - ycoff * rad
            return p_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(numpy.pi - xcoff * rrad, numpy.pi - ycoff * rrad, 1))
    #再算左上角
    for idx in range(ppnum):
        angle = (idx + 0.5) * gap
        xcoff = numpy.cos(angle)
        ycoff = numpy.sin(angle)
        def __raddisp(rad):
            '''以半径为基准的色散'''
            xcord = xcoff * rad - numpy.pi
            ycord = numpy.pi - ycoff * rad
            return p_band_disp(xcord, ycord)
        rrad = optimize.bisect(__raddisp, 0., maxr)
        patches.append(Point(xcoff * rrad - numpy.pi, numpy.pi - ycoff * rrad, 1))
    return patches


def band_uval(basisuval, bd1, bd2, bd3, bd4, kv1, kv2, kv3, kv4):
    '''bd*代表能带，0=s, 1=p\n
    kv*需要是一个Point，代表动量矢量\n
    '''
    #pylint: disable=invalid-name
    #s带里面是 \mu - \sqrt{\mu^{2} + 2 \cos{kx} + 2}
    #p带里面是 \mu + \sqrt{\mu^{2} + 2 \cos{kx} + 2}
    #bd1这个带的本征向量的分量
    sign_bd1 = -1 if bd1 == 0 else 1
    sq1 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv1.coord[0]) + 2)
    #
    eig_A_bd1 = 2 * numpy.cos(kv1.coord[0] / 2.0)
    eig_A_bd1 = eig_A_bd1 / (STRIPE + sign_bd1 * sq1)
    #求归一化系数
    norm_bd1 = 1 + numpy.square(eig_A_bd1)
    #求第一个bd需要的两个T^*
    Ts_A_bd1 = eig_A_bd1 / numpy.sqrt(norm_bd1)
    Ts_B_bd1 = -1 / numpy.sqrt(norm_bd1)
    #print('T^{*}(k_1)', Ts_A_bd1, Ts_B_bd1)
    #这里没有按照公式使用exp(-i kx/2), 因为对B子格子的求和一定是两个
    #exp(-i kx/2)和两个exp(i kx/2)，最后就是1
    #bd2这个带的本征向量的分量
    sign_bd2 = -1 if bd2 == 0 else 1
    sq2 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv2.coord[0]) + 2)
    #
    eig_A_bd2 = 2 * numpy.cos(kv2.coord[0] / 2.0)
    eig_A_bd2 = eig_A_bd2 / (STRIPE + sign_bd2 * sq2)
    #求归一化系数
    norm_bd2 = 1 + numpy.square(eig_A_bd2)
    #求第二个bd需要的两个T^*
    Ts_A_bd2 = eig_A_bd2 / numpy.sqrt(norm_bd2)
    Ts_B_bd2 = -1 / numpy.sqrt(norm_bd2)
    #print('T^{*}(k_2)', Ts_A_bd2, Ts_B_bd2)
    #bd3这个带的本征向量的分量
    sign_bd3 = -1 if bd3 == 0 else 1
    sq3 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv3.coord[0]) + 2)
    #
    eig_A_bd3 = 2 * numpy.cos(kv3.coord[0] / 2.0)
    eig_A_bd3 = eig_A_bd3 / (STRIPE + sign_bd3 * sq3)
    #求归一化系数
    norm_bd3 = 1 + numpy.square(eig_A_bd3)
    #求第三个bd需要的两个T
    T_A_bd3 = eig_A_bd3 / numpy.sqrt(norm_bd3)
    T_B_bd3 = 1 / numpy.sqrt(norm_bd3)
    #print('T(K_3)', T_A_bd3, T_B_bd3)
    #bd4这个带的本征向量的分量
    sign_bd4 = -1 if bd4 == 0 else 1
    sq4 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv4.coord[0]) + 2)
    #
    eig_A_bd4 = 2 * numpy.cos(kv4.coord[0] / 2.0)
    eig_A_bd4 = eig_A_bd4 / (STRIPE + sign_bd4 * sq4)
    #求归一化系数
    norm_bd4 = 1 + numpy.square(eig_A_bd4)
    #求第四个bd需要的两个T
    T_A_bd4 = eig_A_bd4 / numpy.sqrt(norm_bd4)
    T_B_bd4 = 1 / numpy.sqrt(norm_bd4)
    #print('T(k_4)', T_A_bd4, T_B_bd4)
    #
    result = Ts_A_bd1 * Ts_A_bd2 * T_A_bd3 * T_A_bd4
    result += Ts_B_bd1 * Ts_B_bd2 * T_B_bd3 * T_B_bd4
    if numpy.abs(result) < 1.e-10:
        result = 0.
    return result * basisuval


def get_initu(spats, ppats, uval):
    '''获取初始化的u值'''
    pnum = len(spats)
    if pnum != len(ppats):
        raise ValueError('patch数量不一致')
    place_holder = numpy.ndarray(
        (2, 2, 2, 2, pnum, pnum, pnum)
    )
    #kv4直接通过前三个kv可以确定，而他属于的idx4这里是不需要的
    phiter = numpy.nditer(place_holder, flags=['multi_index'])
    while not phiter.finished:
        bd1, bd2, bd3, bd4, idx1, idx2, idx3 = phiter.multi_index
        kv1 = spats[idx1] if bd1 == 0 else ppats[idx1]
        kv2 = spats[idx2] if bd2 == 0 else ppats[idx2]
        kv3 = spats[idx3] if bd3 == 0 else ppats[idx3]
        kv4 = shift_kv(shift_kv(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
        place_holder[bd1, bd2, bd3, bd4, idx1, idx2, idx3] =\
            band_uval(uval, bd1, bd2, bd3, bd4, kv1, kv2, kv3, kv4)
        phiter.iternext()
    return place_holder
