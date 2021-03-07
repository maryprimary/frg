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
    #这里没有按照公式使用exp(-i kx/2), 最后解释
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
    #对A和B求和
    #这里注意，如果四个加起来的kx接近0，则B格子的4个e^i(k_x/2)乘起来是1
    #如果接近2 pi 则会是-1，接近4pi还是1
    ksum = kv1.coord[0] + kv2.coord[0] - kv3.coord[0] - kv4.coord[0]
    coef = numpy.cos(ksum / 2.0)
    result = Ts_A_bd1 * Ts_A_bd2 * T_A_bd3 * T_A_bd4
    result += coef * Ts_B_bd1 * Ts_B_bd2 * T_B_bd3 * T_B_bd4
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


def basis_uval(banduval, st1, st2, st3, st4, kv1, kv2, kv3, kv4, patchdic):
    '''获得basis上的u值'''
    #pylint: disable=invalid-name
    #st1这个格子
    sq1 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv1.coord[0]) + 2)
    #
    eig_s_A = 2 * numpy.cos(kv1.coord[0] / 2.0)
    eig_s_A = eig_s_A / (STRIPE - sq1)
    eig_p_A = 2 * numpy.cos(kv1.coord[0] / 2.0)
    eig_p_A = eig_p_A / (STRIPE + sq1)
    #求归一化系数
    norm1_st1 = 1 + numpy.square(eig_s_A)
    norm2_st1 = 1 + numpy.square(eig_p_A)
    #
    Rs_b1_st1 = numpy.zeros(2, dtype=numpy.complex)
    #如果第一个是A
    if st1 == 0:
        Rs_b1_st1[0] = eig_s_A / numpy.sqrt(norm1_st1)#Rs_s_A
        Rs_b1_st1[1] = eig_p_A / numpy.sqrt(norm2_st1)#Rs_p_A
    #如果第一个是B
    elif st1 == 1:
        Rs_b1_st1 += numpy.cos(kv1.coord[0] / 2.0)#Rs_s_B和Rs_p_B的实部
        Rs_b1_st1.imag += numpy.sin(kv1.coord[0] / 2.0)#Rs_s_B和Rs_p_B的虚部
        Rs_b1_st1[0] /= numpy.sqrt(norm1_st1)
        Rs_b1_st1[1] /= numpy.sqrt(norm2_st1)
    else:
        raise ValueError('st1数值不对')
    #st2这个格子
    sq2 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv2.coord[0]) + 2)
    #
    eig_s_A = 2 * numpy.cos(kv2.coord[0] / 2.0)
    eig_s_A = eig_s_A / (STRIPE - sq2)
    eig_p_A = 2 * numpy.cos(kv2.coord[0] / 2.0)
    eig_p_A = eig_p_A / (STRIPE + sq2)
    #求归一化系数
    norm1_st2 = 1 + numpy.square(eig_s_A)
    norm2_st2 = 1 + numpy.square(eig_p_A)
    #
    Rs_b2_st2 = numpy.zeros(2, dtype=numpy.complex)
    #如果st2是A
    if st2 == 0:
        Rs_b2_st2[0] = eig_s_A / numpy.sqrt(norm1_st2)#Rs_s_A
        Rs_b2_st2[1] = eig_p_A / numpy.sqrt(norm2_st2)#Rs_p_A
    #如果st2是B
    elif st2 == 1:
        Rs_b2_st2 += numpy.cos(kv2.coord[0] / 2.0)#Rs_s_B和Rs_p_B的实部
        Rs_b2_st2.imag += numpy.sin(kv2.coord[0] / 2.0)#Rs_s_B和Rs_p_B的虚部
        Rs_b2_st2[0] /= numpy.sqrt(norm1_st2)
        Rs_b2_st2[1] /= numpy.sqrt(norm2_st2)
    else:
        raise ValueError('st2数值不对')
    #st3这个格子
    sq3 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv3.coord[0]) + 2)
    #
    eig_s_A = 2 * numpy.cos(kv3.coord[0] / 2.0)
    eig_s_A = eig_s_A / (STRIPE - sq3)
    eig_p_A = 2 * numpy.cos(kv3.coord[0] / 2.0)
    eig_p_A = eig_p_A / (STRIPE + sq3)
    #求归一化系数
    norm1_st3 = 1 + numpy.square(eig_s_A)
    norm2_st3 = 1 + numpy.square(eig_p_A)
    #
    R_b3_st3 = numpy.zeros(2, dtype=numpy.complex)
    #如果st3是A
    if st3 == 0:
        R_b3_st3[0] = eig_s_A / numpy.sqrt(norm1_st3)#R_s_A
        R_b3_st3[1] = eig_p_A / numpy.sqrt(norm2_st3)#R_p_A
    #如果st3是B
    elif st3 == 1:
        R_b3_st3 += numpy.cos(kv3.coord[0] / 2.0)#R_s_B和R_p_B的实数
        R_b3_st3.imag += -numpy.sin(kv3.coord[0] / 2.0)#R_s_B和R_p_B的虚数
        R_b3_st3[0] /= numpy.sqrt(norm1_st3)
        R_b3_st3[1] /= numpy.sqrt(norm2_st3)
    else:
        raise ValueError('st3数值不对')
    #st4这个格子
    sq4 = numpy.sqrt(numpy.square(STRIPE) + 2 * numpy.cos(kv4.coord[0]) + 2)
    #
    eig_s_A = 2 * numpy.cos(kv4.coord[0] / 2.0)
    eig_s_A = eig_s_A / (STRIPE - sq4)
    eig_p_A = 2 * numpy.cos(kv4.coord[0] / 2.0)
    eig_p_A = eig_p_A / (STRIPE + sq4)
    #求归一化系数
    norm1_st4 = 1 + numpy.square(eig_s_A)
    norm2_st4 = 1 + numpy.square(eig_p_A)
    #
    R_b4_st4 = numpy.zeros(2, dtype=numpy.complex)
    #如果st4是A
    if st4 == 0:
        R_b4_st4[0] = eig_s_A / numpy.sqrt(norm1_st4)#R_s_A
        R_b4_st4[1] = eig_p_A / numpy.sqrt(norm2_st4)#R_p_A
    elif st4 == 1:
        R_b4_st4 += numpy.cos(kv4.coord[0] / 2.0)#R_s_B和R_p_B的实数
        R_b4_st4.imag += -numpy.sin(kv4.coord[0] / 2.0)#R_s_B和R_p_B的虚数
        R_b4_st4[0] /= numpy.sqrt(norm1_st4)
        R_b4_st4[1] /= numpy.sqrt(norm2_st4)
    else:
        raise ValueError('st4数值不对')
    #对b1, b2, b3, b4求和
    #U_{st1,st2,st3,st4}(k1,k2,k3,k4) = \sum_{b1,b2,b3,b4}
    # R*_{b1,st1}(k1)R*_{b2,st2}(k2)R_{b3,st3}(k3)R_{b4,st4}(k4)
    # U_{b1,b2,b3,b4}(k1,k2,k3,k4) --> U_{b1,b2,b3,b4}(P_b1(k1),P_b2(k2),P_b3(k3),P_b4(k4))
    #patchdic应该是（2，3）其中第一个是s,p，第二个是第几个k，这个在调用这个函数之前就应该是知道的
    #就是kv1，kv2,kv3向s,p的投影
    place_holder = numpy.ndarray((2, 2, 2, 2))
    result = 0.
    ndit = numpy.nditer(place_holder, flags=['multi_index'])
    while not ndit.finished:
        ib1, ib2, ib3, ib4 = ndit.multi_index
        p_b1_k1 = patchdic[ib1, 0]
        p_b2_k2 = patchdic[ib2, 1]
        p_b3_k3 = patchdic[ib3, 2]
        #val = Rs_b1_st1[ib1] * Rs_b2_st2[ib2] * R_b3_st3[ib3] * R_b4_st4[ib4]
        #print(val)
        #print(banduval[ib1, ib2, ib3, ib4, p_b1_k1, p_b2_k2, p_b3_k3])
        #print(ib1, ib2, ib3, ib4)
        #if val < 0:
        #    print(kv1, kv2, kv3, kv4, val)
        #    print(banduval[ib1, ib2, ib3, ib4, p_b1_k1, p_b2_k2, p_b3_k3])
        #    print(Rs_b1_st1[ib1], Rs_b2_st2[ib2], R_b3_st3[ib3], R_b4_st4[ib4])
        #    raise
        #bv = band_uval(2., ib1, ib2, ib3, ib4, kv1, kv2, kv3, kv4)
        #if not numpy.allclose(bv, banduval[ib1, ib2, ib3, ib4, p_b1_k1, p_b2_k2, p_b3_k3]):
        #    print(ib1, ib2, ib3, ib4)
        #    print(kv1, kv2, kv3, kv4)
        #    print(p_b1_k1, p_b2_k2, p_b3_k3)
        #    print(bv, banduval[ib1, ib2, ib3, ib4, p_b1_k1, p_b2_k2, p_b3_k3])
        #    kv1 = Point(-2.8574, -1.4287, 1)
        #    kv2 = Point(-2.8574, -1.4287, 1)
        #    kv3 = Point(-2.3549, -1.9642, 1)
        #    kv4 = shift_kv(shift_kv(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
        #    print(kv1, kv2, kv3, kv4)
        #    bv2 = band_uval(2., ib1, ib2, ib3, ib4, kv1, kv2, kv3, kv4)
        #    print(bv2)
        #    raise
        result += Rs_b1_st1[ib1] * Rs_b2_st2[ib2] * R_b3_st3[ib3] * R_b4_st4[ib4]\
            * banduval[ib1, ib2, ib3, ib4, p_b1_k1, p_b2_k2, p_b3_k3]#bv#
        ndit.iternext()
    return result


def inverse_uval(pinfos, spats, ppats, uval):
    '''将band表示转换回basis表示，需要一系列的patches，
    然后利用find_patch找到这些patches对应的值'''
    from .patches import find_patch
    pnum = len(pinfos)
    place_holder = numpy.zeros(
        (2, 2, 2, 2, pnum, pnum, pnum)
    )
    img_part = numpy.zeros_like(place_holder)
    phiter = numpy.nditer(place_holder, flags=['multi_index'])
    step = numpy.pi / 100
    #找出这些patches对应的P_b(k)
    allpatdic = numpy.ndarray((2, pnum), dtype=numpy.int)
    #print(pinfos[0])
    #print(find_patch(pinfos[0], spats, s_band_disp, s_band_gd, step))
    #print(find_patch(pinfos[0], ppats, p_band_disp, p_band_gd, step))
    #raise
    for idx, pat in enumerate(pinfos, 0):
        allpatdic[0, idx] = find_patch(pat, spats, s_band_disp, s_band_gd, step)
        allpatdic[1, idx] = find_patch(pat, ppats, p_band_disp, p_band_gd, step)
    while not phiter.finished:
        st1, st2, st3, st4, idx1, idx2, idx3 = phiter.multi_index
        kv1 = pinfos[idx1]
        kv2 = pinfos[idx2]
        kv3 = pinfos[idx3]
        kv4 = shift_kv(shift_kv(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
        #每个点在不同能带下对应的patch
        patdic = numpy.ndarray((2, 3), dtype=numpy.int)
        patdic[:, 0] = allpatdic[:, idx1]
        patdic[:, 1] = allpatdic[:, idx2]
        patdic[:, 2] = allpatdic[:, idx3]
        bval = basis_uval(uval, st1, st2, st3, st4, kv1, kv2, kv3, kv4, patdic)
        place_holder[st1, st2, st3, st4, idx1, idx2, idx3] = numpy.real(bval)
        img_part[st1, st2, st3, st4, idx1, idx2, idx3] = numpy.imag(bval)
        phiter.iternext()
    return place_holder, img_part
