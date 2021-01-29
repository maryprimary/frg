"""计算费米面上点的位置"""


import numpy
from scipy import optimize
from basics import Square, Point


def get_patches(brlu: Square, npatch, dispfun):
    '''获得费米面上面的patch\n
    dispfun是色散关系\n
    '''
    gap = numpy.pi * 2 / npatch
    angles = [gap * (idx + 0.5) for idx in range(npatch)]
    #解出每个角度下和费米面的交点
    patches = []
    #半径最大是这么多
    maxv = brlu.width * 1.414 / 2.
    for ang in angles:
        xcoff = numpy.cos(ang)
        ycoff = numpy.sin(ang)
        def __raddisp(rad):
            kxv = rad * xcoff
            if numpy.abs(kxv) > numpy.pi:
                kxv = numpy.sign(kxv) * numpy.pi
            kyv = rad * ycoff
            if numpy.abs(kyv) > numpy.pi:
                kyv = numpy.sign(kyv) * numpy.pi
            return dispfun(kxv, kyv)
        rrad = optimize.bisect(
            __raddisp,
            0., maxv
        )
        patches.append(Point(rrad * xcoff, rrad * ycoff, 1))
    return patches


def find_patch(pnt: Point, patches, dispfun, dispgdfun):
    '''找到这个点是属于哪个patch的\n
    dispfun是色散关系的函数，dispgdfun是向费米面投影的梯度\n
    '''
    #从这个点引出一条直线
    kxv, kyv = pnt.coord
    cita = dispgdfun(kxv, kyv)
    #
    def __disp_by_dis(dis):
        '''从pnt这个点沿着slope走dis这么长的位置上的能量'''
        xdis = kxv + dis * numpy.cos(cita)
        ydis = kyv + dis * numpy.sin(cita)
        if numpy.abs(xdis) > numpy.pi:
            xdis = numpy.sign(xdis) * numpy.pi
        if numpy.abs(ydis) > numpy.pi:
            ydis = numpy.sign(ydis) * numpy.pi
        return dispfun(xdis, ydis)
    #首先寻找一个初始的位置，向更接近费米面的位置，（牛顿法容易找到远处的交点）
    #如果向0.1的方向更小，那么是正数
    if numpy.abs(__disp_by_dis(-0.1)) > numpy.abs(__disp_by_dis(0.1)):
        gsign = 1
    else:
        gsign = -1
    intsign = numpy.sign(dispfun(kxv, kyv))
    gues = 0.01
    while True:
        #如果已经穿过了费米面就停下
        if numpy.sign(__disp_by_dis(gsign * gues)) != intsign:
            break
        gues += 0.01
    #这个时候gues已经反号，而gues-0.01还没有
    rootd = optimize.bisect(__disp_by_dis, (gues-0.01) * gsign, gues * gsign)
    crsx = kxv + rootd * numpy.cos(cita)
    crsy = kyv + rootd * numpy.sin(cita)
    dis_to_patch = [numpy.square(crsx - pat.coord[0]) +\
        numpy.square(crsy - pat.coord[1])\
        for pat in patches]
    return numpy.argmin(dis_to_patch)
