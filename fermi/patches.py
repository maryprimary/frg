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


def find_patch(pnt: Point, patches, dispfun, dispgdfun, step):
    '''找到这个点是属于哪个patch的\n
    dispfun是色散关系的函数，dispgdfun是向费米面投影的梯度\n
    注意这个step最好小于pi / 2 * mesh
    '''
    #从这个点引出一条线，如果两端反号，则停止
    kxv, kyv = pnt.coord
    olddisp = dispfun(kxv, kyv)
    while True:
        cita = dispgdfun(kxv, kyv)
        #确定方向
        kxp = kxv + step * numpy.cos(cita)
        kyp = kyv + step * numpy.sin(cita)
        #大于pi就贴边
        if numpy.abs(kxp) > numpy.pi:
            kxp = numpy.sign(kxp) * numpy.pi
        if numpy.abs(kyp) > numpy.pi:
            kyp = numpy.sign(kyp) * numpy.pi
        newdispp = dispfun(kxp, kyp)
        #另一个方向
        kxn = kxv - step * numpy.cos(cita)
        kyn = kyv - step * numpy.sin(cita)
        if numpy.abs(kxn) > numpy.pi:
            kxn = numpy.sign(kxn) * numpy.pi
        if numpy.abs(kyn) > numpy.pi:
            kyn = numpy.sign(kyn) * numpy.pi
        newdispn = dispfun(kxn, kyn)
        #反号
        if newdispp * olddisp < 0:
            gsign = +1
            break
        if newdispn * olddisp < 0:
            gsign = -1
            break
        #看谁下降得快
        #如果p方向下降的快
        if numpy.abs(newdispn) > numpy.abs(newdispp):
            kxv, kyv = kxp, kyp
        else:#如果n方向下降的快
            kxv, kyv = kxn, kyn
        if numpy.abs(kxv) > numpy.pi or numpy.abs(kyv) > numpy.pi:
            raise ValueError('出界了')
    #现在kxv，kyv向cita方向step长度的符号是不同的
    def __disp_by_dis(dis):
        '''从pnt这个点沿着slope走dis这么长的位置上的能量'''
        xdis = kxv + dis * numpy.cos(cita)
        ydis = kyv + dis * numpy.sin(cita)
        if numpy.abs(xdis) > numpy.pi:
            xdis = numpy.sign(xdis) * numpy.pi
        if numpy.abs(ydis) > numpy.pi:
            ydis = numpy.sign(ydis) * numpy.pi
        return dispfun(xdis, ydis)
    rootd = optimize.bisect(__disp_by_dis, 0, gsign * step)
    crsx = kxv + rootd * numpy.cos(cita)
    crsy = kyv + rootd * numpy.sin(cita)
    dis_to_patch = [numpy.square(crsx - pat.coord[0]) +\
        numpy.square(crsy - pat.coord[1])\
        for pat in patches]
    return numpy.argmin(dis_to_patch)


def find_patch_old(pnt: Point, patches, dispfun, dispgdfun):
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
        #如果始终没有穿过费米面
        if numpy.abs(gues) > numpy.pi*4:
            raise ValueError(str(pnt) + str(cita) + ' 没有穿过Umklapp')
        gues += 0.01
    #这个时候gues已经反号，而gues-0.01还没有
    rootd = optimize.bisect(__disp_by_dis, (gues-0.01) * gsign, gues * gsign)
    crsx = kxv + rootd * numpy.cos(cita)
    crsy = kyv + rootd * numpy.sin(cita)
    dis_to_patch = [numpy.square(crsx - pat.coord[0]) +\
        numpy.square(crsy - pat.coord[1])\
        for pat in patches]
    return numpy.argmin(dis_to_patch)
