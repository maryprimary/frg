"""计算费米面上点的位置"""


import numpy
from scipy import optimize
from basics import Square, Point
from basics.point import get_absolute_angle


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


def find_patch(
        pnt: Point, patches, dispfun, dispgdfun, step,
        brlim=(numpy.pi, numpy.pi), mode=1
    ):
    '''找到这个点是属于哪个patch的\n
    dispfun是色散关系的函数，dispgdfun是向费米面投影的梯度\n
    注意这个step最好小于pi / 2 * mesh\n
    brlim是布里渊区的边界，这里是最大的绝对值\n
    如果mode=1，就是向费米面投影然后找patch的算法，如果mode=2，
    就是直接找最近的一个点的算法
    '''
    if mode == 2:
        return find_patch_mode2(pnt, patches)
    if mode == 3:
        return find_patch_mode3(pnt, patches)
    #从这个点引出一条线，如果两端反号，则停止
    kxv, kyv = pnt.coord
    olddisp = dispfun(kxv, kyv)
    while True:
        cita = dispgdfun(kxv, kyv)
        #print(kxv, kyv, cita)
        #确定方向
        kxp = kxv + step * numpy.cos(cita)
        kyp = kyv + step * numpy.sin(cita)
        #大于pi就贴边
        if numpy.abs(kxp) > brlim[0]:
            kxp = numpy.sign(kxp) * brlim[0]
        if numpy.abs(kyp) > brlim[1]:
            kyp = numpy.sign(kyp) * brlim[1]
        newdispp = dispfun(kxp, kyp)
        #另一个方向
        kxn = kxv - step * numpy.cos(cita)
        kyn = kyv - step * numpy.sin(cita)
        if numpy.abs(kxn) > brlim[0]:
            kxn = numpy.sign(kxn) * brlim[0]
        if numpy.abs(kyn) > brlim[1]:
            kyn = numpy.sign(kyn) * brlim[1]
        newdispn = dispfun(kxn, kyn)
        #反号，有些时候会直接碰到0，这个时候，如果是old等于0了，那么gsign无关紧要，
        #如果是新的等于零，那么朝它的方向也是对的
        if newdispp * olddisp <= 0:
            gsign = +1
            break
        if newdispn * olddisp <= 0:
            gsign = -1
            break
        #看谁下降得快
        #如果p方向下降的快
        if numpy.abs(newdispn) > numpy.abs(newdispp):
            kxv, kyv = kxp, kyp
        else:#如果n方向下降的快
            kxv, kyv = kxn, kyn
        #print(newdispn, newdispp)
        #raise
        if numpy.abs(kxv) > brlim[0] or numpy.abs(kyv) > brlim[1]:
            raise ValueError('出界了')
    #现在kxv，kyv向cita方向step长度的符号是不同的
    def __disp_by_dis(dis):
        '''从pnt这个点沿着slope走dis这么长的位置上的能量'''
        xdis = kxv + dis * numpy.cos(cita)
        ydis = kyv + dis * numpy.sin(cita)
        if numpy.abs(xdis) > brlim[0]:
            xdis = numpy.sign(xdis) * brlim[0]
        if numpy.abs(ydis) > brlim[1]:
            ydis = numpy.sign(ydis) * brlim[1]
        return dispfun(xdis, ydis)
    rootd = optimize.bisect(__disp_by_dis, 0, gsign * step)
    crsx = kxv + rootd * numpy.cos(cita)
    crsy = kyv + rootd * numpy.sin(cita)
    dis_to_patch = [numpy.square(crsx - pat.coord[0]) +\
        numpy.square(crsy - pat.coord[1])\
        for pat in patches]
    return numpy.argmin(dis_to_patch)


def find_patch_mode2(pnt: Point, patches):
    '''直接寻找最近的patch'''
    dislist = []
    for pat in patches:
        dis = numpy.square(pat.coord[0] - pnt.coord[0])
        dis += numpy.square(pat.coord[1] - pnt.coord[1])
        dislist.append(dis)
    return numpy.argmin(dislist)


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


def find_patch_mode3(pnt: Point, patches):
    '''在六角型的布里渊区中找到属于哪个patches
    适合von Hove filling附近, patches也需要是按照von Hove的
    费米面平分的, 而且必须是偶数乘以6
    '''
    pat_per_k = len(patches) // 6
    if numpy.mod(pat_per_k, 2) != 0:
        raise ValueError("必须是偶数×6")
    kxv, kyv = pnt.coord[0], pnt.coord[1]
    ang = get_absolute_angle(kxv, kyv)
    #首先看这个点是不是在nesting的小六边形中
    isinner = False
    kyp = numpy.sqrt(3) * kyv / 2.0
    eng = -2*numpy.cos(kxv) - 4*numpy.cos(kyp)*numpy.cos(kxv/2)
    if eng <= 2.0:
        isinner = True
    #如果不在中间，那么找到合适的K点
    gap = numpy.pi / 6
    idx = ang / gap
    idx = int(numpy.floor(idx))
    mline = None
    kline = (kyv / (kxv+1e-10), 0)
    targets = None
    if idx == 1 or idx == 2:
        #这一条线的方程是 y = - x / sqrt(3) + 2pi / sqrt(3)
        #这个K点的坐标（2pi/3, 2pi / sqrt(3）)
        targets = patches[0:pat_per_k]
        mline = (-1/numpy.sqrt(3), 2*numpy.pi/numpy.sqrt(3))
        if not isinner:
            slope = (2*numpy.pi / numpy.sqrt(3) - kyv) / (2*numpy.pi/3 - kxv + 1e-10)
            inc = numpy.pi * 2 / numpy.sqrt(3) - slope * numpy.pi * 2 / 3
            kline = (slope, inc)
        start_patch = 0
    elif idx == 3 or idx == 4:
        #这一条线的方程是 y = x / sqrt(3) + 2pi / sqrt(3)
        #这个K点的坐标（-2pi/3, 2pi / sqrt(3）)
        targets = patches[pat_per_k:2*pat_per_k]
        mline = (1/numpy.sqrt(3), 2*numpy.pi/numpy.sqrt(3))
        if not isinner:
            slope = (2*numpy.pi / numpy.sqrt(3) - kyv) / (-2*numpy.pi/3 - kxv + 1e-10)
            inc = numpy.pi * 2 / numpy.sqrt(3) + slope * numpy.pi * 2 / 3
            kline = (slope, inc)
        start_patch = pat_per_k
    elif idx == 5 or idx == 6:
        #这一条线的方程是 x = -pi
        #这个K点的坐标（-4pi/3, 0)
        crsx = -numpy.pi
        targets = patches[2*pat_per_k:3*pat_per_k]
        #print(targets)
        #raise
        if not isinner:
            slope = kyv / (kxv + 4*numpy.pi/3 + 1e-10)
            inc = slope * 4 * numpy.pi / 3
            kline = (slope, inc)
        crsy = -kline[0] * numpy.pi + kline[1]
        idx = [numpy.square(tar.coord[0] - crsx)\
            + numpy.square(tar.coord[1] - crsy)  for tar in targets]
        return 2*pat_per_k + numpy.argmin(idx)
    elif idx == 7 or idx == 8:
        #这一条线的方程是 y = -x / sqrt(3) - 2pi / sqrt(3)
        #这个K点的坐标（-2pi/3, -2pi / sqrt(3）)
        targets = patches[3*pat_per_k:4*pat_per_k]
        mline = (-1/numpy.sqrt(3), -2*numpy.pi/numpy.sqrt(3))
        if not isinner:
            slope = (kyv + 2*numpy.pi / numpy.sqrt(3)) / (kxv + 2*numpy.pi/3 + 1e-10)
            inc = slope*2*numpy.pi/3 - 2*numpy.pi / numpy.sqrt(3)
            kline = (slope, inc)
        start_patch = 3*pat_per_k
    elif idx == 9 or idx == 10:
        #这一条线的方程是 y = x / sqrt(3) - 2pi / sqrt(3)
        #这个K点的坐标（2pi/3, -2pi / sqrt(3）)
        targets = patches[4*pat_per_k:5*pat_per_k]
        mline = (1/numpy.sqrt(3), -2*numpy.pi/numpy.sqrt(3))
        if not isinner:
            slope = (kyv + 2*numpy.pi / numpy.sqrt(3)) / (kxv - 2*numpy.pi/3 + 1e-10)
            inc = -2*numpy.pi / numpy.sqrt(3) - slope*2*numpy.pi/3
            kline = (slope, inc)
        start_patch = 4*pat_per_k
    else:
        #这个时候idx == 11 or idx == 0
        #这一条线的方程是 x = pi
        #这个K点的坐标（4pi/3, 0)
        targets = patches[5*pat_per_k:]
        crsx = numpy.pi
        if not isinner:
            slope = kyv / (kxv - 4*numpy.pi/3 + 1e-10)
            inc = -slope * 4 * numpy.pi / 3
            kline = (slope, inc)
        crsy = kline[0] * numpy.pi + kline[1]
        idx = [numpy.square(tar.coord[0] - crsx)\
            + numpy.square(tar.coord[1] - crsy)  for tar in targets]
        return 5*pat_per_k + numpy.argmin(idx)
    crsx = (kline[1] - mline[1]) / (mline[0] - kline[0])
    crsy = mline[0] * crsx + mline[1]
    idx = [numpy.square(tar.coord[0] - crsx)\
        + numpy.square(tar.coord[1] - crsy)  for tar in targets]
    return start_patch + numpy.argmin(idx)
