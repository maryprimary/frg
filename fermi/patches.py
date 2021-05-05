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
