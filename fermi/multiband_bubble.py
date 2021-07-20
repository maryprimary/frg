"""定义在10.112中的bubble integrals
"""

import warnings
import numpy
from basics import Point
from basics.point import middle_point
#pylint: disable=pointless-string-statement


#warnings.simplefilter('once', RuntimeWarning)


def pi_ab_plus_ec(posia, negaa, lamb, qval, dispb, ksft, area):
    '''使用能量cutoff作为flow parameter的bubble\n
    posi是dispa为+LAMBDA的边，nega是dispa为-LAMBDA的边, lamb是LAMBDA\n
    dispb是第二个色散关系，qval是需要平移的大小，应该用一个Point来包装，\n
    kshf是动量相加的函数, 这个函数应该能处理好到第一布里渊区的映射\n
    ```(10.112)本身已经处理好了动量守恒，k, k-q是需要满足动量守恒的关系的，而处理好```
    ```k-q到第一布里渊区的映射就处理好了Umklapp```
    '''
    '''
    10.112中的 PI^+(n, q) = +LAMBDA (2pi)^-2 beta^-1 Int_{k in k_n} G'(k)G(k - Q)
    其中有一个beta是频率积分带来的，2pi^2是动量积分带来的
    G(k)=CITA(LAMBDA < abs(disp(k))) / i*omega - disp(k)
    G'(k)=-DELTA(abs(disp(k))-LAMBDA) / i*omege - disp(k)
    在零温的情况下10.112中的频率部分可以积分出来，此后的k都是不包含频率的
    = +LAMBDA (2pi)^-2 Int_{k in k_n} CITA() -DELTA()
    { beta^-1 sum_{omega} [(i*omega-disp(k))(i*omega-disp(k - q))]^-1 }
    花括号中的内容求和完之后等于 - CITA(-disp(k)disp(k-q)) / (abs(disp(k)) + abs(disp(k-p)))
    积分会变成
    = +LAMBDA (2pi)^-2 Int_{k in k_n} DELTA(abs(disp(k))-LAMBDA) CITA(LAMBDA<abs(disp(k-q)))
    CITA(-disp(k)disp(k-q)) / (abs(disp(k)) + abs(disp(k-p)))
    因为采用的能量cutoff中有一个 DELTA(abs(disp(k))-LAMBDA)，disp(k)等于正的或者负的LAMBDA
    而CITA(-disp(k)disp(k-q))限制了disp(k)和disp(k-q)符号相反
    所以上式变成
    (第一项disp(k)=LAMBDA>0，于是disp(k-q)<0，而且abs(disp(k))=-disp(k)>LAMBDA)
    (第二项类似，分子中的abs(disp(k))都可以直接换成LAMBDA，abs(disp(k-q))也都知道符号)
    = +LAMBDA (2pi)^-2 Int_{k in kn} {
        DELTA(disp(k)-LAMBDA)CITA(-disp(k-q)-LAMBDA) / (LAMBDA - disp(k - q))
        DELTA(disp(k)+LAMBDA)CITA(disp(k-q)-LAMBDA) / (LAMBDA + disp(k - q)) }
    还可以从积分里面把DELTA给积分掉，这样对于二维平面的积分也会变成对
    disp(k) = LAMBDA 或者 -LAMBDA的线的积分
    = +LAMBDA (2pi)^-2 *
    [Int_{disp(k) = +LAMBDA} CITA(-disp(k-q)-LAMBDA) / (LAMBDA - disp(k - q))]
    +[Int_{disp(k) = -LAMBDA} CITA(disp(k-q)-LAMBDA) / (LAMBDA + disp(k - q))  ]
    '''
    nega_q = Point(-qval.coord[0], -qval.coord[1], 1)
    #积分正LAMBDA的线
    intposi = 0.
    for edg in posia:
        kval = middle_point(edg.ends[0], edg.ends[1])
        kprim = ksft(kval, nega_q)
        #CITA
        disp_kprim = dispb(kprim.coord[0], kprim.coord[1])
        if -disp_kprim < lamb:
            continue
        #线积分，计算线元的长度
        intposi += edg.length / (lamb - disp_kprim)
    #积分负LAMBDA的线
    intnega = 0.
    for edg in negaa:
        kval = middle_point(edg.ends[0], edg.ends[1])
        kprim = ksft(kval, nega_q)
        #CITA
        disp_kprim = dispb(kprim.coord[0], kprim.coord[1])
        if disp_kprim < lamb:
            continue
        intnega += edg.length / (lamb + disp_kprim)
    #乘上系数
    result = lamb * (intposi + intnega) / area#numpy.square(numpy.pi*2)
    return result


def pi_ab_minus_ec(posia, negaa, lamb, qval, dispb, ksft, area):
    '''使用能量cutoff作为flow parameter的bubble\n
    posi是dispa为+LAMBDA的边，nega是dispa为-LAMBDA的边, lamb是LAMBDA\n
    这两个边应该是限制在dispa这个带的第n个patch中的，这两个边也就暗含了n\n
    dispa和dispb是两个带的色散关系\n
    qval是需要平移的大小，应该用一个Point来包装，\n
    kshf是动量相加的函数, 这个函数应该能处理好到第一布里渊区的映射\n
    ```(10.112)本身已经处理好了动量守恒，k, k-q是需要满足动量守恒的关系的，而处理好```
    ```k-q到第一布里渊区的映射就处理好了Umklapp```
    '''
    '''
    10.112中的 PI^-(n, q) = -LAMBDA (2pi)^-2 beta^-1 Int_{k in k_n} G'(k)G(- k + Q)
    = -LAMBDA (2pi)^-2 Int_{k in k_n} CITA() -DELTA()
    { beta^-1 sum_{omega} [(i*omega-disp(k))(-i*omega-disp(-k + q))]^-1 }
    在零温下这个频率积分等于，注意-k那里把频率也给反过来了
    +CITA(+disp(k)disp(-k+q)) / (abs(disp(k)) + abs(disp(-k+q)))
    原式就等于
    = LAMBDA (2pi)^-2 Int_{k in k_n} {
        DELTA(abs(disp(k))-LAMBDA) CITA(abs(disp(-k+q)-LAMBDA))
        CITA(disp(k)disp(-k+q)) / (abs(disp(k)) + abs(disp(-k+q))) }
    第二个CITA限制了disp(k)和disp(-k+q)同号，积分积掉DELTA，分类讨论正负
    = LAMBDA (2pi)^-2 {
        Int_{disp(k) = +LAMBDA} CITA(disp(-k+q) - LAMBDA) / (LAMBDA + disp(-k+q)) +
        Int_{disp(k) = -LAMBDA} CITA(-disp(-k+q) -LAMBDA) / (LAMBDA - disp(-k+q))
    }
    '''
    #积分正LAMBDA的线
    intposi = 0.
    for edg in posia:
        kval = middle_point(edg.ends[0], edg.ends[1])
        nega_k = Point(-kval.coord[0], -kval.coord[1], 1)
        kprim = ksft(nega_k, qval)
        #CITA
        disp_kprim = dispb(kprim.coord[0], kprim.coord[1])
        if disp_kprim < lamb:
            continue
        #要计算线元的长度
        intposi += edg.length / (lamb + disp_kprim)
    #积分负LAMBDA的线
    intnega = 0.
    for edg in negaa:
        kval = middle_point(edg.ends[0], edg.ends[1])
        nega_k = Point(-kval.coord[0], -kval.coord[1], 1)
        kprim = ksft(nega_k, qval)
        #CITA
        disp_kprim = dispb(kprim.coord[0], kprim.coord[1])
        if -disp_kprim < lamb:
            continue
        intnega += edg.length / (lamb - disp_kprim)
    #乘上系数
    result = lamb * (intposi + intnega) / area#numpy.square(numpy.pi*2)
    return result


def pi_ab_plus_tf(ltris, tarea, lamb, dispa, dispb, qval, ksft, area):
    '''温度流的+
    这里的lamb就是T，ltris中的所有三角都应该要在同一个patch中,
    tarea是每个小三角形的面积，dispa是和k相关的那个能带，dispb是k-q相关的
    '''
    nega_q = Point(-qval.coord[0], -qval.coord[1], 1)
    result = 0.
    for tri in ltris:
        #这个小三角形的k值
        kval = tri.center
        #k-q
        kprim = ksft(kval, nega_q)
        #epsilon_k
        eps_k = dispa(kval.coord[0], kval.coord[1])
        #epsilon_{k-q}
        eps_kp = dispb(kprim.coord[0], kprim.coord[1])
        if numpy.abs(eps_k - eps_kp) < 1.e-10:
            #如果特别小，可以利用
            # lim (eps_k -> eps_kp) Pi^{+} =
            # 1/T (e^{eps/T} (-eps/T*e^{eps/T} + eps/T + e^{eps/T} + 1)) / (e^{eps/T} + 1)^3
            bval = eps_kp / lamb
            #如果本身就很大，分母会比较大导致接近0
            if bval > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                return 0.
            expb = numpy.exp(bval)
            num = expb * (-bval * expb + bval + expb + 1)
            den = numpy.power((1+expb), 3)
            d_val = num / den / lamb
        else:
            if (eps_k / lamb) > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                num_left = 0.
            else:
                #exp^{epsilon_k / T}
                exp_k_t = numpy.exp(eps_k / lamb)
                num_left = eps_k / lamb * exp_k_t / numpy.square(1 + exp_k_t)
            if (eps_kp / lamb) > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                num_righ = 0.
            else:
                #e^{epsilon_{k-q} / T}
                exp_kp_t = numpy.exp(eps_kp / lamb)
                num_righ = eps_kp / lamb * exp_kp_t\
                    / numpy.square(1 + exp_kp_t)
            d_val = (num_left - num_righ) / (eps_k - eps_kp)
        result += d_val * tarea
    result = result / area
    return result


def pi_ab_minus_tf(ltris, tarea, lamb, dispa, dispb, qval, ksft, area):
    '''温度流的-
    这里的lamb就是T，ltris中的所有三角都应该要在同一个patch中,
    tarea是每个小三角形的面积，dispa是和k相关的那个能带，dispb是-k+q相关的
    '''
    result = 0.
    for tri in ltris:
        #这个小三角形的k值
        kval = tri.center
        nega_k = Point(-kval.coord[0], -kval.coord[1], 1)
        #-k+q
        kprim = ksft(nega_k, qval)
        #epsilon_k
        eps_k = dispa(kval.coord[0], kval.coord[1])
        #-epsilon_{-k+q}
        neps_kp = -dispb(kprim.coord[0], kprim.coord[1])
        #这个时候，因为epsilon_{-k+q}前面已经有了负号，分母上还是负号
        if numpy.abs(eps_k - neps_kp) < 1.e-10:
            #如果两个数值比较接近, Pi^{-}和Pi^{+}的公式完全一样，就是第二个能量要加个负号
            # lim (eps_k -> -eps_kp) Pi^{-} =
            # 1/T (e^{eps/T} (-eps/T*e^{eps/T} + eps/T + e^{eps/T} + 1)) / (e^{eps/T} + 1)^3
            bval = eps_k / lamb
            if bval > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                return 0.
            expb = numpy.exp(bval)
            num = expb * (-bval * expb + bval + expb + 1)
            den = numpy.power((1+expb), 3)
            d_val = num / den / lamb
        else:
            if (eps_k / lamb) > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                num_left = 0.
            else:
                #e^{epsilon_k / T}
                exp_k_t = numpy.exp(eps_k / lamb)
                num_left = eps_k / lamb * exp_k_t / numpy.square(1 + exp_k_t)
            if (neps_kp / lamb) > 25:
                warnings.warn("数值不稳定", RuntimeWarning)
                num_righ = 0.
            else:
                #e^{-epsilon_{-k+q} / T}
                exp_nkp_t = numpy.exp(neps_kp / lamb)
                num_righ = neps_kp / lamb * exp_nkp_t\
                / numpy.square(1 + exp_nkp_t)
            d_val = (num_left - num_righ) / (eps_k - neps_kp)
        result += d_val * tarea
    result = result / area
    return result


def val_test(eps_k, neps_kp, lamb):
    #如果两个数值比较接近, Pi^{-}和Pi^{+}的公式完全一样，就是第二个能量要加个负号
    # lim (eps_k -> -eps_kp) Pi^{-} =
    # 1/T (e^{eps/T} (-eps/T*e^{eps/T} + eps/T + e^{eps/T} + 1)) / (e^{eps/T} + 1)^3
    bval = eps_k / lamb
    expb = numpy.exp(bval)
    num = expb * (-bval * expb + bval + expb + 1)
    den = numpy.power((1+expb), 3)
    d_val1 = num / den / lamb
    #
    #e^{epsilon_k / T}
    exp_k_t = numpy.exp(eps_k / lamb)
    #e^{-epsilon_{-k+q} / T}
    exp_nkp_t = numpy.exp(neps_kp / lamb)
    #
    num_left = eps_k / lamb * exp_k_t / numpy.square(1 + exp_k_t)
    num_righ = neps_kp / lamb * exp_nkp_t\
    / numpy.square(1 + exp_nkp_t)
    #e^{epsilon_k / T}
    exp_k_t = numpy.exp(eps_k / lamb)
    d_val2 = (num_left - num_righ) / (eps_k - neps_kp)
    return d_val1, d_val2
