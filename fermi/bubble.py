"""定义在10.112中的bubble integrals
"""

import numpy
from basics import Point
from basics.point import middle_point
#pylint: disable=pointless-string-statement

def pi_plus_ec(posi, nega, lamb, qval, disp, ksft):
    '''使用能量cutoff作为flow parameter的bubble\n
    posi是能量为+LAMBDA的边，nega是能量为-LAMBDA的边, lamb是LAMBDA\n
    disp是色散关系，qval是需要平移的大小，应该用一个Point来包装，\n
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
    for edg in posi:
        kval = middle_point(edg.ends[0], edg.ends[1])
        kprim = ksft(kval, nega_q)
        #CITA
        disp_kprim = disp(kprim.coord[0], kprim.coord[1])
        if -disp_kprim < lamb:
            continue
        #线积分，计算线元的长度
        intposi += edg.length / (lamb - disp_kprim)
    #积分负LAMBDA的线
    intnega = 0.
    for edg in nega:
        kval = middle_point(edg.ends[0], edg.ends[1])
        kprim = ksft(kval, nega_q)
        #CITA
        disp_kprim = disp(kprim.coord[0], kprim.coord[1])
        if disp_kprim < lamb:
            continue
        intnega += edg.length / (lamb + disp_kprim)
    #乘上系数
    result = lamb * (intposi + intnega) / numpy.square(numpy.pi*2)
    return result


def pi_minus_ec(posi, nega, lamb, qval, disp, ksft):
    '''使用能量cutoff作为flow parameter的bubble\n
    posi是能量为+LAMBDA的边，nega是能量为-LAMBDA的边, lamb是LAMBDA\n
    disp是色散关系，qval是需要平移的大小，应该用一个Point来包装，\n
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
    for edg in posi:
        kval = middle_point(edg.ends[0], edg.ends[1])
        nega_k = Point(-kval.coord[0], -kval.coord[1], 1)
        kprim = ksft(nega_k, qval)
        #CITA
        disp_kprim = disp(kprim.coord[0], kprim.coord[1])
        if disp_kprim < lamb:
            continue
        #要计算线元的长度
        intposi += edg.length / (lamb + disp_kprim)
    #积分负LAMBDA的线
    intnega = 0.
    for edg in nega:
        kval = middle_point(edg.ends[0], edg.ends[1])
        nega_k = Point(-kval.coord[0], -kval.coord[1], 1)
        kprim = ksft(nega_k, qval)
        #CITA
        disp_kprim = disp_kprim(kprim.coord[0], kprim.coord[1])
        if -disp_kprim < lamb:
            continue
        intnega += edg.length / (lamb - disp_kprim)
    #乘上系数
    result = lamb * (intposi + intnega) / numpy.square(numpy.pi*2)
    return result
