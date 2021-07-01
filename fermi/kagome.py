r"""Kagome格子相关的内容
``````
Kagome是三角晶系，定义和triangle.py中类似
初基矢量的选择
a1 = (-1/2, \sqrt{3}/2)
a2 = (1/2, \sqrt{3}/2)
倒格子的选择
b1 = (-2\pi, 2\pi/\sqrt{3})  #(-1, 1/\sqrt{3})
b2 = ( 2\pi, 2\pi/\sqrt{3})  #(1, 1/\sqrt{3})

特别需要注意的是数值精度，使用sympy运算的时候精度更高一些
或者使用更高精度的float128
"""


import numpy
import sympy
from scipy import optimize
from basics import Hexagon, Point
from basics.point import get_absolute_angle, middle_point


def brillouin():
    '''布里渊区'''
    return Hexagon(Point(0, 0, 1), 2*numpy.pi/numpy.sqrt(3))


def get_high_symmetry_point():
    '''获得三角晶格的高对称点'''
    gamma = Point(0, 0, 1)
    mpt = Point(0, 2*numpy.pi / (numpy.sqrt(3)), 1)
    kpt = Point(-2*numpy.pi / 3, 2*numpy.pi / (numpy.sqrt(3)), 1)
    return gamma, mpt, kpt


def s_disp(kxv, kyv):#pylint: disable=unused-argument
    '''s能带的色散关系'''
    return 2


def p_disp(kxv, kyv):
    '''p能带的色散关系'''
    xval = kxv / 4
    yval = numpy.sqrt(3) * kyv / 4
    sqr = 2 * numpy.cos(2*xval - 2*yval)
    sqr += 2 * numpy.cos(2*xval + 2*yval)
    sqr += 2 * numpy.cos(4*xval) + 3
    if sqr < 0:
        if numpy.isclose(sqr, 0):
            sqr = 0
        else:
            raise RuntimeError("能带有错误")
    return - 1 + numpy.sqrt(sqr) - 0.11

def d_disp(kxv, kyv):
    '''d能带的色散关系'''
    xval = kxv / 4
    yval = numpy.sqrt(3) * kyv / 4
    sqr = 2 * numpy.cos(2*xval - 2*yval)
    sqr += 2 * numpy.cos(2*xval + 2*yval)
    sqr += 2 * numpy.cos(4*xval) + 3
    if sqr < 0:
        if numpy.isclose(sqr, 0):
            sqr = 0
        else:
            raise RuntimeError("能带有错误")
    return - 1 - numpy.sqrt(sqr)


def get_nu_numpy(kxv, kyv):
    '''获取nu变换的矩阵'''
    #pylint: disable=invalid-name
    nu1 = numpy.zeros(3)
    nu2 = numpy.zeros(3)
    nu3 = numpy.zeros(3)
    x = kxv / numpy.square(2, dtype=numpy.float128)
    y = numpy.sqrt(3, dtype=numpy.float128) * kyv / 4
    #p和d需要的根号下的数值
    sqr = 2 * numpy.cos(2*x - 2*y)
    sqr += 2 * numpy.cos(2*x + 2*y)
    sqr += 2 * numpy.cos(4*x) + 3
    if sqr < 0:
        if numpy.isclose(sqr, 0):
            sqr = 0
        else:
            raise RuntimeError("能带有错误")
    #第一组本征态
    nu1[0] = -numpy.cos(2*x)/2 + numpy.cos(2*y)/2
    nu1[1] = -numpy.cos(x-y)/2 + numpy.cos(3*x+y)/2
    nu1[2] = 1 - numpy.square(numpy.cos(x+y))
    #归一化
    n1 = numpy.square(nu1[0]) + numpy.square(nu1[1]) + numpy.square(nu1[2])
    n1 = 1.0 / numpy.sqrt(n1)
    nu1 = n1 * nu1
    #第二组本征态
    lamb2 = 1 - numpy.sqrt(sqr)
    nu2[0] = numpy.square(lamb2)/2 - 2*numpy.square(numpy.cos(x-y))
    nu2[1] = 2*numpy.cos(2*x)*numpy.cos(x-y) + numpy.cos(x+y)*lamb2
    nu2[2] = numpy.cos(2*x)*lamb2 + 2*numpy.cos(x-y)*numpy.cos(x+y)
    #归一化
    n2 = numpy.square(nu2[0]) + numpy.square(nu2[1]) + numpy.square(nu2[2])
    n2 = 1.0 / numpy.sqrt(n2)
    nu2 = n2 * nu2
    #第三组本征态
    lamb3 = 1 + numpy.sqrt(sqr)
    nu3[0] = numpy.square(lamb3)/2 - 2*numpy.square(numpy.cos(x-y))
    nu3[1] = 2*numpy.cos(2*x)*numpy.cos(x-y) + numpy.cos(x+y)*lamb3
    nu3[2] = numpy.cos(2*x)*lamb3 + 2*numpy.cos(x-y)*numpy.cos(x+y)
    #归一化
    n3 = numpy.square(nu3[0]) + numpy.square(nu3[1]) + numpy.square(nu3[2])
    n3 = 1.0 / numpy.sqrt(n3)
    nu3 = n3 * nu3
    return nu1, nu2, nu3


def get_nu(kxv, kyv):
    '''获取nu变换的矩阵'''
    #pylint: disable=invalid-name
    nu1 = sympy.zeros(3, 1)
    nu2 = sympy.zeros(3, 1)
    nu3 = sympy.zeros(3, 1)
    #x = kxv / numpy.square(2, dtype=numpy.float128)
    #y = numpy.sqrt(3, dtype=numpy.float128) * kyv / 4
    x, y = sympy.symbols("x y")
    #x = x2 / 4.0
    #y = sympy.sqrt(3) * y2 / 4.0
    #p和d需要的根号下的数值
    sqr = 2 * sympy.cos(2*x - 2*y)
    sqr += 2 * sympy.cos(2*x + 2*y)
    sqr += 2 * sympy.cos(4*x) + 3
    #第一组本征态
    nu1[0] = -sympy.cos(2*x)/2 + sympy.cos(2*y)/2
    nu1[1] = -sympy.cos(x-y)/2 + sympy.cos(3*x+y)/2
    nu1[2] = 1 - sympy.Pow(sympy.cos(x+y), 2)
    #归一化
    n1 = sympy.Pow(nu1[0], 2) + sympy.Pow(nu1[1], 2) + sympy.Pow(nu1[2], 2)
    n1 = 1.0 / sympy.sqrt(n1)
    nu1 = n1 * nu1
    #第二组本征态
    lamb2 = 1 - sympy.sqrt(sqr)
    nu2[0] = sympy.Pow(lamb2, 2)/2 - 2*sympy.Pow(sympy.cos(x-y), 2)
    nu2[1] = 2*sympy.cos(2*x)*sympy.cos(x-y) + sympy.cos(x+y)*lamb2
    nu2[2] = sympy.cos(2*x)*lamb2 + 2*sympy.cos(x-y)*sympy.cos(x+y)
    #归一化
    n2 = sympy.Pow(nu2[0], 2) + sympy.Pow(nu2[1], 2) + sympy.Pow(nu2[2], 2)
    n2 = 1.0 / sympy.sqrt(n2)
    nu2 = n2 * nu2
    #第三组本征态
    lamb3 = 1 + sympy.sqrt(sqr)
    nu3[0] = sympy.Pow(lamb3, 2)/2 - 2*sympy.Pow(sympy.cos(x-y), 2)
    nu3[1] = 2*sympy.cos(2*x)*sympy.cos(x-y) + sympy.cos(x+y)*lamb3
    nu3[2] = sympy.cos(2*x)*lamb3 + 2*sympy.cos(x-y)*sympy.cos(x+y)
    #归一化
    n3 = sympy.Pow(nu3[0], 2) + sympy.Pow(nu3[1], 2) + sympy.Pow(nu3[2], 2)
    n3 = 1.0 / sympy.sqrt(n3)
    nu3 = n3 * nu3
    #整理成numpy的结果
    nnu1 = nu1.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu2 = nu2.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu3 = nu3.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu1 = numpy.array(nnu1, dtype=numpy.float64).reshape([3])
    nnu2 = numpy.array(nnu2, dtype=numpy.float64).reshape([3])
    nnu3 = numpy.array(nnu3, dtype=numpy.float64).reshape([3])
    return nnu1, nnu2, nnu3


def band_u(uval, pinfos):
    '''带内的相互作用'''
    npat = len(pinfos)
    ret = numpy.zeros((npat, npat, npat))
    ndit = numpy.nditer(ret, flags=['multi_index'])
    while not ndit.finished:
        k1idx, k2idx, k3idx = ndit.multi_index
        #p2这个带
        k1v = pinfos[k1idx]
        k2v = pinfos[k2idx]
        k3v = pinfos[k3idx]
        k4v = shift_kv(k1v, k2v)
        k4v = shift_kv(k4v, Point(-k3v.coord[0], -k3v.coord[1], 1))
        #p这个带只需要第二个
        _, k1nu, _ = get_nu_numpy(k1v.coord[0], k1v.coord[1])
        _, k2nu, _ = get_nu_numpy(k2v.coord[0], k2v.coord[1])
        _, k3nu, _ = get_nu_numpy(k3v.coord[0], k3v.coord[1])
        _, k4nu, _ = get_nu_numpy(k4v.coord[0], k4v.coord[1])
        #计算数值
        for idx in range(3):
            ret[k1idx, k2idx, k3idx] +=\
                uval * k1nu[idx] * k2nu[idx] * k3nu[idx] * k4nu[idx]
        ndit.iternext()
    return ret


def band_v(vval, pinfos):
    '''带内的相互作用'''
    npat = len(pinfos)
    ret = numpy.zeros((npat, npat, npat))
    ndit = numpy.nditer(ret, flags=['multi_index'])
    while not ndit.finished:
        k1idx, k2idx, k3idx = ndit.multi_index
        #p2这个带
        k1v = pinfos[k1idx]
        k2v = pinfos[k2idx]
        k3v = pinfos[k3idx]
        k4v = shift_kv(k1v, k2v)
        k4v = shift_kv(k4v, Point(-k3v.coord[0], -k3v.coord[1], 1))
        #公式中的K
        ckv = shift_kv(k3v, Point(-k2v.coord[0], -k2v.coord[1], 1))
        #p这个带只需要第二个
        _, k1nu, _ = get_nu_numpy(k1v.coord[0], k1v.coord[1])
        _, k2nu, _ = get_nu_numpy(k2v.coord[0], k2v.coord[1])
        _, k3nu, _ = get_nu_numpy(k3v.coord[0], k3v.coord[1])
        _, k4nu, _ = get_nu_numpy(k4v.coord[0], k4v.coord[1])
        #计算数值
        s3_4 = 0.25*numpy.sqrt(3)
        vab = k1nu[0]*k2nu[1]*k3nu[1]*k4nu[0] + k1nu[1]*k2nu[0]*k3nu[0]*k4nu[1]
        ret[k1idx, k2idx, k3idx] += 2 * vval * \
            numpy.cos(0.25*ckv.coord[0] + s3_4*ckv.coord[1]) * vab
        vac = k1nu[0]*k2nu[2]*k3nu[2]*k4nu[0] + k1nu[2]*k2nu[0]*k3nu[0]*k4nu[2]
        ret[k1idx, k2idx, k3idx] += 2 * vval * \
            numpy.cos(0.5*ckv.coord[0]) * vac
        vbc = k1nu[1]*k2nu[2]*k3nu[2]*k4nu[1] + k1nu[2]*k2nu[1]*k3nu[1]*k4nu[2]
        ret[k1idx, k2idx, k3idx] += 2 * vval * \
            numpy.cos(-0.25*ckv.coord[0] + s3_4*ckv.coord[1]) * vbc
        ndit.iternext()
    return ret


def get_sublattice_u(umat, pinfos):
    '''将能带表示变回到子格子表示'''
    npat = len(pinfos)
    ret = numpy.zeros((3, 3, 3, 3, npat, npat, npat))
    place_holder = numpy.zeros((npat, npat, npat))
    ndit = numpy.nditer(place_holder, flags=['multi_index'])
    while not ndit.finished:
        k1idx, k2idx, k3idx = ndit.multi_index
        #p2这个带
        k1v = pinfos[k1idx]
        k2v = pinfos[k2idx]
        k3v = pinfos[k3idx]
        k4v = shift_kv(k1v, k2v)
        k4v = shift_kv(k4v, Point(-k3v.coord[0], -k3v.coord[1], 1))
        _, k1nu, _ = get_nu_numpy(k1v.coord[0], k1v.coord[1])
        _, k2nu, _ = get_nu_numpy(k2v.coord[0], k2v.coord[1])
        _, k3nu, _ = get_nu_numpy(k3v.coord[0], k3v.coord[1])
        _, k4nu, _ = get_nu_numpy(k4v.coord[0], k4v.coord[1])
        place_holder2 = numpy.zeros((3, 3, 3, 3))
        ndit2 = numpy.nditer(place_holder2, flags=['multi_index'])
        while not ndit2.finished:
            si1, si2, si3, si4 = ndit2.multi_index
            ret[si1, si2, si3, si4, k1idx, k2idx, k3idx] =\
                umat[k1idx, k2idx, k3idx] * k1nu[si1] * k2nu[si2] * k3nu[si3] * k4nu[si4]
            ndit2.iternext()
        ndit.iternext()
    return ret


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


def get_von_hove_patches(npat):
    '''获取平分von Hove的patches'''
    #pylint: disable=cell-var-from-loop
    #一共有6个M点
    mpts = [
        Point(3.1415926, 3.1415926 / 1.7320508, 1),
        Point(0, 2*3.1415926 / 1.7320508, 1),
        Point(-3.1415926, 3.1415926 / 1.7320508, 1),
        Point(-3.1415926, -3.1415926 / 1.7320508, 1),
        Point(0, -2*3.1415926 / 1.7320508, 1),
        Point(3.1415926, -3.1415926 / 1.7320508, 1)
    ]
    angs = []
    pat_per_k = npat // 6
    gap = 1 / pat_per_k
    for idx in range(6):
        ridx = idx + 1 if idx + 1 < 6 else 0
        for pidx in range(pat_per_k):
            omega = (pidx + 0.5) * gap
            vpt = middle_point(mpts[idx], mpts[ridx], sc1=1-omega, sc2=omega)
            angs.append(get_absolute_angle(vpt.coord[0], vpt.coord[1]))
    pats = []
    for ang in angs:
        rrad = optimize.bisect(
            lambda rad: p_disp(rad*numpy.cos(ang), rad*numpy.sin(ang)),
            0, 2*numpy.pi/numpy.sqrt(3)
        )
        pats.append(Point(rrad*numpy.cos(ang), rrad*numpy.sin(ang), 1))
    return pats


def check_patches_converge(pat):
    '''
    检测这个点的nu值是不是合理的数值
    '''
    nu11, nu12, nu13 = get_nu(pat.coord[0], pat.coord[1])
    nu21, nu22, nu23 = get_nu(pat.coord[0]+0.01, pat.coord[1]+0.01)
    nu1 = numpy.abs(numpy.stack([nu11, nu12, nu13]))
    nu2 = numpy.abs(numpy.stack([nu21, nu22, nu23]))
    return numpy.allclose(nu1, nu2, rtol=0., atol=0.1)



def get_patches(nang, nrad):
    '''PHYSICAL REVIEW B92, 155137 (2015)
    这里面的patches取法，从K点为中心
    '''
    raise NotImplementedError()
