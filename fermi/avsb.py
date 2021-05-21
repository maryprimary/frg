"""
AV3Sb5的有效哈密顿量
arxiv:2104.05671v1
一个两带的Kagome
"""


import numpy
#使用sympy的任意精度算数
import sympy
from scipy import optimize
from basics import Hexagon, Point



def brillouin():
    '''布里渊区'''
    return Hexagon(Point(0, 0, 1), 2*numpy.pi/numpy.sqrt(3))


def get_nu(kxv, kyv):
    '''获取nu变换的矩阵'''
    #pylint: disable=invalid-name
    nu1 = sympy.zeros(3, 1)
    nu2 = sympy.zeros(3, 1)
    nu3 = sympy.zeros(3, 1)
    x, y = sympy.symbols("x y")
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
    #转换成numpy数组
    nnu1 = nu1.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu2 = nu2.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu3 = nu3.evalf(30, subs={x: kxv/4, y: numpy.sqrt(3)*kyv/4})
    nnu1 = numpy.array(nnu1, dtype=numpy.float64).reshape([3])
    nnu2 = numpy.array(nnu2, dtype=numpy.float64).reshape([3])
    nnu3 = numpy.array(nnu3, dtype=numpy.float64).reshape([3])
    return nnu1, nnu2, nnu3
