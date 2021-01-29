"""
Hamiltonian = H_hop + H_int\n
H_hop = sum_{k,s} disp(k) C^+_{k,s} C_{k,s}，t的负号出现在disp之中\n
(*) H_int = sum_{k1,k2,k3,k4} {U C^+_{k1,u}C^+_{k2,d}C_{k3,d}C_{k4,u} +\n
0.5 U C^+_{k1,u}C^+_{k2,u}C_{k3,u}C_{k4,u} + 0.5 U u -> d}\n
换成Tau的标记\n
H_int = 0.25 sum_{k1s1,k2s2,k3s3,k4s4} {
    T(k1s1,k2s2;k2s3,k4s4) C^+_{k1s1}C^+_{k2s2}C_{k3s3}C_{k4s3}
}\n
可以推得\n
Tau(1,2;3,4) = Delta(s1,s4)Delta(s2,s3) U(1,2;3,4) +
Delta(s1,s3)Delta(s2,s4) U(1,2;4,3)\n
这里的U就是上面的(*)中的U
Hubbard模型（只有onsite的相互作用）中U没有动量的依赖\n
"""

import numpy

U = None

def uinit(initv, pnum):
    '''初始化'''
    global U
    if U is not None:
        raise RuntimeError('已经初始化过了')
    U = numpy.ndarray((pnum, pnum, pnum))
    U[:, :, :] = initv
