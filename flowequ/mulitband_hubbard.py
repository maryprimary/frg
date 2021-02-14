"""多带Hubbard模型用的流方程"""

#pylint: disable=pointless-string-statement


import multiprocessing
import numpy
from basics import Point
from fermi.patches import find_patch
from fermi.surface import const_energy_line_in_patches
from fermi.multiband_bubble import pi_ab_plus_ec, pi_ab_minus_ec


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

"""
和单带不同的是，多带的k中除了(omega, k)，还有一个带的指标(omega, k, b)\n
"""

U = None

def uinit(initv, pnum, bnum):
    '''初始化u，这时的initv不应该是简单的一个数字了\n
    '''
    global U
    if U is not None:
        raise ValueError('已经初始化过了')
    #前面三个动量可以随便取，最后一个动量是固定的，所在的能带不固定
    U = numpy.zeros((bnum, bnum, bnum, bnum, pnum, pnum, pnum))
    for sh1, sh2 in zip(U.shape, initv.shape):
        if sh1 != sh2:
            raise ValueError('维度不一致')
    U += initv


CONFIG = None

class _Config():
    """不会变的参数\n
    现在m个能带，这m个能带之间不能有相互交叉的单体格林函数\n
    这里面l开头的三个都是和剖分的小三角有关的，ltris是所有的小三角的list\n
    ladjs是每个小三角的相邻小三角的idx，ladjs[n] = (i1, i2, i3),这里面的n\n
    就是ltris中的第n个小三角\n
    现在mpinfo是[m, num_patches]的一个二维数组，每个能带有num_patches个patch\n
    mlpats是[m, len(ltris)]，对于每个能带来说，每个patch的区域是不一样的\n
    mdisp是[m, func]，每个能带的色散\n
    mdispgd是没个能带的色散的梯度\n
    ksft是动量相加的函数\n
    lamb0是初始值\n
    """
    def __init__(self, ltris, ladjs, mpinfo, mlpats, \
        mdisp, mdispgd, ksft, lamb0):
        global U
        self._ltris = ltris
        self._nps = numpy.sqrt(len(ltris)) / 2
        self._bandnum = len(mpinfo)
        self._ladjs = ladjs
        self._mpinfo = mpinfo
        self._mlpats = mlpats
        self._mdisp = mdisp
        self._mdispgd = mdispgd
        self._ksft = ksft
        self._lamb0 = lamb0
        self._patchnum = len(self._mpinfo[0])
        #找到每个对应的n4
        #现在每个带都有自己的idx4
        #self._mk4tab = numpy.ndarray(U.shape, dtype=numpy.int)
        pool = multiprocessing.Pool(4)
        data_list = []
        idxit = numpy.nditer(U, flags=['multi_index'])
        while not idxit.finished:
            bd1, bd2, bd3, bd4, idx1, idx2, idx3 = idxit.multi_index
            kv1, kv2, kv3 = mpinfo[bd1, idx1], mpinfo[bd2, idx2], mpinfo[bd3, idx3]
            kv4 = ksft(ksft(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
            data_list.append(
                (kv4, mpinfo[bd4, :], mdisp[bd4], mdispgd[bd4],\
                    numpy.pi / 2 / self._nps)
            )
            #idx4 = find_patch(kv4, mpinfo[bd4, :], mdisp[bd4],\
            #    mdispgd[bd4], numpy.pi / 2 / self._nps)
            #self._mk4tab[idxit.multi_index] = idx4
            idxit.iternext()
        self._mk4tab = pool.starmap(find_patch, data_list)
        self._mk4tab = numpy.reshape(self._mk4tab, U.shape)
        #for idx1 in range(self._patchnum):
        #    for idx2 in range(self._patchnum):
        #        for idx3 in range(self._patchnum):
        #            kv1, kv2, kv3 = pinfo[idx1], pinfo[idx2], pinfo[idx3]
        #            kv4 = ksft(ksft(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
        #            #每个能带都会把会把k4投影到自己的费米面上
        #            for idxb in range(self._bandnum):
        #                idx4 = find_patch(kv4, pinfo, disp, dispgd, numpy.pi / 2 / self._nps)
        #                self._k4tab[idx1, idx2, idx3] = idx4
        #初始化

    @property
    def ltris(self):
        '''小三角的列表'''
        return self._ltris

    @property
    def ladjs(self):
        '''每个小三角相邻的小三角'''
        return self._ladjs

    @property
    def bandnum(self):
        '''带的数量'''
        return self._bandnum

    @property
    def patchnum(self):
        '''patch的数量'''
        return self._patchnum

    @property
    def mpinfo(self):
        '''所有patch的中心点'''
        return self._mpinfo

    @property
    def mlpats(self):
        '''每个小三角属于第几个patch'''
        return self._mlpats

    @property
    def mdisp(self):
        '''色散关系'''
        return self._mdisp

    @property
    def mdispgd(self):
        '''向费米面投影'''
        return self._mdispgd

    @property
    def ksft(self):
        '''动量空间中相加'''
        return self._ksft

    @property
    def lamb0(self):
        '''lambda的初始值'''
        return self._lamb0

    @property
    def mk4tab(self):
        '''从n1,n2,n3确定n4'''
        return self._mk4tab

def config_init(ltris, ladjs, mpinfo, mlpats, mdisp, mdispgd, ksft, lamb0):
    '''初始化配置'''
    global CONFIG
    if CONFIG is not None:
        raise RuntimeError('已经初始化过了')
    CONFIG = _Config(ltris, ladjs, mpinfo, mlpats, mdisp, mdispgd, ksft, lamb0)


def dl_ec(lval, bd1, bd2, bd3, bd4, idx1, idx2, idx3):
    '''(10.113)中定义的对l的求导数\n
    '''
    global CONFIG, U
    mpinfo = CONFIG.mpinfo
    ksft = CONFIG.ksft
    #
    mdisp, mdispgd = CONFIG.mdisp, CONFIG.mdispgd
    kv1, kv2, kv3 = mpinfo[bd1, idx1], mpinfo[bd2, idx2], mpinfo[bd3, idx3]
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    #
    idx4 = CONFIG.mk4tab[bd1, bd2, bd3, bd4, idx1, idx2, idx3]
    #确定三个沟道的动量
    q_pp = ksft(kv1, kv2)
    q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
    nq_fs = Point(-q_fs.coord[0], -q_fs.coord[1], 1)
    q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
    nq_ex = Point(-q_ex.coord[0], -q_ex.coord[1], 1)
    #
    ltris, ladjs = CONFIG.ltris, CONFIG.ladjs
    mlpats = CONFIG.mlpats
    #每个带都有一条等能线
    mposi = []
    mpeidx = []
    mnega = []
    mneidx = []
    for bdi in range(CONFIG.bandnum):
        lpats = mlpats[bdi, :]
        disp = mdisp[bdi]
        posi, peidx = const_energy_line_in_patches(ltris, ladjs, lpats, lamb, disp)
        nega, neidx = const_energy_line_in_patches(ltris, ladjs, lpats, -lamb, disp)
        mposi.append(posi)
        mpeidx.append(peidx)
        mnega.append(nega)
        mneidx.append(neidx)
    #
    value = 0.
    #需要被循环的变量
    idxs = numpy.ndarray((CONFIG.bandnum, CONFIG.bandnum, CONFIG.patchnum))
    idxit = numpy.nditer(idxs, flags=['multi_index'])
    while not idxit.finished:
        alpha, beta, nidx = idxit.multi_index
        #对第alpha个能带上的第ndix个patch进行积分，这个patch上的
        #U都是一个数，bubble需要通过等能线积分出来
        posi = mposi[alpha]
        peidx = mpeidx[alpha]
        nega = mnega[alpha]
        neidx = mneidx[alpha]
        anposi = [pos for pos, eid in zip(posi, peidx) if eid == nidx]
        annega = [neg for neg, eid in zip(nega, neidx) if eid == nidx]
        pi_min_ab_n_qpp =\
            pi_ab_minus_ec(anposi, annega, lamb, q_pp, mdisp[beta], ksft)
        pi_plu_ab_n_qfs =\
            pi_ab_plus_ec(anposi, annega, lamb, q_fs, mdisp[beta], ksft)
        pi_plu_ab_n_nqfs =\
            pi_ab_plus_ec(anposi, annega, lamb, nq_fs, mdisp[beta], ksft)
        pi_plu_ab_n_qex =\
            pi_ab_plus_ec(anposi, annega, lamb, q_ex, mdisp[beta], ksft)
        pi_plu_ab_n_nqex =\
            pi_ab_plus_ec(anposi, annega, lamb, nq_ex, mdisp[beta], ksft)
        value += U[bd2, bd1, alpha, beta, idx2, idx1, nidx] *\
            U[bd3, bd4, alpha, beta, idx3, idx4, nidx] * pi_min_ab_n_qpp
        value += U[bd1, bd2, alpha, beta, idx1, idx2, nidx] *\
            U[bd4, bd3, alpha, beta, idx4, idx3, nidx] * pi_min_ab_n_qpp
        value += 2*U[alpha, bd4, bd1, beta, nidx, idx4, idx1] *\
            U[alpha, bd2, bd3, beta, nidx, idx2, idx3] * pi_plu_ab_n_qfs
        value += 2*U[alpha, bd1, bd4, beta, nidx, idx1, idx4] *\
            U[alpha, bd3, bd2, beta, nidx, idx3, idx2] * pi_plu_ab_n_nqfs
        value -= U[bd4, alpha, bd1, beta, idx4, nidx, idx1] *\
            U[alpha, bd2, bd3, beta, nidx, idx2, idx3] * pi_plu_ab_n_qfs
        value -= U[bd1, alpha, bd4, beta, idx1, nidx, idx4] *\
            U[alpha, bd3, bd2, beta, nidx, idx3, idx2] * pi_plu_ab_n_nqfs
        value -= U[alpha, bd4, bd1, beta, nidx, idx4, idx1] *\
            U[bd2, alpha, bd3, beta, idx2, nidx, idx3] * pi_plu_ab_n_qfs
        value -= U[alpha, bd1, bd4, beta, nidx, idx1, idx4] *\
            U[bd3, alpha, bd2, beta, idx3, nidx, idx2] * pi_plu_ab_n_nqfs
        value -= U[bd3, alpha, bd1, beta, idx3, nidx, idx1] *\
            U[bd2, alpha, bd4, beta, idx2, nidx, idx4] * pi_plu_ab_n_qex
        value -= U[bd1, alpha, bd3, beta, idx1, nidx, idx3] *\
            U[bd4, alpha, bd2, beta, idx4, nidx, idx2] * pi_plu_ab_n_nqex
        idxit.iternext()
    #总的负号
    value = -value
    return value
