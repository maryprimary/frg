"""Hubbard模型用的流方程"""

#pylint: disable=pointless-string-statement, global-statement


import multiprocessing
import numpy
from basics import Point, get_procs_num
from fermi.patches import find_patch
from fermi.surface import const_energy_line_in_patches
from fermi.bubble import pi_plus_ec, pi_minus_ec



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

U = None

def uinit(initv, pnum):
    '''初始化'''
    global U
    if U is not None:
        raise RuntimeError('已经初始化过了')
    U = numpy.ndarray((pnum, pnum, pnum))
    U[:, :, :] = initv


CONFIG = None

class _Config():
    """不会变的参数\n
    这里面l开头的三个都是和剖分的小三角有关的
    """
    def __init__(self, brlu, ltris, ladjs, pinfo, lpats, disp, dispgd,\
        ksft, lamb0, find_mode=1):
        self._brlu = brlu
        self._ltris = ltris
        self._nps = numpy.sqrt(len(ltris)) / 2
        self._ladjs = ladjs
        self._pinfo = pinfo
        self._lpats = lpats
        self._disp = disp
        self._dispgd = dispgd
        self._ksft = ksft
        self._lamb0 = lamb0
        self._patchnum = len(self._pinfo)
        #找到每个对应的n4
        self._k4tab = numpy.ndarray(
            (self._patchnum, self._patchnum, self._patchnum),
            dtype=int
        )
        step = numpy.minimum(brlu.width, brlu.height) / 10 / self._nps
        for idx1 in range(self._patchnum):
            for idx2 in range(self._patchnum):
                for idx3 in range(self._patchnum):
                    kv1, kv2, kv3 = pinfo[idx1], pinfo[idx2], pinfo[idx3]
                    kv4 = ksft(ksft(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
                    #这里的投影是向Umklapp的
                    #这种方法运算速度显然比find_patch要慢
                    #故不采用
                    #dislist = [numpy.square(tri.center.coord[0] - kv4.coord[0]) +\
                    #    numpy.square(tri.center.coord[1] - kv4.coord[1])\
                    #    for tri in self._ltris]
                    #idx4_ = numpy.argmin(dislist)
                    #idx4_ = self._lpats[idx4_]
                    idx4 = find_patch(kv4, pinfo, disp, dispgd, step, mode=find_mode)
                    #assert idx4_ == idx4
                    self._k4tab[idx1, idx2, idx3] = idx4
        #初始化

    @property
    def patchnum(self):
        '''patch数量'''
        return self._patchnum

    @property
    def brlu(self):
        '''第一布里渊区'''
        return self._brlu

    @property
    def ltris(self):
        '''小三角的列表'''
        return self._ltris

    @property
    def ladjs(self):
        '''每个小三角相邻的小三角'''
        return self._ladjs

    @property
    def pinfo(self):
        '''所有patch的中心点'''
        return self._pinfo

    @property
    def lpats(self):
        '''每个小三角属于第几个patch'''
        return self._lpats

    @property
    def disp(self):
        '''色散关系'''
        return self._disp

    @property
    def dispgd(self):
        '''向费米面投影'''
        return self._dispgd

    @property
    def ksft(self):
        '''动量空间中相加'''
        return self._ksft

    @property
    def lamb0(self):
        '''lambda的初始值'''
        return self._lamb0

    @property
    def k4tab(self):
        '''从n1,n2,n3确定n4'''
        return self._k4tab


def config_init(brlu, ltris, ladjs, pinfo, lpats, disp, dispgd, ksft, lamb0):
    '''初始化配置'''
    global CONFIG
    if CONFIG is not None:
        raise RuntimeError('已经初始化过了')
    CONFIG = _Config(brlu, ltris, ladjs, pinfo, lpats, disp, dispgd, ksft, lamb0)


QUICKCONTOUR = None

def precompute_contour(lval):
    '''预先计算等能线，在后续的dl_ec的计算中（一共pnum^3次）
    每次都需要计算这个等能线的话会很浪费时间'''
    global QUICKCONTOUR, CONFIG
    if QUICKCONTOUR is None:
        QUICKCONTOUR = {}
    keys = list(QUICKCONTOUR.keys())
    for key in keys:
        del QUICKCONTOUR[key]
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    ltris, ladjs, lpats = CONFIG.ltris, CONFIG.ladjs, CONFIG.lpats
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    posi, peidx = const_energy_line_in_patches(ltris, ladjs, lpats, lamb, disp)
    nega, neidx = const_energy_line_in_patches(ltris, ladjs, lpats, -lamb, disp)
    QUICKCONTOUR[lval] = (posi, peidx, nega, neidx)


QUICKQPP = None

def precompute_qpp(lval):
    '''pp沟道的bubble和idx3是没有关系的，可以省略'''
    global QUICKQPP
    if lval not in QUICKCONTOUR:
        raise ValueError('precompute_qpp %.2f 没有提前计算contour' % lval)
    if QUICKQPP is None:
        QUICKQPP = {}
    #删除掉除了现在的所有内容
    keys = list(QUICKQPP.keys())
    for key in keys:
        del QUICKQPP[key]
    #QUICKQPP[lval] = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    #
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    area = CONFIG.brlu.width * CONFIG.brlu.height
    #所有patch上面的等能线
    posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    #切分好patch
    lnposi = []
    lnnega = []
    for nidx in range(CONFIG.patchnum):
        lnposi.append([pos for pos, eid in zip(posi, peidx) if eid == nidx])
        lnnega.append([neg for neg, eid in zip(nega, neidx) if eid == nidx])
    #pp沟道所有bubble的自由度
    data_list = []
    place_holder = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    nditer = numpy.nditer(
        place_holder,
        flags=['multi_index'])
    while not nditer.finished:
        nidx, idx1, idx2 = nditer.multi_index
        kv1, kv2 = pinfo[idx1], pinfo[idx2]
        q_pp = ksft(kv1, kv2)
        data_list.append((lnposi[nidx], lnnega[nidx], lamb, q_pp, disp, ksft, area))
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_minus_ec, data_list)
    #pi_minus_ec(nposi, nnega, lamb, q_pp, disp, ksft)
    QUICKQPP[lval] = numpy.reshape(result, place_holder.shape)


QUICKQFS = None

def precompute_qfs(lval):
    '''fs沟道的bubble和idx1是没有关系的，对idx1进行循环的时候重复计算浪费时间'''
    global QUICKQFS
    if lval not in QUICKCONTOUR:
        raise ValueError('precompute_qfs %.2f 没有提前运算等能线' % lval)
    if QUICKQFS is None:
        QUICKQFS = {}
    #清空内容
    keys = list(QUICKQFS.keys())
    for key in keys:
        del QUICKQFS[key]
    #numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    area = CONFIG.brlu.width * CONFIG.brlu.height
    #所有patch上面的等能线
    posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    #切分好patch
    lnposi = []
    lnnega = []
    for nidx in range(CONFIG.patchnum):
        lnposi.append([pos for pos, eid in zip(posi, peidx) if eid == nidx])
        lnnega.append([neg for neg, eid in zip(nega, neidx) if eid == nidx])
    #fs沟道所有bubble的自由度
    data_list = []
    place_holder = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    nditer = numpy.nditer(
        place_holder,
        flags=['multi_index'])
    while not nditer.finished:
        nidx, idx2, idx3 = nditer.multi_index
        kv2, kv3 = pinfo[idx2], pinfo[idx3]
        q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
        data_list.append((lnposi[nidx], lnnega[nidx], lamb, q_fs, disp, ksft, area))
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_plus_ec, data_list)
    #pi_plus_ec(nposi, nnega, lamb, q_fs, disp, ksft)
    QUICKQFS[lval] = numpy.reshape(result, place_holder.shape)


QUICKNQFS = None

def precompute_nqfs(lval):
    '''nfs沟道的bubble和idx1是没有关系的，对idx1进行循环的时候重复计算浪费时间'''
    global QUICKNQFS
    if lval not in QUICKCONTOUR:
        raise ValueError('precompute_nqfs %.2f 没有提前运算等能线' % lval)
    if QUICKNQFS is None:
        QUICKNQFS = {}
    #清空内容
    keys = list(QUICKNQFS.keys())
    for key in keys:
        del QUICKNQFS[key]
    #numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    area = CONFIG.brlu.width * CONFIG.brlu.height
    #所有patch上面的等能线
    posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    #切分好patch
    lnposi = []
    lnnega = []
    for nidx in range(CONFIG.patchnum):
        lnposi.append([pos for pos, eid in zip(posi, peidx) if eid == nidx])
        lnnega.append([neg for neg, eid in zip(nega, neidx) if eid == nidx])
    #nfs沟道所有bubble的自由度
    data_list = []
    place_holder = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    nditer = numpy.nditer(
        place_holder,
        flags=['multi_index'])
    while not nditer.finished:
        nidx, idx2, idx3 = nditer.multi_index
        kv2, kv3 = pinfo[idx2], pinfo[idx3]
        q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
        nq_fs = Point(-q_fs.coord[0], -q_fs.coord[1], 1)
        data_list.append((lnposi[nidx], lnnega[nidx], lamb, nq_fs, disp, ksft, area))
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_plus_ec, data_list)
    #pi_plus_ec(nposi, nnega, lamb, nq_fs, disp, ksft)
    QUICKNQFS[lval] = numpy.reshape(result, place_holder.shape)


QUICKQEX = None

def precompute_qex(lval):
    '''ex沟道的bubble和idx2是没有关系的，对idx2的循环计算浪费时间'''
    global QUICKQEX
    if lval not in QUICKCONTOUR:
        raise ValueError('precompute_qex %.2f 没有提前运算等能线' % lval)
    if QUICKQEX is None:
        QUICKQEX = {}
    #清空内容
    keys = list(QUICKQEX.keys())
    for key in keys:
        del QUICKQEX[key]
    #numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    area = CONFIG.brlu.width * CONFIG.brlu.height
    #所有patch上面的等能线
    posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    #切分好patch
    lnposi = []
    lnnega = []
    for nidx in range(CONFIG.patchnum):
        lnposi.append([pos for pos, eid in zip(posi, peidx) if eid == nidx])
        lnnega.append([neg for neg, eid in zip(nega, neidx) if eid == nidx])
    #ex沟道所有bubble的自由度
    data_list = []
    place_holder = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    nditer = numpy.nditer(
        place_holder,
        flags=['multi_index'])
    while not nditer.finished:
        nidx, idx1, idx3 = nditer.multi_index
        kv1, kv3 = pinfo[idx1], pinfo[idx3]
        q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
        data_list.append((lnposi[nidx], lnnega[nidx], lamb, q_ex, disp, ksft, area))
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_plus_ec, data_list)
    #pi_plus_ec(nposi, nnega, lamb, q_ex, disp, ksft)
    QUICKQEX[lval] = numpy.reshape(result, place_holder.shape)


QUICKNQEX = None

def precompute_nqex(lval):
    '''nex沟道的bubble和idx2是没有关系的，对idx2的循环计算浪费时间'''
    global QUICKNQEX, CONFIG, QUICKCONTOUR
    if lval not in QUICKCONTOUR:
        raise ValueError('precompute_qex %.2f 没有提前运算等能线' % lval)
    if QUICKNQEX is None:
        QUICKNQEX = {}
    #清空内容
    keys = list(QUICKNQEX.keys())
    for key in keys:
        del QUICKNQEX[key]
    #numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    area = CONFIG.brlu.width * CONFIG.brlu.height
    #所有patch上面的等能线
    posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    #切分好patch
    lnposi = []
    lnnega = []
    for nidx in range(CONFIG.patchnum):
        lnposi.append([pos for pos, eid in zip(posi, peidx) if eid == nidx])
        lnnega.append([neg for neg, eid in zip(nega, neidx) if eid == nidx])
    #nex沟道所有bubble的自由度
    data_list = []
    place_holder = numpy.ndarray((CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum))
    nditer = numpy.nditer(
        place_holder,
        flags=['multi_index'])
    while not nditer.finished:
        nidx, idx1, idx3 = nditer.multi_index
        kv1, kv3 = pinfo[idx1], pinfo[idx3]
        q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
        nq_ex = Point(-q_ex.coord[0], -q_ex.coord[1], 1)
        data_list.append((lnposi[nidx], lnnega[nidx], lamb, nq_ex, disp, ksft, area))
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_plus_ec, data_list)
    #pi_plus_ec(nposi, nnega, lamb, q_ex, disp, ksft)
    QUICKNQEX[lval] = numpy.reshape(result, place_holder.shape)


def dl_ec(lval, idx1, idx2, idx3):
    '''(10.113)中定义的对l的求导数\n
    '''
    '''
    从(10.40)出发D_L U(k1,k2;k3,k4) =
    - Int_{k} {G'(k)G(-k+Qpp) U(k1,k2;-k+Qpp,k) U(k,-k+Qpp;k3,k4)}
    - Int_{k} {G(k)G'(-k+Qpp) U(k1,k2;-k+Qpp,k) U(k,-k+Qpp;k3,k4)}
    + Int_{k} {G'(k)G(k-Qfs) 2*U(k1,k-Qfs;k,k4) U(k2,k;k-Qfs,k3)}
    + Int_{k} {G(k)G'(k-Qfs) 2*U(k1,k-Qfs;k,k4) U(k2,k;k-Qfs,k3)}
    - Int_{k} {G'(k)G(k-Qfs) U(k1,k-Qfs;k4,k) U(k2,k;k-Qfs,k3)}
    - Int_{k} {G(k)G'(k-Qfs) U(k1,k-Qfs;k4,k) U(k2,k;k-Qfs,k3)}
    - Int_{k} {G'(K)G(k-Qfs) U(k1,k-Qfs;k,k4) U(k2,k;k3,k-Qfs)}
    - Int_{k} {G(k)G'(k-Qfs) U(k1,k-Qfs;k,k4) U(k2,k;k3,k-Qfs)}
    - Int_{k} {G'(k)G(k-Qex) U(k1,k-Qex;k3,k) U(k,k2;k-Qex,k4)}
    - Int_{k} {G(k)G'(k-Qex) U(k1,k-Qex;k3,k) U(k,k2;k-Qex,k4)}
    积分换元把G()G'()合并进来
    - Int_{k} { G'(k)G(-k+Qpp) //k'=-k+Qpp，k=-k'+Qpp所以换完还是-k+Qpp，但是第二项顺序变了
    [U(k1,k2;-k+Qpp,k) U(k,-k+Qpp;k3,k4) + U(k1,k2;k,-k+Qpp) U(-k+Qpp,k;k3,k4)]}
    + Int_{k} { G'(k)G(k-Qfs) 2*U(k1,k-Qfs;k,k4) U(k2,k;k-Qfs,k3) }
    // k'=k-Qfs，k=k'+Qfs，Qfs = k3 - k2 = k1 - k4
    + Int_{k} { G'(k)G(k+Qfs) 2*U(k1,k;k+Qfs,k4) U(k2,k+Qfs;k,k3) }
    - Int_{k} { G'(k)G(k-Qfs) U(k1,k-Qfs;k4,k) U(k2,k;k-Qfs,k3) }
    - Int_{k} { G'(k)G(k+Qfs) U(k1,k;k4,k+Qfs) U(k2,k+Qfs;k,k3) }
    - Int_{k} { G'(K)G(k-Qfs) U(k1,k-Qfs;k,k4) U(k2,k;k3,k-Qfs) }
    - Int_{k} { G'(k)G(k+Qfs) U(k1,k;k+Qfs,k4) U(k2,k+Qfs;k3,k) }
    - Int_{k} { G'(k)G(k-Qex) U(k1,k-Qex;k3,k) U(k,k2;k-Qex,k4) }
    // k'=k-Qex，k=k'+Qex，Qex = k4 - k2 = k1 - k3
    - Int_{k} { G'(k)G(k+Qex) U(k1,k;k3,k+Qex) U(k+Qex,k2;k,k4)}
    利用U本身的对称性，U(k1,k2;k3,k4) = U(k2,k1;k4,k3) = U(k4,k3,k2,k1)
    *第二个等式是时空反演对称性保证的*
    把包含Qpp, Qfs, Qex的项全都放到后面去
    - Int_{k} { G'(k)G(-k+Qpp)
    [U(k2,k1;k,-k+Qpp) U(k3,k4;k,-k+Qpp) + U(k1,k2;k,-k+Qpp) U(k4,k3;k,-k+Qpp)]}
    + Int_{k} { G'(k)G(k-Qfs) 2*U(k,k4;k1,k-Qfs) U(k,k2;k3,k-Qfs) }
    + Int_{k} { G'(k)G(k+Qfs) 2*U(k,k1;k4,k+Qfs) U(k,k3;k2,k+Qfs) }
    - Int_{k} { G'(k)G(k-Qfs) U(k4,k;k1,k-Qfs) U(k,k2;k3,k-Qfs) }
    - Int_{k} { G'(k)G(k+Qfs) U(k1,k;k4,k+Qfs) U(k,k3;k2,k+Qfs) }
    - Int_{k} { G'(K)G(k-Qfs) U(k,k4;k1,k-Qfs) U(k2,k;k3,k-Qfs) }
    - Int_{k} { G'(k)G(k+Qfs) U(k,k1;k4,k+Qfs) U(k3,k;k2,k+Qfs) }
    - Int_{k} { G'(k)G(k-Qex) U(k3,k;k1,k-Qex) U(k2,k;k4,k-Qex) }
    - Int_{k} { G'(k)G(k+Qex) U(k1,k;k3,k+Qex) U(k4,k;k2,k+Qex) }
    这时，利用U在角度上变化快，径向变化小的特点，给动量空间分成patches，有
    U(k1,k2;k3,k4) = u(n1,n2;n3,n4) 其中k1属于第n1个patch，以此类推，整个patch中的U都
    是一样大的，这样，左边的
    D_L U(k1,k2;k3,k4) = D_L u(n1,n2;n3,n4)
    这时的n1,n2,n3,n4的patch的中心加起来不一定是动量守恒的，因为k4投到k_n4，移动了
    可以通过前三个来求出k4，很可能不在费米面上。但所有满足动量守恒的
    U(k1,k2,k3,k4)也等于u(n1,n2,n3,n4)，（四个k不见得在patch中心），所以求u就好了
    右边的积分中，对于每一个patch，U都是常数u，从积分中提取出来
    D_L u(n1,n2;n3,n4) =
    + sum_{n}{[u(n2,n1;n,n(-k+Qpp))u(n3,n4;n,n(-k+Qpp)) + u(n1,n2;n,n(-k+Qpp))u(n4,n3;n,n(-k+Qpp))]
    Int_{k in kn} { - G'(k)G(-k+Qpp) } }
    + sum_{n}{ 2*u(n,n4;n1,n(k-Qfs)) u(n,n2;n3,n(k-Qfs)) Int_{k in kn} {G'(k)G(k-Qfs)}}
    + sum_{n}{ 2*u(n,n1;n4,n(k+Qfs)) u(n,n3;n2,n(k+Qfs)) Int_{k in kn} {G'(k)G(k+Qfs)}}
    - sum_{n}{ u(n4,n;n1,n(k-Qfs)) u(n,n2;n3,n(k-Qfs)) Int_{k in kn} {G'(k)G(k-Qfs)}}
    - sum_{n}{ u(n1,n;n4,n(k+Qfs)) u(n,n3;n2,n(k+Qfs)) Int_{k in kn} {G'(k)G(k+Qfs)}}
    - sum_{n}{ u(n,n4;n1,n(k-Qfs)) u(n2,n;n3,n(k-Qfs)) Int_{k in kn} {G'(K)G(k-Qfs)}}
    - sum_{n}{ u(n,n1;n4,n(k+Qfs)) u(n3,n;n2,n(k+Qfs)) Int_{k in kn} {G'(k)G(k+Qfs)}}
    - sum_{n}{ u(n3,n;n1,n(k-Qex)) u(n2,n;n4,n(k-Qex)) Int_{k in kn} {G'(k)G(k-Qex)}}
    - sum_{n}{ u(n1,n;n3,n(k+Qex)) u(n4,n;n2,n(k+Qex)) Int_{k in kn} {G'(k)G(k+Qex)}}
    }
    通过n1,n2,n3中心的k1,k2,k3可以确定k4，从而确定n4。所以从u里面去掉最后一个
    然后等式左右两边同时乘一个 -LAMBDA， LAMBDA = LAMBDA0 * exp(-l) ,D_l = -LAMBDA D_L
    D_l u(n1,n2,n3) = - {
    +sum_{n}{ PI_MINUS(n, Qpp)
    [u(n2,n1;n) u(n3,n4;n) + u(n1,n2;n) u(n4,n3;n)]}
    +sum_{n}{ 2*u(n,n4;n1) u(n,n2;n3) PI_PLUS(n,Qfs) }
    +sum_{n}{ 2*u(n,n1;n4) u(n,n3;n2) PI_PLUS(n,-Qfs) }
    -sum_{n}{ u(n4,n;n1) u(n,n2;n3) PI_PLUS(n,Qfs) }
    -sum_{n}{ u(n1,n;n4) u(n,n3;n2) PI_PLUS(n,-Qfs) }
    -sum_{n}{ u(n,n4;n1) u(n2,n;n3) PI_PLUS(n,Qfs) }
    -sum_{n}{ u(n,n1;n4) u(n3,n;n2) PI_PLUS(n,-Qfs) }
    -sum_{n}{ u(n3,n;n1) u(n2,n;n4) PI_PLUS(n,Qex) }
    -sum_{n}{ u(n1,n;n3) u(n4,n;n2) PI_PLUS(n,-Qex) }
    }
    其中定义了
    PI_MINUS(n, q) = - LAMBDA * Int_{k in kn} {G'(k)G(-k+q)}
    PI_PLUS(n, q) = LAMBDA * Int_{k in kn} {G'(k)G(k-q)}
    注意上面的PI_MINUS, PI_PLUS, u都是包含了对LAMBDA或者u的依赖的
    这样就有了完整的方程
    '''
    global CONFIG, U
    global QUICKCONTOUR
    global QUICKQPP
    global QUICKQFS, QUICKNQFS
    global QUICKQEX, QUICKNQEX
    pinfo = CONFIG.pinfo
    ksft = CONFIG.ksft
    disp, dispgd = CONFIG.disp, CONFIG.dispgd
    kv1, kv2, kv3 = pinfo[idx1], pinfo[idx2], pinfo[idx3]
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    #左边u的第四个idx
    idx4 = CONFIG.k4tab[idx1, idx2, idx3]
    #确定三个沟道的动量
    #q_pp = ksft(kv1, kv2)
    #q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
    #nq_fs = Point(-q_fs.coord[0], -q_fs.coord[1], 1)
    #q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
    #nq_ex = Point(-q_ex.coord[0], -q_ex.coord[1], 1)
    #用来数值计算的量
    ltris, ladjs, lpats = CONFIG.ltris, CONFIG.ladjs, CONFIG.lpats
    if lval in QUICKCONTOUR:
        posi, peidx, nega, neidx = QUICKCONTOUR[lval]
    else:
        print('%.2f没有提前计算等能线' % lval)
        raise RuntimeError('%s' % __file__)
        #posi, peidx = const_energy_line_in_patches(ltris, ladjs, lpats, lamb, disp)
        #nega, neidx = const_energy_line_in_patches(ltris, ladjs, lpats, -lamb, disp)
    #
    value = 0.
    for nidx, _ in enumerate(pinfo, 0):
        #第n个patch上的PI函数需要用到属于n的两个等能线
        #nposi = [pos for pos, eid in zip(posi, peidx) if eid == nidx]
        #nnega = [neg for neg, eid in zip(nega, neidx) if eid == nidx]
        if lval not in QUICKQPP:
            raise RuntimeError('没有提前计算QPP %.2f' % lval)
            #pi_min_n_q_pp =\
            #    pi_minus_ec(nposi, nnega, lamb, q_pp, disp, ksft)
        pi_min_n_q_pp = QUICKQPP[lval][nidx, idx1, idx2]
        if lval not in QUICKQFS:
            raise RuntimeError('没有提前计算QFS %.2f' % lval)
            #pi_plu_n_q_fs =\
            #    pi_plus_ec(nposi, nnega, lamb, q_fs, disp, ksft)
        pi_plu_n_q_fs = QUICKQFS[lval][nidx, idx2, idx3]
        if lval not in QUICKNQFS:
            raise RuntimeError('没有提前计算NQFS %.2f' % lval)
            #pi_plu_n_nq_fs =\
            #    pi_plus_ec(nposi, nnega, lamb, nq_fs, disp, ksft)
        pi_plu_n_nq_fs = QUICKNQFS[lval][nidx, idx2, idx3]
        if lval not in QUICKQEX:
            raise RuntimeError('没有提前计算QEX %.2f' % lval)
            #pi_plu_n_q_ex =\
            #    pi_plus_ec(nposi, nnega, lamb, q_ex, disp, ksft)
        pi_plu_n_q_ex = QUICKQEX[lval][nidx, idx1, idx3]
        if lval not in QUICKNQEX:
            raise RuntimeError('没有提前计算NQEX %.2f' % lval)
            #pi_plu_n_nq_ex =\
            #    pi_plus_ec(nposi, nnega, lamb, nq_ex, disp, ksft)
        pi_plu_n_nq_ex = QUICKNQEX[lval][nidx, idx1, idx3]
        #方程
        value += pi_min_n_q_pp * (
            U[idx2, idx1, nidx] * U[idx3, idx4, nidx] + U[idx1, idx2, nidx] * U[idx4, idx3, nidx]
        )
        value += pi_plu_n_q_fs * 2 * U[nidx, idx4, idx1] * U[nidx, idx2, idx3]
        value += pi_plu_n_nq_fs *2 * U[nidx, idx1, idx4] * U[nidx, idx3, idx2]
        value -= pi_plu_n_q_fs * U[idx4, nidx, idx1] * U[nidx, idx2, idx3]
        value -= pi_plu_n_nq_fs* U[idx1, nidx, idx4] * U[nidx, idx3, idx2]
        value -= pi_plu_n_q_fs * U[nidx, idx4, idx1] * U[idx2, nidx, idx3]
        value -= pi_plu_n_nq_fs* U[nidx, idx1, idx4] * U[idx3, nidx, idx2]
        value -= pi_plu_n_q_ex * U[idx3, nidx, idx1] * U[idx2, nidx, idx4]
        value -= pi_plu_n_nq_ex* U[idx1, nidx, idx3] * U[idx4, nidx, idx2]
    #改成对l求导后的负号
    value = -value
    return value
