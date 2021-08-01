"""多带Hubbard模型用的流方程"""

#pylint: disable=pointless-string-statement, global-statement

import multiprocessing
import numpy
from basics import Point, get_procs_num
from fermi.patches import find_patch_mode2, find_patch_mode3, find_patch
from fermi.multiband_bubble import pi_ab_plus_tf, pi_ab_minus_tf


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

class _Config_Tf():#pylint: disable=invalid-name, too-many-instance-attributes
    """温度流\n
    现在m个能带，这m个能带之间不能有相互交叉的单体格林函数\n"""
    def __init__(self, brlu, ltris, ladjs, mpinfo, mlpats, \
        mdisp, mdispgd, ksft, lamb0, find_mode):
        self._brlu = brlu
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
        if find_mode == 1:
            data_list = []
            idxit = numpy.nditer(U, flags=['multi_index'])
            step = numpy.minimum(brlu.width, brlu.height) / 10 / self._nps
            while not idxit.finished:
                bd1, bd2, bd3, bd4, idx1, idx2, idx3 = idxit.multi_index
                kv1, kv2, kv3 = mpinfo[bd1, idx1], mpinfo[bd2, idx2], mpinfo[bd3, idx3]
                kv4 = ksft(ksft(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
                data_list.append(
                    (kv4, mpinfo[bd4, :], mdisp[bd4], mdispgd[bd4], step)
                )
                #idx4 = find_patch(kv4, mpinfo[bd4, :], mdisp[bd4],\
                #    mdispgd[bd4], numpy.pi / 2 / self._nps)
                #self._mk4tab[idxit.multi_index] = idx4
                idxit.iternext()
            with multiprocessing.Pool(get_procs_num()) as pool:
                self._mk4tab = pool.starmap(find_patch, data_list)
            self._mk4tab = numpy.reshape(self._mk4tab, U.shape)
            return
        data_list = []
        idxit = numpy.nditer(U, flags=['multi_index'])
        while not idxit.finished:
            bd1, bd2, bd3, bd4, idx1, idx2, idx3 = idxit.multi_index
            kv1, kv2, kv3 = mpinfo[bd1, idx1], mpinfo[bd2, idx2], mpinfo[bd3, idx3]
            kv4 = ksft(ksft(kv1, kv2), Point(-kv3.coord[0], -kv3.coord[1], 1))
            data_list.append(
                (kv4, mpinfo[bd4, :])
            )
            idxit.iternext()
        find_func = {
            2: find_patch_mode2,
            3: find_patch_mode3
        }[find_mode]
        with multiprocessing.Pool(get_procs_num()) as pool:
            self._mk4tab = pool.starmap(find_func, data_list)
        self._mk4tab = numpy.reshape(self._mk4tab, U.shape)

    @property
    def brlu(self):
        '''布里渊区'''
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


def config_init(brlu, ltris, ladjs, mpinfo, mlpats, mdisp, mdispgd,\
    ksft, lamb0, find_mode=1):
    '''初始化配置'''
    global CONFIG
    if CONFIG is not None:
        raise RuntimeError('已经初始化过了')
    CONFIG = _Config_Tf(brlu, ltris, ladjs, mpinfo,\
        mlpats, mdisp, mdispgd, ksft, lamb0, find_mode=find_mode)


PATTRIS = None

def precompute_patches():
    '''提前计算好patch中的块'''
    global PATTRIS
    if PATTRIS is not None:
        raise ValueError("已经调用过了")
    #找到每个n对应的ltris
    mlpats = CONFIG.mlpats
    ltris = CONFIG.ltris
    PATTRIS = numpy.ndarray((CONFIG.bandnum, CONFIG.patchnum), dtype=object)
    #这个时候的npatch是按照alpha带的来区分的，
    #这个时候被近似的是和（k,alpha）相关的那个动量
    for bidx in range(CONFIG.bandnum):
        lpats = mlpats[bidx]
        for tri, pat in zip(ltris, lpats):
            #如果这个还没有，增加一个新的[]
            if PATTRIS[bidx, pat] is None:
                PATTRIS[bidx, pat] = []
            PATTRIS[bidx, pat].append(tri)


def config_reset_ltris(ltris, ladjs, mlpats):
    '''重新设置ltris'''
    global CONFIG, PATTRIS
    if CONFIG is None:
        raise RuntimeError('还未初始化')
    PATTRIS = None
    CONFIG._ltris = ltris
    CONFIG._ladjs = ladjs
    CONFIG._mlpats = mlpats


QUICKQPP = None

def precompute_qpp(lval):
    '''Pi^{-}实际上不需要7个指标，只需要
    bd1, idx1, bd2, idx2
    另外的alpha，beta，nidx在所有情况下都是需要遍历的，所以可以提前计算出来，节约计算时间
    '''
    global QUICKQPP
    data_list = []
    place_holder = numpy.ndarray(
        (   #alpha, beta, bd1, bd2
            CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum,
            #nidx(alpha), idx1(bd1), idx2(bd2)
            CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum
        )
    )
    nditer = numpy.nditer(
        place_holder, flags=['multi_index']
    )
    #
    area = CONFIG.brlu.width * CONFIG.brlu.height
    ltris = CONFIG.ltris
    tarea = area / len(ltris)
    mpinfo = CONFIG.mpinfo
    ksft = CONFIG.ksft
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    mdisp = CONFIG.mdisp
    #
    while not nditer.finished:
        alpha, beta, bd1, bd2, nidx, idx1, idx2 = nditer.multi_index
        kv1, kv2 = mpinfo[bd1, idx1], mpinfo[bd2, idx2]
        q_pp = ksft(kv1, kv2)
        #
        data_list.append(
            (PATTRIS[alpha, nidx], tarea, lamb,\
                mdisp[alpha], mdisp[beta], q_pp, ksft, area)
        )
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_ab_minus_tf, data_list)
    QUICKQPP = (lval, numpy.reshape(result, place_holder.shape))


QUICKQFS = None

def precompute_qfs(lval):
    '''计算q_fs实际上也不需要多余的变量
    bd2, bd3, idx2, idx3
    '''
    global QUICKQFS
    #fs的所有自由度
    data_list = []
    place_holder = numpy.ndarray(
        (   #alpha, beta, bd2, bd3
            CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum,
            #nidx(alpha), idx2(bd2), idx3(bd3)
            CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum
        )
    )
    nditer = numpy.nditer(
        place_holder, flags=['multi_index']
    )
    #
    area = CONFIG.brlu.width * CONFIG.brlu.height
    ltris = CONFIG.ltris
    tarea = area / len(ltris)
    mpinfo = CONFIG.mpinfo
    ksft = CONFIG.ksft
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    mdisp = CONFIG.mdisp
    #
    while not nditer.finished:
        alpha, beta, bd2, bd3, nidx, idx2, idx3 = nditer.multi_index
        kv2, kv3 = mpinfo[bd2, idx2], mpinfo[bd3, idx3]
        q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
        data_list.append(
            (PATTRIS[alpha, nidx], tarea, lamb,\
                mdisp[alpha], mdisp[beta], q_fs, ksft, area)
        )
        nditer.iternext()
    #
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_ab_plus_tf, data_list)
    QUICKQFS = (lval, numpy.reshape(result, place_holder.shape))


QUICKQEX = None

def precompute_qex(lval):
    '''bd1, bd3, idx1, idx3
    '''
    global QUICKQEX
    #ex的所有自由度
    data_list = []
    place_holder = numpy.ndarray(
        (   #alpha, beta, bd1, bd3
            CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum, CONFIG.bandnum,
            #nidx(alpha), idx1(bd1), idx3(bd3)
            CONFIG.patchnum, CONFIG.patchnum, CONFIG.patchnum
        )
    )
    nditer = numpy.nditer(
        place_holder, flags=['multi_index']
    )
    #
    area = CONFIG.brlu.width * CONFIG.brlu.height
    ltris = CONFIG.ltris
    tarea = area / len(ltris)
    mpinfo = CONFIG.mpinfo
    ksft = CONFIG.ksft
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    mdisp = CONFIG.mdisp
    #
    while not nditer.finished:
        alpha, beta, bd1, bd3, nidx, idx1, idx3 = nditer.multi_index
        kv1, kv3 = mpinfo[bd1, idx1], mpinfo[bd3, idx3]
        q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
        data_list.append(
            (PATTRIS[alpha, nidx], tarea, lamb,\
                mdisp[alpha], mdisp[beta], q_ex, ksft, area)
        )
        nditer.iternext()
    with multiprocessing.Pool(get_procs_num()) as pool:
        result = pool.starmap(pi_ab_plus_tf, data_list)
    QUICKQEX = (lval, numpy.reshape(result, place_holder.shape))


def dl_tf(lval, bd1, bd2, bd3, bd4, idx1, idx2, idx3):
    '''(10.113)中定义的对l的求导数\n
    '''
    mpinfo = CONFIG.mpinfo
    ksft = CONFIG.ksft
    #
    mdisp = CONFIG.mdisp
    kv1, kv2, kv3 = mpinfo[bd1, idx1], mpinfo[bd2, idx2], mpinfo[bd3, idx3]
    lamb = CONFIG.lamb0 * numpy.exp(-lval)
    #
    idx4 = CONFIG.mk4tab[bd1, bd2, bd3, bd4, idx1, idx2, idx3]
    #确定三个沟道的动量
    q_pp = ksft(kv1, kv2)
    q_fs = ksft(kv3, Point(-kv2.coord[0], -kv2.coord[1], 1))
    q_ex = ksft(kv1, Point(-kv3.coord[0], -kv3.coord[1], 1))
    #
    area = CONFIG.brlu.width * CONFIG.brlu.height
    ltris = CONFIG.ltris
    mlpats = CONFIG.mlpats
    tarea = area / len(ltris)
    #
    value = 0.
    #需要被循环的变量
    idxs = numpy.ndarray((CONFIG.bandnum, CONFIG.bandnum, CONFIG.patchnum))
    idxit = numpy.nditer(idxs, flags=['multi_index'])
    #找到每个n对应的ltris
    #pattris = numpy.ndarray((CONFIG.bandnum, CONFIG.patchnum), dtype=object)
    ##这个时候的npatch是按照alpha带的来区分的，
    ##这个时候被近似的是和（k,alpha）相关的那个动量
    #for bidx in range(CONFIG.bandnum):
    #    lpats = mlpats[bidx]
    #    for tri, pat in zip(ltris, lpats):
    #        #如果这个还没有，增加一个新的[]
    #        if pattris[bidx, pat] is None:
    #            pattris[bidx, pat] = []
    #        pattris[bidx, pat].append(tri)
    #pattris = PATTRIS
    while not idxit.finished:
        alpha, beta, nidx = idxit.multi_index
        #对第alpha个能带上的第ndix个patch进行积分，这个patch上的
        #U都是一个数，bubble需要通过等能线积分出来
        if QUICKQPP[0] == lval:
            #QUICKQPP里面的指标的顺序，就是根据这些指标的作用来的
            pi_min_ab_n_qpp = QUICKQPP[1][alpha, beta, bd1, bd2, nidx, idx1, idx2]
        else:
            raise ValueError("没有计算lval=%.2f" % lval)
            #pi_min_ab_n_qpp = pi_ab_minus_tf(
            #    pattris[alpha, nidx], tarea,
            #    lamb, mdisp[alpha], mdisp[beta],
            #    q_pp, ksft, area)
        if QUICKQFS[0] == lval:
            pi_plu_ab_n_qfs = QUICKQFS[1][alpha, beta, bd2, bd3, nidx, idx2, idx3]
        else:
            raise ValueError("没有计算lval=%.2f" % lval)
            #pi_plu_ab_n_qfs = pi_ab_plus_tf(
            #    pattris[alpha, nidx], tarea,
            #    lamb, mdisp[alpha], mdisp[beta],
            #    q_fs, ksft, area
            #)
        if QUICKQEX[0] == lval:
            pi_plu_ab_n_qex = QUICKQEX[1][alpha, beta, bd1, bd3, nidx, idx1, idx3]
        else:
            raise ValueError("没有计算lval=%.2f" % lval)
            #pi_plu_ab_n_qex = pi_ab_plus_tf(
            #    pattris[alpha, nidx], tarea,
            #    lamb, mdisp[alpha], mdisp[beta],
            #    q_ex, ksft, area
            #)
        value += U[bd2, bd1, alpha, beta, idx2, idx1, nidx] *\
            U[bd3, bd4, alpha, beta, idx3, idx4, nidx] * pi_min_ab_n_qpp
        value += 2*U[alpha, bd4, bd1, beta, nidx, idx4, idx1] *\
            U[alpha, bd2, bd3, beta, nidx, idx2, idx3] * pi_plu_ab_n_qfs
        value -= U[bd4, alpha, bd1, beta, idx4, nidx, idx1] *\
            U[alpha, bd2, bd3, beta, nidx, idx2, idx3] * pi_plu_ab_n_qfs
        value -= U[alpha, bd4, bd1, beta, nidx, idx4, idx1] *\
            U[bd2, alpha, bd3, beta, idx2, nidx, idx3] * pi_plu_ab_n_qfs
        value -= U[bd3, alpha, bd1, beta, idx3, nidx, idx1] *\
            U[bd2, alpha, bd4, beta, idx2, nidx, idx4] * pi_plu_ab_n_qex
        idxit.iternext()
    #总的负号
    value = -value
    #太大的时候就线暂停
    if numpy.abs(value) > 1.e32:
        #print('%.2f大于1.e32' % lval)
        value = 0
    return value
