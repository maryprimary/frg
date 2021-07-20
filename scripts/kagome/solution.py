"""
解AVSb的方程
"""

import os
import multiprocessing
import argparse
import numpy
from basics import get_procs_num
from fermi.kagome import shift_kv, get_von_hove_patches
from fermi.kagome import p_disp
from fermi.kagome import band_u, band_v
import flowequ.mulitband_hubbard as hubbard
from helpers.ettriangulated import load_from as triload
from helpers.discretization import load_from as patload



def load_brillouin(args):
    '''加载布里渊区'''
    brlu, nps, ltris, ladjs =\
        triload('{0}_tris.txt'.format(args.prefix))
    if nps != args.mesh:
        raise ValueError('mesh数量对应不上')
    #找到patches
    mpinfo = numpy.ndarray((1, args.patches), dtype=object)
    mpinfo[0, :] = get_von_hove_patches(args.patches)
    #找到所有的小三角形属于哪个patch
    mlpats = numpy.ndarray((1, len(ltris)), dtype=int)
    mlpats[0, :] = patload('{0}_pats.txt'.format(args.prefix))
    return brlu, ltris, ladjs, mpinfo, mlpats


def slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats):
    '''解方程'''
    #初始化U
    uval = numpy.zeros((1, 1, 1, 1, args.patches, args.patches, args.patches))
    uval[0, 0, 0, 0, :, :, :] += band_u(6.0, mpinfo[0])
    uval[0, 0, 0, 0, :, :, :] += band_v(6.0, mpinfo[0])
    #初始化hubbard
    hubbard.uinit(uval, args.patches, 1)
    lamb0 = 2.0
    print('lamb0 = ', lamb0)
    hubbard.config_init(
        brlu, ltris, ladjs, mpinfo, mlpats,
        [p_disp], [None],
        shift_kv, lamb0, find_mode=3
    )
    #输出文件夹
    if not os.path.isdir('heatmap8'):
        os.mkdir('heatmap8')
    rpath = 'heatmap8/kagome4'
    if not os.path.isdir(rpath):
        os.mkdir(rpath)
    lval = 0.
    lstep = 0.01
    numpy.save('{0}/{1:.2f}U.npy'.format(rpath, lval), hubbard.U)
    #读取配置
    #lval = 6.70
    #hubbard.U = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, lval))
    for _ in range(1000):
        hubbard.precompute_contour(lval)
        hubbard.precompute_qpp(lval)
        hubbard.precompute_qfs(lval)
        hubbard.precompute_nqfs(lval)
        hubbard.precompute_qex(lval)
        hubbard.precompute_nqex(lval)
        #计算每个idx的导数
        data_list = []
        ndit = numpy.nditer(hubbard.U, flags=['multi_index'])
        while not ndit.finished:
            bd1, bd2, bd3, bd4, idx1, idx2, idx3 = ndit.multi_index
            data_list.append((lval, bd1, bd2, bd3, bd4, idx1, idx2, idx3))
            ndit.iternext()
        #进程池
        #KNOWN ISSUE: 在修改全局变量之间建立的Pool，里面不会包含全局变量
        with multiprocessing.Pool(get_procs_num()) as pool:
            result = pool.starmap(hubbard.dl_ec, data_list)
        duval = numpy.reshape(result, hubbard.U.shape)
        #把每个idx的值加上
        #这两个过程不能放在一起，因为计算dl_ec的时候用到了hubbard.U
        hubbard.U += duval * lstep
        lval += lstep
        #
        del data_list, result, duval
        #
        #uval2 = numpy.load('{0}/{1:.2f}U.chk'.format(rpath, 10.76))
        #if lval == 10.76:
        #    assert numpy.allclose(hubbard.U, uval2)
        numpy.save('{0}/{1:.2f}U.npy'.format(rpath, lval), hubbard.U)

def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 scripts/avsb/solution.py',
        description='compute equation'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-m', '--mesh', type=int, default=100, help='triangles number')
    parser.add_argument('--prefix', type=str, default='scripts/kagome/kag',\
        help='saved file prefix')
    args = parser.parse_args()
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    print('读取自 ', args.prefix)
    brlu, ltris, ladjs, mpinfo, mlpats = load_brillouin(args)
    slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats)



if __name__ == '__main__':
    main()
