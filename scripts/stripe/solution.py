"""解stripe系统"""

import os
import multiprocessing
import argparse
import numpy
from basics import get_procs_num
from fermi.stripesquare import set_stripe, set_potential
from fermi.stripesquare import get_s_band_patches, get_p_band_patches
from fermi.stripesquare import get_initu
from fermi.stripesquare import s_band_disp, s_band_gd, p_band_disp, p_band_gd
from fermi.stripesquare import shift_kv, get_max_val
import flowequ.mulitband_hubbard as hubbard
from helpers.drawer import draw_heatmap
from helpers.triangulated import load_from as triload
from helpers.discretization import load_from as patload
from helpers.discretization import district_visualize

def load_brillouin(args):
    '''加载布里渊区的配置'''
    set_stripe(args.stripe)
    set_potential(args.nu)
    brlu, nps, ltris, ladjs =\
        triload('{0}_{1:.2f}_{2:.2f}_tris.txt'.\
            format(args.prefix, args.stripe, args.nu))
    if nps != args.mesh:
        raise ValueError('mesh数量对应不上')
    #找到patches
    spats = get_s_band_patches(args.patches)
    ppats = get_p_band_patches(args.patches)
    #s带的patch
    slpats = patload('{0}_{1:.2f}_{2:.2f}_spt.txt'\
        .format(args.prefix, args.stripe, args.nu))
    #p带的patch
    plpats = patload('{0}_{1:.2f}_{2:.2f}_ppt.txt'\
        .format(args.prefix, args.stripe, args.nu))
    #district_visualize(ltris, slpats, 'show')
    #district_visualize(ltris, plpats, 'show')
    #整理成数组
    mpinfo = numpy.ndarray((2, args.patches), dtype=object)
    mpinfo[0, :] = spats
    mpinfo[1, :] = ppats
    mlpats = numpy.ndarray((2, len(ltris)), dtype=numpy.int)
    mlpats[0, :] = slpats
    mlpats[1, :] = plpats
    #
    return brlu, ltris, ladjs, mpinfo, mlpats


def slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats):
    '''解方程'''
    #初始化U
    spats = mpinfo[0, :]
    ppats = mpinfo[1, :]
    hubbard.uinit(get_initu(spats, ppats, 1.0), args.patches, 2)
    #初始化hubbard模型
    lamb0 = get_max_val()
    print('lamb0 = ', lamb0)
    hubbard.config_init(
        brlu, ltris, ladjs, mpinfo, mlpats,
        [s_band_disp, p_band_disp], [s_band_gd, p_band_gd],
        shift_kv, lamb0
    )
    #
    #输出文件夹
    if not os.path.isdir('heatmap6'):
        os.mkdir('heatmap6')
    rpath = 'heatmap6/s{0:.2f}nu{1:.2f}'.\
        format(args.stripe, args.nu)
    if not os.path.isdir(rpath):
        os.mkdir(rpath)
    #
    lval = 0.
    lstep = 0.01
    numpy.save('{0}/{1:.2f}U.npy'.format(rpath, lval), hubbard.U)
    #draw_heatmap(
    #    hubbard.U[0, 0, 0, 0, :, :, 12],
    #    save='{0}/{1:.2f}ssss.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[0, 0, 0, 1, :, :, 12],
    #    save='{0}/{1:.2f}sssp.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[1, 1, 1, 0, :, :, 12],
    #    save='{0}/{1:.2f}ppps.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[1, 1, 0, 0, :, :, 12],
    #    save='{0}/{1:.2f}ppss.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[1, 0, 0, 1, :, :, 12],
    #    save='{0}/{1:.2f}pssp.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[0, 1, 0, 1, :, :, 12],
    #    save='{0}/{1:.2f}spsp.jpg'.format(rpath, lval)
    #)
    #draw_heatmap(
    #    hubbard.U[1, 1, 1, 1, :, :, 12],
    #    save='{0}/{1:.2f}pppp.jpg'.format(rpath, lval)
    #)
    print(hubbard.U[0, 0, 0, 0, 0, 0, 0])
    print(hubbard.U[1, 1, 1, 1, 0, 0, 0])
    print(hubbard.U[1, 1, 1, 0, 1, 1, 1])
    print(hubbard.U[0, 1, 1, 1, 1, 1, 1])
    print(hubbard.U[1, 1, 0, 0, 2, 2, 2])
    print(hubbard.U[0, 0, 1, 1, 2, 2, 2])
    print(hubbard.U[1, 1, 1, 0, 3, 3, 3])
    print(hubbard.U[0, 1, 1, 1, 3, 3, 3])
    for _ in range(1000):
        hubbard.precompute_contour(lval)
        hubbard.precompute_qpp(lval)
        hubbard.precompute_qfs(lval)
        hubbard.precompute_nqfs(lval)
        hubbard.precompute_qex(lval)
        hubbard.precompute_nqex(lval)
        #进程池
        #KNOWN ISSUE: 在修改全局变量之间建立的Pool，里面不会包含全局变量
        pool = multiprocessing.Pool(get_procs_num())
        #计算每个idx的导数
        data_list = []
        ndit = numpy.nditer(hubbard.U, flags=['multi_index'])
        while not ndit.finished:
            bd1, bd2, bd3, bd4, idx1, idx2, idx3 = ndit.multi_index
            data_list.append((lval, bd1, bd2, bd3, bd4, idx1, idx2, idx3))
            ndit.iternext()
        result = pool.starmap(hubbard.dl_ec, data_list)
        duval = numpy.reshape(result, hubbard.U.shape)
        #把每个idx的值加上
        #这两个过程不能放在一起，因为计算dl_ec的时候用到了hubbard.U
        hubbard.U += duval * lstep
        lval += lstep
        #
        del data_list, result, duval, pool
        #
        #draw_heatmap(
        #    hubbard.U[0, 0, 0, 0, :, :, 12],
        #    save='{0}/{1:.2f}ssss.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[0, 0, 0, 1, :, :, 12],
        #    save='{0}/{1:.2f}sssp.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[1, 1, 1, 0, :, :, 12],
        #    save='{0}/{1:.2f}ppps.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[1, 1, 0, 0, :, :, 12],
        #    save='{0}/{1:.2f}ppss.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[1, 0, 0, 1, :, :, 12],
        #    save='{0}/{1:.2f}pssp.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[0, 1, 0, 1, :, :, 12],
        #    save='{0}/{1:.2f}spsp.jpg'.format(rpath, lval)
        #)
        #draw_heatmap(
        #    hubbard.U[1, 1, 1, 1, :, :, 12],
        #    save='{0}/{1:.2f}pppp.jpg'.format(rpath, lval)
        #)
        numpy.save('{0}/{1:.2f}U.npy'.format(rpath, lval), hubbard.U)
        print(hubbard.U[0, 0, 0, 0, 0, 0, 0])
        print(hubbard.U[1, 1, 1, 1, 0, 0, 0])
        print(hubbard.U[1, 1, 1, 0, 1, 1, 1])
        print(hubbard.U[0, 1, 1, 1, 1, 1, 1])
        print(hubbard.U[1, 1, 0, 0, 2, 2, 2])
        print(hubbard.U[0, 0, 1, 1, 2, 2, 2])
        print(hubbard.U[1, 1, 1, 0, 3, 3, 3])
        print(hubbard.U[0, 1, 1, 1, 3, 3, 3])


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 solution.py',
        description='compute equation'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-s', '--stripe', type=float, required=True, help='stripe strength')
    parser.add_argument('-n', '--nu', type=float, required=True, help='hole doped')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str, default='scripts/stripe/str',\
        help='saved file prefix')
    args = parser.parse_args()
    print('stripe强度 ', args.stripe)
    print('掺杂的化学势 ', args.nu)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    print('读取自 ', args.prefix)
    brlu, ltris, ladjs, mpinfo, mlpats = load_brillouin(args)
    slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats)


if __name__ == '__main__':
    main()
