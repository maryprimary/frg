"""
正方格子在温度流下面的结果
"""


import os
import multiprocessing
import argparse
import numpy
from basics import get_procs_num
#
from fermi.square import brillouin, shift_kv, dispersion, dispersion_gradient
from fermi.square import hole_disp
from fermi.patches import get_patches, find_patch
from helpers.triangulated import load_from as triload
from helpers.discretization import load_from as patload
#
from helpers.drawer import draw_heatmap, draw_components
from helpers.triangle_refine import lrttris_refine
from flowequ import multiband_hubbard_tf as hubbard

def load_brillouin(args):
    '''加载布里渊区的配置'''
    disp = {
        'square': dispersion, 'hole': hole_disp
    }[args.disp]
    dispgd = {
        'square': dispersion_gradient, 'hole': dispersion_gradient
    }[args.disp]
    brlu, nps, ltris, ladjs =\
        triload('{0}_triangle_{1}.txt'.format(args.prefix, args.disp))
    if nps != args.mesh:
        raise ValueError('mesh数量对应不上')
    pinfo = get_patches(brlu, args.patches, disp)
    lpats = patload('{0}_district_{1}.txt'.format(args.prefix, args.disp))
    mpinfo = numpy.ndarray([1, args.patches], dtype=object)
    mpinfo[0, :] = pinfo
    #多个带的patch区域
    mlpats = numpy.ndarray([1, len(ltris)], dtype=int)
    mlpats[0, :] = lpats
    return brlu, ltris, ladjs, mpinfo, mlpats


def slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats):
    '''解方程'''
    disp = {
        'square': dispersion, 'hole': hole_disp
    }[args.disp]
    dispgd = {
        'square': dispersion_gradient, 'hole': dispersion_gradient
    }[args.disp]
    #初始化U
    uini = numpy.zeros((1, 1, 1, 1, args.patches, args.patches, args.patches))
    uini[0, 0, 0, 0, ...] = 2.
    hubbard.uinit(uini, args.patches, 1)
    #初始化hubbard模型
    hubbard.config_init(
        brlu, ltris, ladjs, mpinfo, mlpats,
        [disp], [dispgd], shift_kv, 8.
    )
    hubbard.precompute_patches()
    #输出文件夹
    if not os.path.isdir('heatmap6'):
        os.mkdir('heatmap6')
    lval = 0.
    lstep = 0.01
    numpy.save('heatmap6/{:.2f}.npy'.format(lval), hubbard.U)
    #加载结果
    #lval = 6.00
    #hubbard.U = numpy.load('heatmap6/{:.2f}.npy'.format(lval))
    for idx in range(1500):
        if numpy.mod(idx, 100) == 0:
            temp = 8. * numpy.exp(-lval)
            newltris = []
            for tri in hubbard.CONFIG.ltris:
                eng = disp(tri.center.coord[0], tri.center.coord[1])
                if abs(eng / temp) < 25:
                    newltris.append(tri)
            if len(newltris) < len(hubbard.CONFIG.ltris)*0.8:
                newltris = lrttris_refine(newltris)
                newmlpats = numpy.ndarray([1, len(newltris)], dtype=int)
                step = 3.1415927 / args.mesh / 2
                nlpat = [find_patch(tri.center, mpinfo[0, :], disp, dispgd, step)\
                    for tri in newltris]
                newmlpats[0, :] = nlpat
                hubbard.config_reset_ltris(newltris, None, newmlpats)
                hubbard.precompute_patches()
                draw_components([], brlu.edges, newltris, rtcc=nlpat,\
                    save='heatmap6/{:.2f}pats.svg'.format(lval))
        hubbard.precompute_qpp(lval)
        hubbard.precompute_qfs(lval)
        hubbard.precompute_qex(lval)
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
            result = pool.starmap(hubbard.dl_tf, data_list)
        duval = numpy.reshape(result, hubbard.U.shape)
        #把每个idx的值加上
        #这两个过程不能放在一起，因为计算dl_ec的时候用到了hubbard.U
        hubbard.U += duval * lstep
        lval += lstep
        numpy.save('heatmap6/{:.2f}.npy'.format(lval), hubbard.U)


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 square_brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-d', '--disp', type=str, default='square', help='dispersion')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str, default='scripts/square/square', help='saved file prefix')
    args = parser.parse_args()
    print('色散 ', args.disp)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    print('读取自 ', args.prefix)
    brlu, ltris, ladjs, mpinfo, mlpats = load_brillouin(args)
    slove_equ(args, brlu, ltris, ladjs, mpinfo, mlpats)


if __name__ == '__main__':
    main()
