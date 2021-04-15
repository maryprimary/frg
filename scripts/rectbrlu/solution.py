"""多线程版本的解方程"""

import os
#import cProfile
import multiprocessing
import argparse
import numpy
from basics import get_procs_num
#格子的相关功能
from fermi.rectangle import dispersion, dispersion_gradient, shift_kv
#加载布里渊区的配置
from fermi.rectangle import get_rect_patches
from helpers.rttriangulated import load_from as triload
from helpers.discretization import load_from as patload
#和方程相关
from helpers.drawer import draw_heatmap
from flowequ import hubbard


def load_brillouin(args):
    '''加载布里渊区的配置'''
    brlu, nps, ltris, ladjs =\
        triload('{0}_tris.txt'.format(args.prefix))
    if nps != args.mesh:
        raise ValueError('mesh数量对应不上')
    pinfo = get_rect_patches(args.patches, args.disp)
    lpats = patload('{0}_district.txt'.format(args.prefix))
    return brlu, ltris, ladjs, pinfo, lpats


def slove_equ(args, brlu, ltris, ladjs, pinfo, lpats):
    '''解方程'''
    #初始化U
    hubbard.uinit(1.0, args.patches)
    #初始化hubbard模型
    hubbard.config_init(
        brlu, ltris, ladjs, pinfo, lpats,
        dispersion, dispersion_gradient, shift_kv, 4.0
    )
    #输出文件夹
    if not os.path.isdir('heatmap7'):
        os.mkdir('heatmap7')
    lval = 0.
    lstep = 0.01
    draw_heatmap(hubbard.U[:, :, 0], save='heatmap7/{:.2f}.jpg'.format(lval))
    for _ in range(320):
        #duval = numpy.zeros_like(hubbard.U)
        hubbard.precompute_contour(lval)
        hubbard.precompute_qpp(lval)
        hubbard.precompute_qfs(lval)
        hubbard.precompute_nqfs(lval)
        hubbard.precompute_qex(lval)
        hubbard.precompute_nqex(lval)
        #计算每个idx的导数
        data_list = []
        for idx1 in range(args.patches):
            for idx2 in range(args.patches):
                for idx3 in range(args.patches):
                    data_list.append((lval, idx1, idx2, idx3))
                    #duval[idx1, idx2, idx3] = hubbard.dl_ec(lval, idx1, idx2, idx3)
        #进程池
        #KNOWN ISSUE: 在修改全局变量之间建立的Pool，里面不会包含全局变量
        with multiprocessing.Pool(get_procs_num()) as pool:
            result = pool.starmap(hubbard.dl_ec, data_list)
        duval = numpy.reshape(result, (args.patches, args.patches, args.patches))
        #把每个idx的值加上
        #这两个过程不能放在一起，因为计算dl_ec的时候用到了hubbard.U
        hubbard.U += duval * lstep
        lval += lstep
        draw_heatmap(hubbard.U[:, :, 0], save='heatmap7/{:.2f}.jpg'.format(lval))


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 square_brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    #parser.add_argument('-d', '--disp', type=str, default='square', help='dispersion')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str, default='scripts/rectbrlu/squ', help='saved file prefix')
    args = parser.parse_args()
    setattr(args, 'disp', 'square')
    print('色散 ', args.disp)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    print('读取自 ', args.prefix)
    brlu, ltris, ladjs, pinfo, lpats = load_brillouin(args)
    slove_equ(args, brlu, ltris, ladjs, pinfo, lpats)



if __name__ == '__main__':
    main()
