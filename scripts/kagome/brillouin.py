"""Kagome的布里渊区"""

import argparse
#绘制示意图
#
from fermi.kagome import brillouin, p_disp, get_von_hove_patches
from fermi.patches import find_patch
from fermi.surface import const_energy_line, const_energy_line_in_patches
from helpers.drawer import draw_components
from helpers.ettriangulated import hexagon_split
from helpers.ettriangulated import save_to as tri_save_to
from helpers.discretization import save_to as dis_save_to


def precompute(args):
    '''计算'''
    #需要用到的所有东西
    brlu = brillouin()
    ltris, ladjs = hexagon_split(brlu, args.mesh)
    pinfos = get_von_hove_patches(args.patches)
    lpats = [find_patch(tri.center, pinfos, None, None, None, mode=2) for tri in ltris]
    #画一下费米面
    edges = const_energy_line(ltris, ladjs, 0.0, p_disp)
    dedgs = brlu.edges + edges
    draw_components(pinfos, dedgs, ltris, sgcc=[1]*len(dedgs), rtcc=lpats, save=\
        '{0}_sur.svg'.format(args.prefix))
    #保存切分的信息
    tri_save_to(
        '{0}_tris.txt'.format(args.prefix),
        brlu, args.mesh, ltris, ladjs
    )
    #保存区域的信息
    dis_save_to(
        '{0}_pats.txt'.format(args.prefix), lpats
    )
    edges, pidxs = const_energy_line_in_patches(ltris, ladjs, lpats, 0.0, p_disp)
    draw_components([], edges, [], sgcc=pidxs)
    #district_visualize(ltris, lpats, show='window')


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 scripts/kagome/brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str,\
        default='scripts/kagome/kag', help='saved file prefix')
    args = parser.parse_args()
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    precompute(args)
    print('保存到 ', args.prefix)


if __name__ == '__main__':
    main()
