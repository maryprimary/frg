"""
AV3Sb5的布里渊区
"""

import argparse
import numpy
#
from fermi.avsb import brillouin, p2_disp, d1_disp
from fermi.avsb import get_von_hove_patches
from fermi.patches import find_patch
from fermi.surface import const_energy_line, const_energy_line_in_patches
from helpers.ettriangulated import hexagon_split
from helpers.ettriangulated import save_to as tri_save_to
from helpers.discretization import save_to as dis_save_to
from helpers.drawer import draw_components


def precompute(args):
    '''计算'''
    #avsb一个共有两个能带
    #第一个是t=1的d能带，第二个是t=0.5的p能带
    brlu = brillouin()
    ltris, ladjs = hexagon_split(brlu, args.mesh)
    if numpy.mod(args.patches, 6) != 0:
        raise ValueError("patch数量需要是6的倍数")
    pinfos = numpy.ndarray((2, args.patches), dtype=object)
    pinfos[0, :] = get_von_hove_patches(args.patches, d1_disp)
    pinfos[1, :] = get_von_hove_patches(args.patches, p2_disp)
    #lpats = [find_patch(tri.center, pinfos, None, None, None, mode=3) for tri in ltris]
    #画一下费米面
    d1pats = [find_patch(tri.center, pinfos[0, :], None, None, None, mode=3) for tri in ltris]
    edges = const_energy_line(ltris, ladjs, 0.0, d1_disp)
    dedgs = brlu.edges + edges
    draw_components(pinfos[0], dedgs, ltris, sgcc=[1]*len(dedgs), rtcc=d1pats, save=\
        '{0}d1sur.svg'.format(args.prefix))
    #p带的费米面
    p2pats = [find_patch(tri.center, pinfos[1, :], None, None, None, mode=3) for tri in ltris]
    edges = const_energy_line(ltris, ladjs, 0.0, p2_disp)
    dedgs = brlu.edges + edges
    draw_components(pinfos[1], dedgs, ltris, sgcc=[1]*len(dedgs), rtcc=p2pats, save=\
        '{0}p2sur.svg'.format(args.prefix))
    #两个费米面在一起
    d1sur = const_energy_line(ltris, ladjs, 0.0, d1_disp)
    p2sur = const_energy_line(ltris, ladjs, 0.0, p2_disp)
    draw_components(numpy.hstack([pinfos[0], pinfos[1]]),\
        d1sur + p2sur + brlu.edges, [],\
        sgcc=[1]*len(d1sur) + [2]*len(p2sur) + [3]*len(brlu.edges),\
        save='{0}duosur.svg'.format(args.prefix))
    #保存切分的信息
    tri_save_to(
        '{0}tris.txt'.format(args.prefix),
        brlu, args.mesh, ltris, ladjs
    )
    #保存区域的信息
    dis_save_to(
        '{0}d1pats.txt'.format(args.prefix), d1pats
    )
    dis_save_to(
        '{0}p2pats.txt'.format(args.prefix), p2pats
    )
    #
    edges, pidxs = const_energy_line_in_patches(ltris, ladjs, d1pats, 0.5, d1_disp)
    draw_components([], edges, [], sgcc=pidxs)
    edges, pidxs = const_energy_line_in_patches(ltris, ladjs, p2pats, 0.3, p2_disp)
    draw_components([], edges, [], sgcc=pidxs)


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 scripts/avsb/brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str,\
        default='scripts/avsb/', help='saved file prefix')
    args = parser.parse_args()
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    precompute(args)
    print('保存到 ', args.prefix)

if __name__ == '__main__':
    main()
