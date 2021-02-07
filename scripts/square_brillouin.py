"""计算好布里渊区的剖分，patches等内容，以后可以直接load"""

import argparse
from fermi.square import brillouin, dispersion, dispersion_gradient
from fermi.square import hole_disp
from fermi.surface import const_energy_line
from fermi.patches import get_patches, find_patch
from helpers.triangulated import square_split
from helpers.triangulated import save_to as tri_save_to
from helpers.discretization import district_visualize, patches_visualize
from helpers.discretization import save_to as dis_save_to


def precompute(args):
    '''计算patches'''
    disp = {
        'square': dispersion, 'hole': hole_disp
    }[args.disp]
    dispgd = {
        'square': dispersion_gradient, 'hole': dispersion_gradient
    }[args.disp]
    #正方格子的布里渊区
    brlu = brillouin()
    #找到patches
    pats = get_patches(brlu, args.patches, disp)
    #切分布里渊区，切分的三角形的数目是mesh*mesh*4
    ltris, ladjs = square_split(brlu, args.mesh)
    tri_save_to(
        '{0}_triangle_{1}.txt'.format(args.prefix, args.disp),
        brlu, args.mesh, ltris, ladjs
        )
    #patches和费米面的图片
    lsur = const_energy_line(ltris, ladjs, 0., disp)
    patches_visualize(pats, lsur, '{0}_surface_{1}.svg'.format(args.prefix, args.disp))
    #求出每个Rtriangle所在的patch
    #这种投影法把交点设置在Umklapp surface
    lpats = [find_patch(tri.center, pats, dispersion, dispgd) for tri in ltris]
    dis_save_to('{0}_district_{1}.txt'.format(args.prefix, args.disp), lpats)
    district_visualize(ltris, lpats, '{0}_patches_{1}.svg'.format(args.prefix, args.disp))



def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 square_brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-d', '--disp', type=str, default='square', help='dispersion')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str, default='scripts/square', help='saved file prefix')
    args = parser.parse_args()
    print('色散 ', args.disp)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    precompute(args)
    print('保存到 ', args.prefix)


if __name__ == '__main__':
    main()
