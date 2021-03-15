"""生成一个长方形的布里渊区"""


import argparse
#绘制示意图
from matplotlib import pyplot
import matplotlib.patches as patches
import matplotlib.path as path
#
from fermi.rectangle import brillouin
from fermi.rectangle import dispersion, dispersion_gradient
from fermi.rectangle import get_rect_patches
from fermi.patches import find_patch
from fermi.surface import const_energy_line
from helpers.rttriangulated import rectangle_split
from helpers.rttriangulated import save_to as tri_save_to
from helpers.discretization import district_visualize
from helpers.discretization import save_to as dis_save_to


def precompute(args):
    '''提前计算'''
    brlu = brillouin()
    #切分布里渊区
    ltris, ladjs = rectangle_split(brlu, args.mesh)
    tri_save_to(
        '{0}_tris.txt'.format(args.prefix),
        brlu, args.mesh, ltris, ladjs
        )
    #找到patches
    pinfos = get_rect_patches(args.patches, args.disp)
    #画一下费米面
    surs = const_energy_line(ltris, ladjs, 0., dispersion)
    #
    pyplot.figure()
    #绘制patches对应的点
    xvals = []
    yvals = []
    for idx, pnt in enumerate(pinfos, 0):
        xvals.append(pnt.coord[0])
        yvals.append(pnt.coord[1])
        pyplot.text(pnt.coord[0]+0.1, pnt.coord[1], 's%s' % idx)
    pyplot.scatter(xvals, yvals, c='g', lw=4)
    #绘制费米面的线
    for seg in surs:
        if seg is None:
            continue
        xvals = [_pt.coord[0] for _pt in seg.ends]
        yvals = [_pt.coord[1] for _pt in seg.ends]
        pyplot.plot(xvals, yvals, c='y', lw=1)
    pyplot.savefig('{0}_surface.svg'.format(args.prefix))
    pyplot.close()
    #
    step = 3.1415927 / args.mesh / 4
    slpats = []
    for tri in ltris:
        pt_ = find_patch(tri.center, pinfos, dispersion, dispersion_gradient, step)
        slpats.append(pt_)
    dis_save_to('{0}_district.txt'\
        .format(args.prefix), slpats)
    district_visualize(ltris, slpats, '{0}_district.svg'\
        .format(args.prefix))


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 square_brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    #parser.add_argument('-d', '--disp', type=str, default='square', help='disp str')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str,\
        default='scripts/rectbrlu/squ', help='saved file prefix')
    args = parser.parse_args()
    setattr(args, 'disp', 'square')
    print('色散 ', args.disp)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    precompute(args)
    print('保存到 ', args.prefix)


if __name__ == '__main__':
    main()
