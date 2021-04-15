"""生成stripe系统的布里渊区"""


import argparse
#绘制示意图
from matplotlib import pyplot
import matplotlib.patches as patches
import matplotlib.path as path
#
from fermi.stripesquare import brillouin, set_stripe, get_max_val, set_potential
from fermi.stripesquare import s_band_disp, p_band_disp
from fermi.stripesquare import s_band_gd, p_band_gd
from fermi.stripesquare import get_s_band_patches, get_p_band_patches
from fermi.patches import find_patch
from fermi.surface import const_energy_line
from helpers.triangulated import square_split
from helpers.triangulated import save_to as tri_save_to
from helpers.discretization import district_visualize
from helpers.discretization import save_to as dis_save_to


def precompute(args):
    '''提前计算'''
    brlu = brillouin()
    set_stripe(args.stripe)
    set_potential(args.nu)
    #切分布里渊区
    ltris, ladjs = square_split(brlu, args.mesh)
    tri_save_to(
        '{0}_{1:.2f}_{2:.2f}_tris.txt'.\
            format(args.prefix, args.stripe, args.nu),
        brlu, args.mesh, ltris, ladjs
        )
    #找到patches
    spats = get_s_band_patches(args.patches)
    ppats = get_p_band_patches(args.patches)
    #画一下费米面
    ssur = const_energy_line(ltris, ladjs, 0., s_band_disp)
    psur = const_energy_line(ltris, ladjs, 0., p_band_disp)
    #
    pyplot.figure()
    #绘制patches对应的点
    xvals = []
    yvals = []
    for idx, pnt in enumerate(spats, 0):
        xvals.append(pnt.coord[0])
        yvals.append(pnt.coord[1])
        pyplot.text(pnt.coord[0]+0.1, pnt.coord[1], 's%s' % idx)
    pyplot.scatter(xvals, yvals, c='g', lw=4)
    #绘制费米面的线
    for seg in ssur:
        if seg is None:
            continue
        xvals = [_pt.coord[0] for _pt in seg.ends]
        yvals = [_pt.coord[1] for _pt in seg.ends]
        pyplot.plot(xvals, yvals, c='y', lw=1)
    #
    xvals = []
    yvals = []
    for idx, pnt in enumerate(ppats, 0):
        xvals.append(pnt.coord[0])
        yvals.append(pnt.coord[1])
        pyplot.text(pnt.coord[0]+0.1, pnt.coord[1], 'p%s' % idx)
    pyplot.scatter(xvals, yvals, c='r', lw=4)
    #
    for seg in psur:
        if seg is None:
            continue
        xvals = [_pt.coord[0] for _pt in seg.ends]
        yvals = [_pt.coord[1] for _pt in seg.ends]
        pyplot.plot(xvals, yvals, c='k', lw=1)
    pyplot.savefig('{0}_{1:.2f}_{2:.2f}_sur.svg'.\
        format(args.prefix, args.stripe, args.nu))
    pyplot.close()
    #
    step = 3.1415927 / args.mesh / 2
    slpats = []
    for tri in ltris:
        pt_ = find_patch(tri.center, spats, s_band_disp, s_band_gd, step)
        slpats.append(pt_)
    dis_save_to('{0}_{1:.2f}_{2:.2f}_spt.txt'\
        .format(args.prefix, args.stripe, args.nu), slpats)
    district_visualize(ltris, slpats, '{0}_{1:.2f}_{2:.2f}_spt.svg'\
        .format(args.prefix, args.stripe, args.nu))
    plpats = []
    for tri in ltris:
        pt_ = find_patch(tri.center, ppats, p_band_disp, p_band_gd, step)
        plpats.append(pt_)
    dis_save_to('{0}_{1:.2f}_{2:.2f}_ppt.txt'\
        .format(args.prefix, args.stripe, args.nu), plpats)
    district_visualize(ltris, plpats, '{0}_{1:.2f}_{2:.2f}_ppt.svg'\
        .format(args.prefix, args.stripe, args.nu))


def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 square_brillouin.py',
        description='precompute patches'
    )
    parser.add_argument('-p', '--patches', type=int, required=True, help='patches number')
    parser.add_argument('-s', '--stripe', type=float, required=True, help='stripe strength')
    parser.add_argument('-n', '--nu', type=float, required=True, help='hole doped')
    parser.add_argument('-m', '--mesh', type=int, default=50, help='triangles number')
    parser.add_argument('--prefix', type=str,\
        default='scripts/stripe/str', help='saved file prefix')
    args = parser.parse_args()
    print('stripe强度 ', args.stripe)
    print('掺杂的化学势 ', args.nu)
    print('patch数量', args.patches)
    print('布里渊区网格数量', args.mesh)
    precompute(args)
    print('保存到 ', args.prefix)


if __name__ == '__main__':
    main()
