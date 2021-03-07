"""可视化计算的结果"""


import argparse
import numpy
from helpers.drawer import draw_heatmap, draw_points
from basics import Point
from fermi.stripesquare import inverse_uval, set_stripe
from fermi.stripesquare import get_s_band_patches, get_p_band_patches


def draw_single_channel(args):
    '''绘制单通道的图片结果'''
    rpath = 'heatmap6/s{:.2f}'.format(args.stripe)
    uval = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, args.lval))
    #
    c2i = {'s': [0], 'p': [1], '?': [0, 1]}
    i2c = {0: 's', 1: 'p'}
    candi = [c2i[bdi] for bdi in args.channel]
    dims = [len(bdi) for bdi in candi]
    place_holder = numpy.ndarray(dims)
    ndit = numpy.nditer(place_holder, flags=['multi_index'])
    while not ndit.finished:
        idx1, idx2, idx3, idx4 = ndit.multi_index
        bd1 = candi[0][idx1]
        bd2 = candi[1][idx2]
        bd3 = candi[2][idx3]
        bd4 = candi[3][idx4]
        print(bd1, bd2, bd3, bd4, '\t', i2c[bd1], i2c[bd2], i2c[bd3], i2c[bd4])
        draw_heatmap(
            uval[bd1, bd2, bd3, bd4, :, :, args.n3idx],
            save='show'
        )
        ndit.iternext()


def draw_mixed_channel(args):
    '''绘制通道混合在一起的结果'''
    rpath = 'heatmap6/s{:.2f}'.format(args.stripe)
    uval = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, args.lval))
    #按照逆时针顺序来给带排序
    shape = numpy.shape(uval)
    bnum, pnum = shape[0], shape[4]
    ppnum = pnum // 4
    idxpairs = []
    #s带左上
    for idx in range(ppnum):
        idxpairs.append((0, 3*ppnum + idx))
    #p带右上
    for idx in range(ppnum):
        idxpairs.append((1, 2*ppnum + idx))
    #p带左上
    for idx in range(ppnum):
        idxpairs.append((1, 3*ppnum + idx))
    #s带右上
    for idx in range(ppnum):
        idxpairs.append((0, idx))
    #s带右下
    for idx in range(ppnum):
        idxpairs.append((0, ppnum + idx))
    #p带左下
    for idx in range(ppnum):
        idxpairs.append((1, idx))
    #p带右下
    for idx in range(ppnum):
        idxpairs.append((1, ppnum + idx))
    #s带左下
    for idx in range(ppnum):
        idxpairs.append((1, 2*ppnum + idx))
    #整理图片
    #print(idxpairs)
    totpnum = len(idxpairs)
    heatmap = numpy.ndarray((totpnum, totpnum))
    idx3pair = idxpairs[args.n3idx]
    for chn in args.channel:
        if chn == 's':
            bd4 = 0
        elif chn == 'p':
            bd4 = 1
        else:
            raise ValueError('bd4不对')
        for hid1, pr1 in enumerate(idxpairs, 0):
            for hid2, pr2 in enumerate(idxpairs, 0):
                bd1, idx1 = pr1
                bd2, idx2 = pr2
                bd3, idx3 = idx3pair[0], idx3pair[1]
                heatmap[hid1, hid2] = uval[bd1, bd2, bd3, bd4, idx1, idx2, idx3]
        draw_heatmap(heatmap)


def draw_count_channel(args):
    '''绘制通道混合在一起的结果'''
    rpath = 'heatmap6/s{:.2f}'.format(args.stripe)
    uval = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, args.lval))
    #按照逆时针顺序来给带排序
    shape = numpy.shape(uval)
    bnum, pnum = shape[0], shape[4]
    ppnum = pnum // 4
    idxpairs = []
    #s带左上
    for idx in range(ppnum):
        idxpairs.append((0, 3*ppnum + idx))
    #p带右上
    for idx in range(ppnum):
        idxpairs.append((1, 2*ppnum + idx))
    #p带左上
    for idx in range(ppnum):
        idxpairs.append((1, 3*ppnum + idx))
    #s带右上
    for idx in range(ppnum):
        idxpairs.append((0, idx))
    #s带右下
    for idx in range(ppnum):
        idxpairs.append((0, ppnum + idx))
    #p带左下
    for idx in range(ppnum):
        idxpairs.append((1, idx))
    #p带右下
    for idx in range(ppnum):
        idxpairs.append((1, ppnum + idx))
    #s带左下
    for idx in range(ppnum):
        idxpairs.append((1, 2*ppnum + idx))
    #整理图片
    #print(idxpairs)
    totpnum = len(idxpairs)
    heatmap = numpy.ndarray((totpnum, totpnum))
    idx3pair = idxpairs[args.n3idx]
    for chn in args.channel:
        for hid1, pr1 in enumerate(idxpairs, 0):
            for hid2, pr2 in enumerate(idxpairs, 0):
                bd1, idx1 = pr1
                bd2, idx2 = pr2
                bd3, idx3 = idx3pair[0], idx3pair[1]
                bdsum = bd1 + bd2 + bd3
                #这种代表需要有偶数个p
                if chn == 'e':
                    #如果已经有偶数个p，则加一个s
                    bd4 = 0 if bdsum % 2 == 0 else 1
                #这种代表有奇数个p
                elif chn == 'o':
                    #如果已经有偶数个p，则加一个p
                    bd4 = 1 if bdsum % 2 == 0 else 0
                else:
                    raise ValueError('bd4不对')
                heatmap[hid1, hid2] = uval[bd1, bd2, bd3, bd4, idx1, idx2, idx3]
        draw_heatmap(heatmap)


def draw_basis_channel(args):
    '''在子格子的表示上显示'''
    set_stripe(args.stripe)
    rpath = 'heatmap6/s{:.2f}'.format(args.stripe)
    uval = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, args.lval))
    #找到动量空间中的几个代表点
    upatches = numpy.ndarray(32, dtype=Point)
    #anggap = numpy.pi / 2 / 8
    #for idx in range(8):
    #    tanv = numpy.tan((idx + 0.5)*anggap)
    #    yval = numpy.pi * (tanv) / (1 + tanv)
    #    xval = yval / tanv
    #    upatches[idx] = Point(xval, yval, 1)
    #    upatches[8 + idx] = Point(-yval, xval, 1)
    #    upatches[16 + idx] = Point(-xval, -yval, 1)
    #    upatches[24 + idx] = Point(xval, -yval, 1)
    #draw_points(upatches)
    set_stripe(0.)
    upatches[:16] = get_s_band_patches(16)
    upatches[16:] = get_p_band_patches(16)
    draw_points(upatches)
    set_stripe(args.stripe)
    #
    spats = get_s_band_patches(16)
    ppats = get_p_band_patches(16)
    ubas, iubas = inverse_uval(upatches, spats, ppats, uval)
    #
    draw_heatmap(ubas[0, 0, 0, 0, :, :, 4])
    draw_heatmap(ubas[0, 1, 1, 0, :, :, 4])
    draw_heatmap(ubas[1, 1, 1, 1, :, :, 4])
    #c2i = {'s': [0], 'p': [1], '?': [0, 1]}
    #i2c = {0: 's', 1: 'p'}
    #candi = [c2i[bdi] for bdi in args.channel]
    #dims = [len(bdi) for bdi in candi]
    #place_holder = numpy.ndarray(dims)
    #ndit = numpy.nditer(place_holder, flags=['multi_index'])

def main():
    '''入口'''
    parser = argparse.ArgumentParser(
        prog='python3 visualize.py',
        description='visualize U'
    )
    parser.add_argument('-m', '--mode', type=str, required=True, help='drawing mode')
    parser.add_argument('-s', '--stripe', type=float, required=True, help='stripe strength')
    parser.add_argument('-l', '--lval', type=float, required=True, help='lval')
    parser.add_argument('-c', '--channel', type=str, required=True, help='which band')
    parser.add_argument('-n', '--n3idx', type=int, help='require if single mode')
    parser.add_argument('--prefix', type=str,\
        default='scripts/stripe/str', help='saved file prefix')
    args = parser.parse_args()
    #
    mode_dict = {
        'single': draw_single_channel,
        'mixed': draw_mixed_channel,
        'count': draw_count_channel,
        'basis': draw_basis_channel
    }
    mode_dict[args.mode](args)


if __name__ == '__main__':
    main()
