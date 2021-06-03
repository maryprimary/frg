'''显示结果'''

import numpy
from helpers.drawer import draw_heatmap

def main():
    '''入口'''
    lval = 5.20
    rpath = 'heatmap8/avsb'
    uval = numpy.load('{0}/{1:.2f}U.npy'.format(rpath, lval))
    draw_heatmap(uval[0, 0, 0, 0, :, :, 0])
    draw_heatmap(uval[1, 1, 1, 1, :, :, 0])
    draw_heatmap(uval[1, 0, 0, 1, :, :, 0])
    draw_heatmap(uval[0, 1, 1, 0, :, :, 0])


if __name__ == '__main__':
    main()
