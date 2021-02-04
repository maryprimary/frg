"""绘图的工具"""

#pylint: disable=unused-import

from typing import List
from matplotlib import pyplot
import matplotlib.patches as patches
import matplotlib.path as path
from basics import Square, Rectangle, Segment, Point


def draw_points(pts: List[Point]):
    '''绘制散点图'''
    pyplot.figure()
    xvals = []
    yvals = []
    for pnt in pts:
        xvals.append(pnt.coord[0])
        yvals.append(pnt.coord[1])
    pyplot.scatter(xvals, yvals, lw=4)
    pyplot.show()


def draw_lines(sgs: List[Segment]):
    '''绘制一组线'''
    pyplot.figure()
    for seg in sgs:
        if seg is None:
            continue
        xvals = [_pt.coord[0] for _pt in seg.ends]
        yvals = [_pt.coord[1] for _pt in seg.ends]
        pyplot.plot(xvals, yvals)
    pyplot.show()


def draw_polygons(rects):
    '''绘制一组正方形'''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    pyplot.figure()
    for idx, rect in enumerate(rects, 0):
        if rect is None:#允许有None作为空的占位
            continue
        vertex = [ver.coord for ver in rect.vertex] + [(0, 0)]
        codes = [path.Path.LINETO] * len(vertex)
        codes[0] = path.Path.MOVETO
        codes[-1] = path.Path.CLOSEPOLY
        rectp = patches.PathPatch(
            path.Path(vertex, codes),
            facecolor=colors[idx % 8],
            lw=2)
        pyplot.gca().add_patch(rectp)
    pyplot.gca().relim()
    pyplot.gca().autoscale_view()
    pyplot.show()


def draw_heatmap(arra, save='show'):
    '''绘制一个热度图'''
    pyplot.figure()
    axe = pyplot.gca()
    img = axe.imshow(arra, cmap='RdBu')
    cbar = axe.figure.colorbar(img, ax=axe)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")
    if save == "show":
        pyplot.show()
    else:
        pyplot.savefig(save)
