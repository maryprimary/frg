"""布里渊区中patches相关的功能"""

from matplotlib import pyplot
import matplotlib.patches as patches
import matplotlib.path as path


def patches_visualize(pats, lsurface, show):
    '''可视化patches对应的点和费米面
    '''
    pyplot.figure()
    #绘制patches对应的点
    xvals = []
    yvals = []
    for pnt in pats:
        xvals.append(pnt.coord[0])
        yvals.append(pnt.coord[1])
    pyplot.scatter(xvals, yvals, c='g', lw=4)
    #绘制费米面的线
    for seg in lsurface:
        if seg is None:
            continue
        xvals = [_pt.coord[0] for _pt in seg.ends]
        yvals = [_pt.coord[1] for _pt in seg.ends]
        pyplot.plot(xvals, yvals, c='k', lw=1)
    if show == 'show':
        pyplot.show()
    else:
        pyplot.savefig(show)


def district_visualize(ltris, lpatches, show):
    '''可视化切分的效果\n
    ltris是切分的小三角，lpathces是每个小三角对应的编号\n
    show = 'window': 显示在窗口\n
    其他: 保存为这个名字的图片
    '''
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    pyplot.figure()
    for tri, pidx in zip(ltris, lpatches):
        vertex = [ver.coord for ver in tri.vertex] + [(0, 0)]
        codes = [path.Path.LINETO] * len(vertex)
        codes[0] = path.Path.MOVETO
        codes[-1] = path.Path.CLOSEPOLY
        rectp = patches.PathPatch(
            path.Path(vertex, codes),
            facecolor=colors[pidx % 8],
            lw=0)
        pyplot.gca().add_patch(rectp)
    pyplot.gca().relim()
    pyplot.gca().autoscale_view()
    ###
    if show == 'show':
        pyplot.show()
    else:
        pyplot.savefig(show)


def save_to(fname, lpatches):
    '''保存patches，lpatches应该是对应好Rtriangles的顺序的\n
    ```没有直接把pidx放到Rtriabgles的attr里面，这个顺序要对应好```
    '''
    outf = open(fname, 'w')
    #第一行记录长度
    outf.write(str(len(lpatches)) + '\n')
    for pidx in lpatches:
        outf.write(str(pidx) + '\n')


def load_from(fname):
    '''读取patches，注意对应好保存时候的顺序'''
    inf = open(fname, 'r')
    length = int(inf.readline())
    lpatches = []
    for _ in range(length):
        lpatches.append(int(inf.readline()))
    return lpatches
