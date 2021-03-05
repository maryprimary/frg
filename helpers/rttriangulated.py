"""将一个长方形切分成三角形"""

import numpy
from basics import Triangle, Rectangle, Point

def single_rtangle_split(rtg: Rectangle):
    '''将一个正方形切分，返回的顺序是逆时针，最上面的一个是第一个\n
    每个直角三角形的边也是按照逆时针进行排序的
    '''
    rtver = rtg.center
    top = Triangle([rtver, rtg.vertex[3], rtg.vertex[0]])
    left = Triangle([rtver, rtg.vertex[0], rtg.vertex[1]])
    btm = Triangle([rtver, rtg.vertex[1], rtg.vertex[2]])
    right = Triangle([rtver, rtg.vertex[2], rtg.vertex[3]])
    return [top, left, btm, right]


def rectangle_split(rtg: Rectangle, nps):
    '''先将一个正方形切分成很多个小正方形\n
    然后把小正方形切分成三角\n
    这里比较窄的一边会变成nps份，比较宽的一边会增加相应的倍数
    '''
    if rtg.width >= rtg.height:#如果比较宽
        nshape = (nps * numpy.floor_divide(rtg.width, rtg.height), nps)
    else:
        nshape = (nps, nps * numpy.floor_divide(rtg.height, rtg.width))
    nshape = (numpy.int(nshape[0]), numpy.int(nshape[1]))
    lsqus = numpy.ndarray(nshape, dtype=Rectangle)
    width = rtg.width / nshape[0]
    height = rtg.height / nshape[1]
    statx, staty = rtg.vertex[1].coord
    #从左下角开始，先水平的顺序进行标号
    # [idx2]
    # nps
    # ...
    # 1   x  x
    # 0   x  x
    #     0  1  ...  nps [idx1]
    for idx1 in range(nshape[0]):
        for idx2 in range(nshape[1]):
            xct = (idx1 + 0.5) * width + statx
            yct = (idx2 + 0.5) * height + staty
            lsqus[idx1, idx2] = \
                Rectangle(Point(xct, yct, 1), width, height)
    #将每个小正方切分
    ltris = numpy.ndarray((nshape[0], nshape[1], 4), dtype=Triangle)
    for idx1 in range(nshape[0]):
        for idx2 in range(nshape[1]):
            ltris[idx1, idx2, :] = single_rtangle_split(lsqus[idx1, idx2])
    #找到每个小三角形挨着的小三角形
    #这个时候的顺序应该是和小三角形的边的顺序一致的
    ladjs = numpy.ndarray((nshape[0], nshape[1], 4), dtype=numpy.object)
    for idx1 in range(nshape[0]):
        for idx2 in range(nshape[1]):
            #上面的那个小三角，第一条边是右侧，第二条是上边
            adjs = [ltris[idx1, idx2, 3], None, ltris[idx1, idx2, 1]]
            if idx2 + 1 < nshape[1]:
                adjs[1] = ltris[idx1, idx2 + 1, 2]
            ladjs[idx1, idx2, 0] = adjs
            #左面的小三角，第一条边是上边，第二条是左面
            adjs = [ltris[idx1, idx2, 0], None, ltris[idx1, idx2, 2]]
            if idx1 - 1 >= 0:
                adjs[1] = ltris[idx1 -1, idx2, 3]
            ladjs[idx1, idx2, 1] = adjs
            #下面的小三角，第一条边是左面，第二条是下面
            adjs = [ltris[idx1, idx2, 1], None, ltris[idx1, idx2, 3]]
            if idx2 - 1 >= 0:
                adjs[1] = ltris[idx1, idx2 - 1, 0]
            ladjs[idx1, idx2, 2] = adjs
            #右面的小三角，第一条边是下面，第二条是右边
            adjs = [ltris[idx1, idx2, 2], None, ltris[idx1, idx2, 0]]
            if idx1 + 1 < nshape[0]:
                adjs[1] = ltris[idx1 + 1, idx2, 1]
            ladjs[idx1, idx2, 3] = adjs
    #将小三角和它对应的近邻都reshape
    ltris = numpy.reshape(ltris, nshape[0]*nshape[1]*4)
    ladjs = numpy.reshape(ladjs, nshape[0]*nshape[1]*4)
    return ltris, ladjs


def save_to(fname, rtg: Rectangle, nps, ltris, ladjs):
    '''保存切分后的结果'''
    if rtg.width >= rtg.height:#如果比较宽
        nshape = (nps * numpy.floor_divide(rtg.width, rtg.height), nps)
    else:
        nshape = (nps, nps * numpy.floor_divide(rtg.height, rtg.width))
    nshape = (numpy.int(nshape[0]), numpy.int(nshape[1]))
    if len(ltris) != nshape[0] * nshape[1] * 4:
        raise ValueError('nps和ltris的大小对应不上')
    #
    outf = open(fname, 'w')
    #记录长方形相关的信息
    outf.write('Rectangle\n')
    outf.write('center: {0:.12f}, {1:.12f}\n'.\
        format(rtg.center.coord[0], rtg.center.coord[1]))
    outf.write('width: {0:.12f}\n'.format(rtg.width))
    outf.write('height: {0:.12f}\n'.format(rtg.height))
    outf.write('nps: {0:d}\n'.format(nps))
    outf.write('End Rectangle\n')
    #记录和三角形有换的信息
    tri_to_idx = {}
    outf.write('Triangle\n')
    for idx, tri in enumerate(ltris, 0):
        tri_to_idx[tri] = idx
        tristr = '\tindex: {0:d} '.format(idx)
        cntcor = tri.vertex[0].coord
        tristr += ':ver0: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = tri.vertex[1].coord
        tristr += ':ver1: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = tri.vertex[2].coord
        tristr += ':ver2: {0:.12f}, {1:.12f}\n'.format(cntcor[0], cntcor[1])
        outf.write(tristr)
    outf.write('End Triangle\n')
    #记录和相邻有关的信息
    outf.write('Adjtri\n')
    #没有的相邻用-1
    tri_to_idx[None] = -1
    for idx, adjs in enumerate(ladjs, 0):
        adjint = [tri_to_idx[tria] for tria in adjs]
        outf.write('\tindex: {3:d} :adjs: {0:d}, {1:d}, {2:d}\n'.\
            format(adjint[0], adjint[1], adjint[2], idx))
    outf.write('End Adjtri\n')


def load_rectangle(lines, rtg, nps, ltris, ladjs):
    '''加载长方形'''
    cntcor = None
    width = None
    height = None
    nps = None
    for line in lines:
        if line.startswith('center: '):
            corstr = line[8:].split(',')
            cntcor = (float(corstr[0]), float(corstr[1]))
        if line.startswith('width: '):
            width = float(line[7:])
        if line.startswith('height: '):
            height = float(line[8:])
        if line.startswith('nps: '):
            nps = int(line[4:])
    rtg = Rectangle(Point(cntcor[0], cntcor[1], 1), width, height)
    return rtg, nps, ltris, ladjs


def load_triangle(lines, rtg, nps, ltris, ladjs):
    '''加载三角形\n
    加载之后会按照idx的顺序组成一个列表，这样加载adj的时候会按照列表的顺序找到
    相应的Triangle
    '''
    if nps == -1 or rtg is None:
        raise ValueError('没有初始化Square')
    #
    if rtg.width >= rtg.height:#如果比较宽
        nshape = (nps * numpy.floor_divide(rtg.width, rtg.height), nps)
    else:
        nshape = (nps, nps * numpy.floor_divide(rtg.height, rtg.width))
    nshape = (numpy.int(nshape[0]), numpy.int(nshape[1]))
    #
    ltris = numpy.ndarray(nshape[0]*nshape[1]*4, dtype=Triangle)
    if len(lines) != nshape[0]*nshape[1]*4:
        raise ValueError('ltris读取时长度对应不上')
    for line in lines:
        lstr = line.split(':')
        idx = int(lstr[1])
        v0str = lstr[3].split(',')
        v1str = lstr[5].split(',')
        v2str = lstr[7].split(',')
        v0pt = Point(float(v0str[0]), float(v0str[1]), 1)
        v1pt = Point(float(v1str[0]), float(v1str[1]), 1)
        v2pt = Point(float(v2str[0]), float(v2str[1]), 1)
        tri = Triangle([v0pt, v1pt, v2pt])
        ltris[idx] = tri
    return rtg, nps, ltris, ladjs


def load_adjtri(lines, rtg, nps, ltris, ladjs):
    '''加载相邻的三角形'''
    if nps == -1 or rtg is None:
        raise ValueError('没有初始化Square')
    #
    if rtg.width >= rtg.height:#如果比较宽
        nshape = (nps * numpy.floor_divide(rtg.width, rtg.height), nps)
    else:
        nshape = (nps, nps * numpy.floor_divide(rtg.height, rtg.width))
    nshape = (numpy.int(nshape[0]), numpy.int(nshape[1]))
    #
    ladjs = numpy.ndarray(nshape[0]*nshape[1]*4, dtype=numpy.object)
    for line in lines:
        lstr = line.split(':')
        idx = int(lstr[1])
        adjints = [int(adjs) for adjs in lstr[3].split(',')]
        adjl = []
        for adji in adjints:
            if adji == -1:
                adjl.append(None)
            else:
                adjl.append(ltris[adji])
        ladjs[idx] = adjl
    return rtg, nps, ltris, ladjs


def load_from(fname):
    '''从结果文件读取'''
    #初始化变量
    inf = open(fname, 'r')
    blockname = None
    blockent = []
    rtg = None
    nps = -1
    ltris = None
    ladjs = None
    #不同段落的处理
    parser = {
        'Rectangle': load_rectangle,
        'Triangle': load_triangle,
        'Adjtri': load_adjtri}
    #读取文件
    line = inf.readline()
    while line:
        sline = line.strip()
        line = inf.readline()
        #跳过空行
        if not sline:
            continue
        #一个段落的开头
        if not blockname:
            blockname = sline
        #段落结束
        elif sline == 'End %s' % blockname:
            rtg, nps, ltris, ladjs =\
                parser[blockname](blockent, rtg, nps, ltris, ladjs)
            blockname = None
            blockent = []
        #没结束的时候读取
        else:
            blockent.append(sline)
    return rtg, nps, ltris, ladjs
