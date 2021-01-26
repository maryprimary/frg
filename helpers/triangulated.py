"""将一个图形切分成三角形"""

import numpy
from basics import Rtriangle, Square, Point

def single_square_split(squ: Square):
    '''将一个正方形切分，返回的顺序是逆时针，最上面的一个是第一个\n
    每个直角三角形的边也是按照逆时针进行排序的
    '''
    rtver = squ.center
    top = Rtriangle(rtver, squ.vertex[3], squ.vertex[0])
    left = Rtriangle(rtver, squ.vertex[0], squ.vertex[1])
    btm = Rtriangle(rtver, squ.vertex[1], squ.vertex[2])
    right = Rtriangle(rtver, squ.vertex[2], squ.vertex[3])
    return [top, left, btm, right]


def square_split(squ: Square, nps):
    '''先将一个正方形切分成很多个小正方形\n
    然后把小正方形切分成三角
    '''
    lsqus = numpy.ndarray((nps, nps), dtype=Square)
    width = squ.width / nps
    statx, staty = squ.vertex[1].coord
    #从左下角开始，先水平的顺序进行标号
    # [idx2]
    # nps
    # ...
    # 1   x  x
    # 0   x  x
    #     0  1  ...  nps [idx1]
    for idx1 in range(nps):
        for idx2 in range(nps):
            xct = (idx1 + 0.5) * width + statx
            yct = (idx2 + 0.5) * width + staty
            lsqus[idx1, idx2] = Square(Point(xct, yct, 1), width)
    #将每个小正方切分
    ltris = numpy.ndarray((nps, nps, 4), dtype=Rtriangle)
    for idx1 in range(nps):
        for idx2 in range(nps):
            ltris[idx1, idx2] = single_square_split(lsqus[idx1, idx2])
    #找到每个小三角形挨着的小三角形
    #这个时候的顺序应该是和小三角形的边的顺序一致的
    ladjs = numpy.ndarray((nps, nps, 4), dtype=numpy.object)
    for idx1 in range(nps):
        for idx2 in range(nps):
            #上面的那个小三角，第一条边是右侧，第二条是上边
            adjs = [ltris[idx1, idx2, 3], None, ltris[idx1, idx2, 1]]
            if idx2 + 1 < nps:
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
            if idx1 + 1 < nps:
                adjs[1] = ltris[idx1 + 1, idx2, 1]
            ladjs[idx1, idx2, 3] = adjs
    #将小三角和它对应的近邻都reshape
    ltris = numpy.reshape(ltris, nps*nps*4)
    ladjs = numpy.reshape(ladjs, nps*nps*4)
    return ltris, ladjs


def save_to(fname, squ: Square, nps, ltris, ladjs):
    '''将结果保存到文件'''
    outf = open(fname, 'w')
    #记录正方形相关的信息
    cntcor = squ.center.coord
    outf.write('Square\n')
    outf.write('center: {0:.12f}, {1:.12f}\n'.format(cntcor[0], cntcor[1]))
    outf.write('width: {0:.12f}\n'.format(squ.width))
    outf.write('nps: {0:d}\n'.format(nps))
    outf.write('End Square\n')
    #记录和三角形有关的信息
    tri_to_idx = {}
    outf.write('Rtriangle\n')
    for idx, rtg in enumerate(ltris, 0):
        tri_to_idx[rtg] = idx
        tristr = '\tindex: {0:d} '.format(idx)
        cntcor = rtg.vertex[0].coord
        tristr += ':rtvec: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = rtg.vertex[1].coord
        tristr += ':ver1: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = rtg.vertex[2].coord
        tristr += ':ver2: {0:.12f}, {1:.12f}\n'.format(cntcor[0], cntcor[1])
        outf.write(tristr)
    outf.write('End Rtriangle\n')
    #记录和相邻有关的信息
    outf.write('Adjtri\n')
    #没有的相邻用-1
    tri_to_idx[None] = -1
    for idx, adjs in enumerate(ladjs, 0):
        adjint = [tri_to_idx[tria] for tria in adjs]
        outf.write('\tindex: {3:d} :adjs: {0:d}, {1:d}, {2:d}\n'.\
            format(adjint[0], adjint[1], adjint[2], idx))
    outf.write('End Adjtri\n')



def load_square(lines, squ, nps, ltris, ladjs):
    '''解析处理正方形的段落'''
    cntcor = None
    width = None
    nps = None
    for line in lines:
        if line.startswith('center: '):
            corstr = line[8:].split(',')
            cntcor = (float(corstr[0]), float(corstr[1]))
        if line.startswith('width: '):
            width = float(line[7:])
        if line.startswith('nps: '):
            nps = int(line[4:])
    squ = Square(Point(cntcor[0], cntcor[1], 1), width)
    return squ, nps, ltris, ladjs


def load_rtriangle(lines, squ, nps, ltris, ladjs):
    '''解析处理三角切分的段落\n
    这个处理完之后会是一个按照idx进行排序的list
    '''
    if nps == -1 or squ is None:
        raise ValueError('没有初始化Square')
    idx_to_tri = {}
    for line in lines:
        lstr = line.split(':')
        idx = int(lstr[1])
        rtvstr = lstr[3].split(',')
        v1str = lstr[5].split(',')
        v2str = lstr[7].split(',')
        rvpt = Point(float(rtvstr[0]), float(rtvstr[1]), 1)
        v1pt = Point(float(v1str[0]), float(v1str[1]), 1)
        v2pt = Point(float(v2str[0]), float(v2str[1]), 1)
        rtri = Rtriangle(rvpt, v1pt, v2pt)
        idx_to_tri[idx] = rtri
    ltris = numpy.ndarray(nps*nps*4, dtype=Rtriangle)
    if len(idx_to_tri) != nps*nps*4:
        raise ValueError('nps和Rtriangle大小不一致')
    for idx in range(nps*nps*4):
        ltris[idx] = idx_to_tri[idx]
    return squ, nps, ltris, ladjs


def load_adjtri(lines, squ, nps, ltris, ladjs):
    '''解析处理相邻三角的段落\n
    现在的ltris应该是按照ladjs中的编号进行排序的'''
    ladjs = numpy.ndarray(nps*nps*4, dtype=numpy.object)
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
    return squ, nps, ltris, ladjs

def load_from(fname):
    '''从结果文件读取'''
    #初始化变量
    inf = open(fname, 'r')
    blockname = None
    blockent = []
    squ = None
    nps = -1
    ltris = None
    ladjs = None
    #不同段落的处理
    parser = {
        'Square': load_square,
        'Rtriangle': load_rtriangle,
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
            squ, nps, ltris, ladjs =\
                parser[blockname](blockent, squ, nps, ltris, ladjs)
            blockname = None
            blockent = []
        #没结束的时候读取
        else:
            blockent.append(sline)
    return squ, nps, ltris, ladjs
