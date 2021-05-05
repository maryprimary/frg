"""将一个正三角形或者正六边形切分成小三角形"""


import numpy
from basics import Eqtriangle, Point, Hexagon, Segment
from basics.point import middle_point
#from .drawer import draw_components


def eqtriangle_split(eqt: Eqtriangle, nps):
    '''将一个正三角形切分成nps个块'''
    #从两个边开始计算
    #nps条分界线，包含本来的底边
    btmlines = []
    for idx in range(nps):
        coef = (idx+1) / nps
        coefb = 1 - coef
        btmlines.append(
            Segment(
                middle_point(eqt.vertex[0], eqt.vertex[1], coefb, coef),
                middle_point(eqt.vertex[0], eqt.vertex[2], coefb, coef)
            )
        )
    #从这些底边开始切分成新的三角形，找到每条底边上的点
    btmpts = []
    for idx, btml in enumerate(btmlines):
        num = idx + 1
        ptslist = [btml.ends[0]]
        for nid in range(num):
            coef = (nid + 1) / num
            coefb = 1 - coef
            ptslist.append(middle_point(btml.ends[0], btml.ends[1], coefb, coef))
        btmpts.append(ptslist)
    #从上面的点开始，添加新的等边三角形
    eqtris = [Eqtriangle([eqt.vertex[0], btmpts[0][0], btmpts[0][1]])]
    for idx in range(1, nps):
        num = idx + 1
        #每一层有idx+2个点在ptlist里面实际上
        #底在这条线上的三角形
        for ptidx in range(num):
            eqtris.append(
                Eqtriangle(
                    [btmpts[idx-1][ptidx], btmpts[idx][ptidx],btmpts[idx][ptidx+1]]
                )
            )
            #顶在这条线上的三角形
            if ptidx < idx:
                eqtris.append(
                    Eqtriangle(
                        [btmpts[idx-1][ptidx], btmpts[idx][ptidx+1], btmpts[idx-1][ptidx+1]]
                    )
                )
    return btmlines, btmpts, eqtris


def _trapezoid_split(lseg: Segment, sseg: Segment, length):
    '''将一个等腰梯形切分成等边三角形,底和x轴平行\n
    切分六边形的时候，把半和下半都处理成等腰梯形\n
    需要输入等腰梯形的两个底，第一个是比较长的底，第二个是短的，\n
    然后还需要等边三角形的边长
    '''
    assert numpy.isclose(lseg.length - sseg.length, length)
    parts = sseg.length / length
    assert numpy.isclose(parts, numpy.round(parts))
    parts = numpy.int(numpy.round(parts))
    #
    lpt = numpy.ndarray(parts+2, dtype=Point)
    spt = numpy.ndarray(parts+1, dtype=Point)
    #
    spt[0] = sseg.ends[0]
    lpt[0] = lseg.ends[0]
    for idx in range(parts):
        coef = (idx + 1) / parts
        coefb = 1 - coef
        spt[idx+1] = middle_point(sseg.ends[0], sseg.ends[1], coefb, coef)
        coef = (idx + 1) / (parts+1)
        coefb = 1 - coef
        lpt[idx+1] = middle_point(lseg.ends[0], lseg.ends[1], coefb, coef)
    lpt[-1] = lseg.ends[1]
    #pts = numpy.concatenate([spt, lpt])
    #draw_components(pts, [sseg, lseg], [])
    #从长的开始计，第二个点到倒数第二个点
    eqtris = [Eqtriangle([lpt[0], spt[0], lpt[1]])]
    for idx in range(1, parts+1):
        eqtris.append(Eqtriangle([lpt[idx], spt[idx], spt[idx-1]]))
        eqtris.append(Eqtriangle([lpt[idx], spt[idx], lpt[idx+1]]))
    #draw_components(pts, [sseg, lseg], eqtris)
    return eqtris


def hexagon_split(hexa: Hexagon, nps):
    '''
    将一个六边形切分成小的正三角形
    '''
    #六边形分成两个等腰梯形
    topedge = hexa.edges[1]
    midedge = Segment(hexa.vertex[0], hexa.vertex[3])
    btmedge = hexa.edges[4]
    #所有的小三角形
    eqtris = []
    length = numpy.sum([edg.length for edg in hexa.edges])
    length /= 6
    length /= nps
    #先处理上面半个梯形
    #注意这里由于Hexagon是逆时针对顶点进行的排序
    #所以梯形和三角形增长的方向是从右往左
    llines = [topedge]
    for idx in range(nps):
        coef = (idx + 1) / nps
        coefb = 1 - coef
        llines.append(
            Segment(
                middle_point(topedge.ends[0], midedge.ends[0], coefb, coef),
                middle_point(topedge.ends[1], midedge.ends[1], coefb, coef)
            )
        )
    #llines中一共有nps+1个线
    #注意由于梯形的点是逆时针的，所以这边三角形的增长也是从右向左的
    for idx in range(nps):
        ets = _trapezoid_split(llines[idx+1], llines[idx], length)
        assert len(ets) == 2*nps + 1 + 2*idx
        eqtris.extend(ets)
    #再处理下面半个梯形的
    #这里需要注意下半个梯形的点的顺序问题
    llines = [midedge]
    for idx in range(nps):
        coef = (idx + 1) / nps
        coefb = 1 - coef
        llines.append(
            Segment(
                middle_point(midedge.ends[0], btmedge.ends[1], coefb, coef),
                middle_point(midedge.ends[1], btmedge.ends[0], coefb, coef)
            )
        )
    #llines中共有nps+1条线
    for idx in range(nps):
        ets = _trapezoid_split(llines[idx], llines[idx+1], length)
        assert len(ets) == 4*nps - 1 - 2*idx
        eqtris.extend(ets)
    #开始计算相邻的三角形
    #这个时候，所有的编号都是从右向左，一层一层在增加的
    ladjs = numpy.ndarray(len(eqtris), dtype=object)
    # Warning: 这个ladjs中的相邻的三角形的顺序一定和Eqtriangle的边的顺序是一样的
    #先是nps个上半层，其中三角形的数量是2*nps+1+2*idx
    #idx从0到nps-1
    #先处理最上面的一层的相邻
    #最上面一层就是从[0,2*nps+1)这个范围内的所有
    for idx in range(2*nps+1):
        #每个三角都有三条边，于是有三个相邻
        #这些是底边在下面的相邻的有下一个，和下一行正对的
        #上半部分中，底边在下面的三角形的点是按照右下-上-左下的顺序来的
        if idx % 2 == 0:
            rightone = None if idx == 0 else eqtris[idx-1]
            leftone = None if idx == 2*nps else eqtris[idx+1]
            ladjs[idx] = [rightone, leftone, eqtris[idx+2*nps+2]]
        #这些是底边在上面的，相邻的有上一个和下一个
        #上半部分中，底边在上面的三角形是按照下-左上-右上的顺序来的
        else:
            ladjs[idx] = [eqtris[idx+1], None, eqtris[idx-1]]
    #之后，还有nps-2个上半部分的梯形
    startrange = 2*nps+1
    for npsidx in range(1, nps):
        #startrange是前面0～npsidx-1行包含的数目
        #这行包含的数目
        #如果到了最后一个，下面的一行的三角形的数量和这一行是一样多的
        #计算的时候不需要多添加一个1
        offset = 1 if npsidx != nps-1 else 0
        stoprange = 2*nps+1+2*npsidx
        for idx in range(stoprange):
            tot_idx = startrange + idx
            #这些是底边在下面的相邻的有下一个，和下一行正对的
            #上半部分中，底边在下面的三角形的点是按照右下-上-左下的顺序来的
            if idx % 2 == 0:
                rightone = None if idx == 0 else eqtris[tot_idx-1]
                leftone = None if idx == stoprange - 1 else eqtris[tot_idx+1]
                ladjs[tot_idx] =\
                    [rightone, leftone, eqtris[tot_idx+stoprange+offset]]
            else:
                #上半部分中，底边在上面的三角形是按照下-左上-右上的顺序来的
                ladjs[tot_idx] =\
                    [eqtris[tot_idx+1], eqtris[tot_idx-stoprange+1],\
                        eqtris[tot_idx-1]]
        startrange += stoprange
    #下班部分从后往前开始处理
    #最下面的一个梯形
    for idx in range(2*nps+1):
        #每个三角都有三条边，于是有三个相邻
        #这些是底边在上面的，相邻的有下一个，上面对的，还有上一个
        #下半部分中，底边在上面的三角形是按照右上-下-左上的顺序来的
        tot_idx = len(eqtris) - (2*nps+1) + idx
        if idx % 2 == 0:
            rightone = None if idx == 0 else eqtris[tot_idx-1]
            leftone = None if idx == 2*nps else eqtris[tot_idx+1]
            ladjs[tot_idx] = [rightone, leftone, eqtris[tot_idx-2-2*nps]]
        #下半部分中，底边在下面的三角形是按照上-左下-右下的顺序来的
        else:
            ladjs[tot_idx] = [eqtris[tot_idx+1], None, eqtris[tot_idx-1]]
    startrange = len(eqtris) - (2*nps+1)
    for npsidx in range(1, nps):
        offset = 1 if npsidx != nps-1 else 0
        stoprange = 2*nps+1+2*npsidx
        for idx in range(stoprange):
            tot_idx = startrange - stoprange + idx
            #下半部分中，底边在上面的三角形是按照右上-下-左上的顺序来的
            if idx % 2 == 0:
                rightone = None if idx == 0 else eqtris[tot_idx-1]
                leftone = None if idx == stoprange - 1 else eqtris[tot_idx+1]
                ladjs[tot_idx] = [rightone, leftone, eqtris[tot_idx-offset-stoprange]]
            #下半部分中，底边在下面的三角形是按照上-左下-右下的顺序来的
            else:
                ladjs[tot_idx] = [eqtris[tot_idx+1], eqtris[tot_idx+stoprange-1], eqtris[tot_idx-1]]
        startrange = startrange - stoprange
    return numpy.array(eqtris, dtype=Eqtriangle), ladjs


def save_to(fname, hexa: Hexagon, nps, ltris, ladjs):
    '''将结果保存到fname'''
    outf = open(fname, 'w')
    #记录六边形相关的信息
    cntcor = hexa.center.coord
    outf.write('Hexagon\n')
    outf.write('center: {0:.12f}, {1:.12f}\n'.format(cntcor[0], cntcor[1]))
    outf.write('height: {0:.12f}\n'.format(hexa.height))
    outf.write('nps: {0:d}\n'.format(nps))
    outf.write('End Hexagon\n')
    #记录和三角形有关的信息
    tri_to_idx = {}
    outf.write('Eqtriangle\n')
    for idx, rtg in enumerate(ltris, 0):
        tri_to_idx[rtg] = idx
        tristr = '\tindex: {0:d} '.format(idx)
        cntcor = rtg.vertex[0].coord
        tristr += ':ver0: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = rtg.vertex[1].coord
        tristr += ':ver1: {0:.12f}, {1:.12f} '.format(cntcor[0], cntcor[1])
        cntcor = rtg.vertex[2].coord
        tristr += ':ver2: {0:.12f}, {1:.12f}\n'.format(cntcor[0], cntcor[1])
        outf.write(tristr)
    outf.write('End Eqtriangle\n')
    #记录和相邻有关的信息
    outf.write('Adjtri\n')
    #没有的相邻用-1
    tri_to_idx[None] = -1
    for idx, adjs in enumerate(ladjs, 0):
        adjint = [tri_to_idx[tria] for tria in adjs]
        outf.write('\tindex: {3:d} :adjs: {0:d}, {1:d}, {2:d}\n'.\
            format(adjint[0], adjint[1], adjint[2], idx))
    outf.write('End Adjtri\n')


def load_hexagon(lines, hexa, nps, ltris, ladjs):
    '''读取六边形的段落'''
    cntcor = None
    height = None
    nps = None
    for line in lines:
        if line.startswith('center: '):
            corstr = line[8:].split(',')
            cntcor = (float(corstr[0]), float(corstr[1]))
        if line.startswith('height: '):
            height = float(line[8:])
        if line.startswith('nps: '):
            nps = int(line[4:])
    hexa = Hexagon(Point(cntcor[0], cntcor[1], 1), height)
    return hexa, nps, ltris, ladjs


def load_eqtriangle(lines, hexa, nps, ltris, ladjs):
    '''读取正三角形'''
    if nps == -1 or hexa is None:
        raise ValueError('没有初始化Square')
    idx_to_tri = {}
    for line in lines:
        lstr = line.split(':')
        idx = int(lstr[1])
        v0str = lstr[3].split(',')
        v1str = lstr[5].split(',')
        v2str = lstr[7].split(',')
        v0pt = Point(float(v0str[0]), float(v0str[1]), 1)
        v1pt = Point(float(v1str[0]), float(v1str[1]), 1)
        v2pt = Point(float(v2str[0]), float(v2str[1]), 1)
        rtri = Eqtriangle([v0pt, v1pt, v2pt])
        idx_to_tri[idx] = rtri
    #最上面是2*nps+1，最下面是4*nps-1，一共nps个
    num = (2*nps+1) + (4*nps-1)
    num = num * nps
    ltris = numpy.ndarray(num, dtype=Eqtriangle)
    #print(len(idx_to_tri), num)
    if len(idx_to_tri) != num:
        raise ValueError('nps和Rtriangle大小不一致')
    for idx in range(num):
        ltris[idx] = idx_to_tri[idx]
    return hexa, nps, ltris, ladjs


def load_adjtri(lines, hexa, nps, ltris, ladjs):
    '''读取相邻的位置'''
    num = (2*nps+1) + (4*nps-1)
    num = num * nps
    ladjs = numpy.ndarray(num, dtype=numpy.object)
    #print(len(lines))
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
    return hexa, nps, ltris, ladjs


def load_from(fname):
    '''加载保存的文件'''
    #初始化变量
    inf = open(fname, 'r')
    blockname = None
    blockent = []
    hexa = None
    nps = -1
    ltris = None
    ladjs = None
    #不同段落的处理
    parser = {
        'Hexagon': load_hexagon,
        'Eqtriangle': load_eqtriangle,
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
            hexa, nps, ltris, ladjs =\
                parser[blockname](blockent, hexa, nps, ltris, ladjs)
            blockname = None
            blockent = []
        #没结束的时候读取
        else:
            blockent.append(sline)
    return hexa, nps, ltris, ladjs
