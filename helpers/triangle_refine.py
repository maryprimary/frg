"""
将一个三角形细分
"""


import numpy
from basics.point import middle_point
from basics.triangle import Rtriangle, Eqtriangle


def split_rttriangle(rtri: Rtriangle):
    '''将直角三角形分成4份，4个还是直角三角形'''
    rtvexs = [rtri.vertex[0]]
    rtvexs.append(middle_point(rtri.vertex[0], rtri.vertex[1]))
    rtvexs.append(middle_point(rtri.vertex[1], rtri.vertex[2]))
    rtvexs.append(middle_point(rtri.vertex[2], rtri.vertex[0]))
    newtris = [Rtriangle(rtvexs[0], rtvexs[1], rtvexs[3])]
    newtris.append(Rtriangle(rtvexs[1], rtri.vertex[1], rtvexs[2]))
    newtris.append(Rtriangle(rtvexs[2], rtvexs[3], rtvexs[1]))
    newtris.append(Rtriangle(rtvexs[3], rtvexs[2], rtri.vertex[2]))
    return newtris



def split_eqtriangle(etri: Eqtriangle):
    '''将正三角形分成4份，4个还是正三角形'''
    rtvexs = [etri.vertex[0]]
    rtvexs.append(middle_point(etri.vertex[0], etri.vertex[1]))
    rtvexs.append(middle_point(etri.vertex[1], etri.vertex[2]))
    rtvexs.append(middle_point(etri.vertex[2], etri.vertex[0]))
    newtris = [Rtriangle(rtvexs[0], rtvexs[1], rtvexs[3])]
    newtris.append(Rtriangle(rtvexs[1], etri.vertex[1], rtvexs[2]))
    newtris.append(Rtriangle(rtvexs[2], rtvexs[3], rtvexs[1]))
    newtris.append(Rtriangle(rtvexs[3], rtvexs[2], etri.vertex[2]))
    return newtris


def lrttris_refine(ltris):
    '''将一个ltris中的所有的tri切分'''
    newltris = numpy.ndarray(4*len(ltris), dtype=Rtriangle)
    for idx, tri in enumerate(ltris):
        newltris[4*idx:4*idx+4] = split_rttriangle(tri)
    return newltris


def leqtris_refine(ltris):
    '''将一个ltris中的所有tri切分'''
    newltris = numpy.ndarray(4*len(ltris), dtype=Eqtriangle)
    for idx, tri in enumerate(ltris):
        newltris[4*idx:4*idx+4] = split_rttriangle(tri)
    return newltris


def __rf_is_adjoint(verts1, verts2):
    '''验证两个三角型是不是相邻'''
    adjv = 0
    for ver1 in verts1:
        x1v, y1v = ver1.coord[0], ver1.coord[1]
        for ver2 in verts2:
            x2v, y2v = ver2.coord[0], ver2.coord[1]
            if numpy.allclose([x1v, y1v], [x2v, y2v], 0.0, 1e-8):
                adjv += 1
                break
    if adjv == 2:
        return True
    return False


def find_adjs_by_adjoint(ltris):
    '''通过相邻的点来判断adj，如果有两个重合的点，那么这两个三角形是相邻的'''
    ladjs = numpy.ndarray(len(ltris), dtype=numpy.object)
    def __add_to_idx(idx1, idx2):
        '''将idx2添加到idx1的adj中'''
        #如果没有相邻，增加一个list
        if ladjs[idx1] is None:
            ladjs[idx1] = []
        #如果已经够了三个，则表示出了问题
        if len(ladjs) == 3:
            raise ValueError("超过三个")
        ladjs[idx1].append(ltris[idx2])
    for idx, tri in enumerate(ltris):
        vert1 = tri.vertex
        #如果已经够了三个，跳过
        if len(ladjs) == 3:
            continue
        for soidx, otri in enumerate(ltris[idx+1:]):
            vert2 = otri.vertex
            oidx = soidx + idx + 1
            if __rf_is_adjoint(vert1, vert2):
                __add_to_idx(idx, oidx)
                __add_to_idx(oidx, idx)
    for adj in ladjs:
        while len(adj) < 3:
            adj.append(None)
    return ladjs
