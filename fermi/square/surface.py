"""这个里面的内容用来提取布里渊区中的表面"""


from basics import Rtriangle
from .brillouin import dispersion

def const_energy_line(rtris, adjs, eng, disp):
    '''获得布里渊区中的等能线\n
    rtris是切分之后的（一部分）布里渊区，应该是一个rtris的列表\n
    adjs是上面每一个小三角的相邻的三角\n
    eng是等能线的能量\n
    disp是色散关系\n
    square: 普通的正方格子\n
    '''
    #色散关系
    dispfun = {
        'square': dispersion
    }[disp]
    eng_dict = {}
    for rtr in rtris:
        kpt = rtr.center.coord
        #能量这里减去需要寻找的能量大小
        eng_dict[rtr] = dispfun(kpt[0], kpt[1]) - eng
    #
    #print(eng_dict)
    edges = []
    for idx, rtr in enumerate(rtris, 0):
        #寻找从负数到正数的边界
        if eng_dict[rtr] > 0.:
            continue
        #查看相邻的几个有没有大于零的
        for eidx, adj in enumerate(adjs[idx], 0):
            #已经到头的时候是不继续添加的
            if adj is None:
                continue
            if eng_dict[adj] > 0:
                edges.append(rtr.edges[eidx])
    return edges
