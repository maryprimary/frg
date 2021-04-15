"""这个里面的内容用来提取布里渊区中的表面"""


def const_energy_line(rtris, adjs, eng, dispfun):
    '''获得布里渊区中的等能线\n
    rtris是切分之后的（一部分）布里渊区，应该是一个rtris的列表\n
    adjs是上面每一个小三角的相邻的三角\n
    eng是等能线的能量\n
    disp是色散关系\n
    '''
    eng_dict = {}
    for rtr in rtris:
        kpt = rtr.center.coord
        #能量这里减去需要寻找的能量大小
        eng_dict[rtr] = dispfun(kpt[0], kpt[1]) - eng
    #
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


def const_energy_line_in_patches(rtris, adjs, lpats, eng, dispfun):
    '''获得布里渊区中的等能线\n
    rtris是切分之后的（一部分）布里渊区，应该是一个rtris的列表\n
    adjs是上面每一个小三角的相邻的三角\n
    lpats是每个rtris处于的patch的idx\n
    eng是等能线的能量\n
    disp是色散关系\n
    '''
    eng_dict = {}
    for rtr in rtris:
        kpt = rtr.center.coord
        #能量这里减去需要寻找的能量大小
        eng_dict[rtr] = dispfun(kpt[0], kpt[1]) - eng
    #
    edges = []
    epidx = []
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
                #如果大于0和小于0的不在一个patch，会产生一些混淆。但是没明显的影响。
                epidx.append(lpats[idx])
    return edges, epidx


def filling_factor(ltris, dispfun):
    '''被占据的面积'''
    tot_area = len(ltris)
    results = []
    for tri in ltris:
        kpt = tri.center.coord
        if dispfun(kpt[0], kpt[1]) < 0.0:
            results.append(tri)
    return len(results) / tot_area, results
