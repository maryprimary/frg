"""表示平面上一个点"""

class Point():
    """type: 1=直角坐标，2=极坐标
    直角坐标val1=x，val2=y, 极坐标val1=r, val2=cita
    """
    def __init__(self, val1, val2, typeid):
        self._val1 = val1
        self._val2 = val2
        self._typeid = typeid
        if self._typeid not in [1, 2]:
            raise ValueError('没有这个typeid')

    def __str__(self):
        pre1 = 'x:' if self._typeid == 1 else 'r:'
        pre2 = 'y:' if self._typeid == 1 else 'cita:'
        return "{Point\n" +\
            "%s%.4f\t" % (pre1, self._val1) +\
                "%s%.4f}\n" % (pre2, self._val2)

    def __repr__(self):
        pre1 = 'x:' if self._typeid == 1 else 'r:'
        pre2 = 'y:' if self._typeid == 1 else 'cita:'
        return "(%s%.4f, %s%.4f)" % (pre1, self._val1, pre2, self._val2)


    @property
    def coord(self):
        '''点的坐标'''
        return (self._val1, self._val2)

    @property
    def typeid(self):
        '''点的种类'''
        return self._typeid


def shift_point(opt: Point, xsh, ysh):
    '''平移一个点'''
    if opt.typeid != 1:
        raise NotImplementedError('只能用直角坐标的')
    ocoor = opt.coord
    return Point(ocoor[0]+xsh, ocoor[1]+ysh, opt.typeid)


def middle_point(pt1: Point, pt2: Point, sc1=0.5, sc2=0.5):
    '''两个点的中点'''
    if pt1.typeid != 1 or pt2.typeid != 1:
        raise NotImplementedError('只能用直角坐标的')
    scs = sc1 + sc2
    cor1 = pt1.coord
    cor2 = pt2.coord
    xpt = cor1[0] * sc1 + cor2[0] * sc2
    xpt /= scs
    ypt = cor1[1] * sc1 + cor2[1] * sc2
    ypt /= scs
    return Point(xpt, ypt, 1)
