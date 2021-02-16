"""处理profile文件"""


import sys
import pstats


def main(fname):
    '''入口'''
    out = ''.join(fname.split('.')[:-1])+'.prof'
    stat = pstats.Stats(fname, stream=open(out, 'w'))
    stat.sort_stats(2)
    stat.print_stats()



if __name__ == '__main__':
    main(sys.argv[1])
