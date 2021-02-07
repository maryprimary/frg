fRG算法演示
===

使用的标记习惯和 *Introduction to the Functional Renormalization Group* 
一致。  
代码中的公式没有其他说明也是出自这本书。  

运行代码需要`python3(>3.5)`，`numpy`，`scipy`  
首先要生成描述第一布里渊区的文件
```bash
python3 scripts/square_brillouin.py -p 16
```
在此基础上，进行计算
```bash
python3 scripts/square_solution_multi.py -p 16
```

## 设置运行时的参数


> -p --patches  
程序的fRG采用npatch算法进行实现，在运行代码时必须要指定patch的数量，
而且两次的数量大小必须一致。

> \[-m --mesh\]  
在数值计算的过程中，布里渊区会被切分成很多个小三角形，mesh指定这个小三角形的数量，对于
正方形格子，数量是4\*mesh\*mesh

> \[-d --disp\]  
色散关系，自由哈密量的色散关系，不同的跃迁和化学势会改变色散关系，square是普通的最近邻跃迁。

> \[--prefix\]  
用来保存和读取布里渊区描述文件的路径
