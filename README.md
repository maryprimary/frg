fRG算法演示
===

使用的标记习惯和 *Introduction to the Functional Renormalization Group* 
一致。  
代码中的公式没有其他说明也是出自这本书。  

运行代码需要```python3(>3.5)```，```numpy```，```scipy```  
首先要生成描述第一布里渊区的文件
```bash
python3 scripts/square_brillouin.py -p 16
```
在此基础上，进行计算
```bash
python3 scripts/square_solution.py -p 16
```
