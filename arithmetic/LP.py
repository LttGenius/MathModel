#线性规划算法库
from scipy.optimize import linprog
import numpy as np

#基础线性规划为
# linprog(c,Au,Bu,Aeq,Beq,b)

###########整数规划##############
#分支定界法
def brandBoundLP(c,Au,Bu,Aeq,Beq,b):
    res=linprog(c,Au,Bu,Aeq,Beq,b)

if __name__=='main':
    pass

