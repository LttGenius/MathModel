"""
插值:
    1、拉格朗日插值法
    2、一维插值
    3、二维插值
    4、Rbf插值
"""
import numpy as np
import matplotlib
import copy
def lagrange(x:np.array,y:np.array,inPoint=None,matplot=False):
    n=len(x)
    temp=np.hstack((x,[inPoint]*n))
    temp=temp.reshape(-1,1)-x
    temp=np.delete(temp,list([[i*n+i,(i+n)*n+i] for i in range(n)])).reshape(2*n,n-1)
    temp=np.prod(temp,axis=1)
    lagrangeFun=lambda i:np.dot(temp[n:]/temp[0:n],y)
    return lagrangeFun(inPoint)

if __name__ == "__main__":
    x=np.array([1,2,3])
    y=np.array([1,4,9])
    res=lagrange(x,y,1.5)
    print(res)
