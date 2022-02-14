"""
模糊数学模型:

"""
import numpy as np

def fuzzyMath(A:tuple,R:np.array=None,w=None,matrixMake='rd',alpha=None,fuzzyFun=4):
    """
    A:(指标信息,方案矩阵array)
    """
    if R == None:
        #构建R
        if matrixMake == 'rd':#相对偏差法
            u=np.array([eval(i)(j) for i,j in zip(A[0],A[1].T)])
            R=abs(A[1]-u)/(np.max(A[1],axis=0)-np.min(A[1],axis=0))
        elif matrixMake == 'RSA':#相对优属性
            cTable={'max':lambda x:x/np.max(x),'min':lambda x:np.min(x)/x,'fixed':lambda x:np.min(abs(x-alpha))/abs(x-alpha)}
            assert alpha,'alpha值错误'
            R=np.array([cTable[i](j) for i,j in zip(A[0],A[1].T)]).T
    #变异系数
    if not w:
        w=np.var(R,axis=0)/np.mean(R,axis=0)
    funTable=(lambda x,w:np.max(np.minimum(w,x),axis=0),lambda x,w:np.max(w/np.sum(w)*x,axis=0),\
        lambda x,w:np.sum(np.minimum(w,x),axis=0),lambda x,w:np.sum(np.dot(w,x),axis=0),lambda x,w:w/np.sum(w)*x)
    return funTable[fuzzyFun](R,w)

if __name__ == '__main__':
    A=(('min','max','fixed'),np.array([[1,2,3],[4,5,2]]))
    res=fuzzyMath(A,matrixMake='RSA',alpha=3.5,fuzzyFun=3)
    print(res)
    