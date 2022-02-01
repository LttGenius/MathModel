"""
线性规划算法库
Arithmetic:
    Basic:
        松弛问题线性规划 linprog
        单纯形法 simplexMethod
    Integer LP:
        分支定界法 brandBoundLP
        割平面法
"""
from scipy.optimize import linprog
import sympy
import numpy as np
import copy
#松弛线性规划
# linprog(c,Au,Bu,Aeq,Beq,b)

#单纯性法
"""
Model:
    Min(Max) cX
    S.T.
        Au<=Bu
    需要保证X>=0
    Max和Min由参数putValue确认
"""
def simplexMethod(c,Au=None,Bu=None,putValue=False,putForm=False):
    #构建矩阵
    """
    矩阵形状:
        array[[b0,a10,a11,a12,...,a1n]
              [b1,a20,a21,a22,...,a2n]
              ...
              [bn,an0,an1,an2,...,ann]
              [Z,C0,C1,C2,C3,c4,..,Cn]]
    b为由Bu组合成的一列
    a是由不等式(加上松弛变量变为等式)组合出的矩阵
    """
    if not Au:#无约束错误
        return False
    if putValue:
        c=[-1*i for i in c]
    length=len(Au)
    temp=np.eye(length)
    simplexMatrix=np.hstack((np.array(Au),temp))
    temp=np.array(Bu)
    temp=temp.reshape(len(temp),1)
    simplexMatrix=np.hstack((temp,simplexMatrix))
    temp=np.hstack((np.array([0]),np.hstack((np.array(c),np.array([0 for _ in range(length)])))))
    simplexMatrix=np.vstack((simplexMatrix,temp))
    #创立各种系数检验列表
    """
    cherkoutMatrix_X:array[Aj,...,An] 基变量，初始挑选为单位矩阵
    checkoutMatrix_6:array[g0,..,gn] 非基变量检验数
    """
    checkoutMatrix_X=np.array([i for i in range(len(c)+1,len(c)+1+length)])
    position6=simplexMatrix.shape[0]-1
    checkoutMatrix_6=simplexMatrix[position6][1:]
    #开始查找最优解:
    while np.any(checkoutMatrix_6>0):
        """
        position:[p0,p1] p1:行 p0:列
        """
        position=[]
        position.append(int(np.argwhere(checkoutMatrix_6==(max(checkoutMatrix_6)))+1))
        #判断有无解
        if np.all(checkoutMatrix_6<=0) and np.any(checkoutMatrix_X>=(length+1)):return False
        #有解执行
        B=simplexMatrix[0:position6,0]/simplexMatrix[0:position6,position[0]]
        B[B<=0]=float('inf')
        if np.all(B==float('inf')):#有无界解
            return "无界"
        position.append(int(np.argwhere(B==(min(B)))))#查找到主元素位置
        #接下里进行初等行变换
        #将主元素这一行除以主元素
        B=simplexMatrix[position[1]]=simplexMatrix[position[1]]/simplexMatrix[position[1],position[0]]
        #其余行减去这一行
        temp=copy.deepcopy(simplexMatrix[position[1]])
        for i in range(position6+1):
            simplexMatrix[i]-=simplexMatrix[i,position[0]]*B
        simplexMatrix[position[1]]=temp
        #更新checkoutMatrix_X
        checkoutMatrix_X[position[1]]=position[0]
        #更新checkoutMatrix_6
        checkoutMatrix_6=simplexMatrix[position6][1:]
    if putValue:
        simplexMatrix[length,0]=-1*simplexMatrix[length,0]
    if putForm:
        return simplexMatrix
    else:
        return [simplexMatrix[0:length,0],simplexMatrix[length,0]]

###########整数规划##############
###########分支定界法############
"""
Model:
    Min(Max) CX 
    S.T.
        AuX<=Bu
        AeqX=Beq
        lb<=X<=ub(b=(lb,ub))
putValue=True:模型是求最大值，会对C进行自动求负
         False:模型求最小值
limitT:误差阈值
"""
def brandBoundLP(C,Au=None,Bu=None,Aeq=None,Beq=None,b=None,limitT=1e-7,putValue=False):#返回[最优值，最优解]
    if putValue:
        c=copy.deepcopy([-1*i for i in C])
    else:
        c=copy.deepcopy(C)
    res=linprog(c,Au,Bu,Aeq,Beq,b)
    optimalValue=res
    if not res.success:return res
    if all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):#判断是否全为整数
        return optimalValue
    limitv=[optimalValue.fun,float('inf')]#值上下界
    #找到不是整数的解
    i=0
    for j in (((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):
        if j:
            i+=1
        else:
            break
    limitx=(np.floor(res.x[i]),np.ceil(res.x[i]))#分支
    ####
    #值copy
    newb0=copy.deepcopy(list(b))
    newb1=copy.deepcopy(newb0)
    ############
    stack=[]
    stackTop=0
    if  newb0[i][1]>limitx[0]:
        newb0[i][1]=limitx[0]
        stack.append(newb0)
        stackTop+=1
    if newb1[i][0]<limitx[1]:
        newb1[i][0]=limitx[1]
        stack.append(newb1)
        stackTop+=1
    #分支定界开始
    while stackTop:
        temp=stack.pop()
        stackTop-=1
        res=linprog(c,Au,Bu,Aeq,Beq,temp)
        if res.success:#如果有解
            if all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):#判断是否全为整数
                if res.fun<limitv[1]:#定界
                    limitv[1]=res.fun
                    optimalValue=copy.deepcopy(res)
            else:#有小数，开始分支
                if res.fun>limitv[1]:#减支
                    continue
                i=0
                for j in (((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):
                    if j:
                        i+=1
                    else:
                        break
                limitx=(np.floor(res.x[i]),np.ceil(res.x[i]))#分支
                newTemp=copy.deepcopy(temp)
                if newTemp[i][0]<limitx[1]:
                    newTemp[i][0]=limitx[1]
                    stack.append(newTemp)
                    stackTop+=1
                if temp[i][1]>limitx[0]:
                    temp[i][1]=limitx[0]
                    stack.append(temp)
                    stackTop+=1
    if all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in optimalValue.x):#判断是否全为整数
        if putValue:
            optimalValue.fun=-optimalValue.fun
        return optimalValue
    else:
        res.success='False'
        return res

###########割平面法###############
"""

"""
def cuttingPlaneApproach(c,Au=None,Bu=None,Aeq=None,Beq=None,b=None,limitT=1e-7,putValue=False):
    ###拷贝###############
    if putValue:
        C=[-1*i for i in c]
    else:
        C=copy.deepcopy(c)
    au=copy.deepcopy(Au)
    bu=copy.deepcopy(Bu)
    newb=copy.deepcopy(b)
    ######################
    res=linprog(C,au,bu,Aeq,Beq,newb)
    if not res.success:return res#无解
    """
    接下来是割平面求最优解
    """
    au=np.array(au)
    bu=np.array(bu)
    while not all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):
        """
        分割平面
        """
        if au.any():
            Matrix=np.eye(au.shape[0])
            Matrix=np.hstack((au,Matrix))
            Matrix=np.hstack((Matrix,bu.reshape(au.shape[0],1)))
            """
            Matrix:
                [[a11,a12,a13,...,one1,0,0,...,b1]
                [a21,a22,a23,...,0,one2,0,...,b2]
                ...
                ...
                [an1,an2,an3,...,0,0,...,onen,bn]
                                                ]
            """
            position0=au.shape[1]
            position1=au.shape[0]
            C=np.hstack((C,np.zeros(position1)))
            newb=np.hstack
        else:#无整数解
            return res
        Matrix=np.array((sympy.Matrix(Matrix).rref())[0])
        """
        Matrix:
            [[1,0,0,...,B11,B12,B12,...,Nb1]
             [0,1,0,...,B21,B22,B23,...,Nb2]
             ...
             ...
             [0,0,...,1,BN1,BN2,BN3,...,Nbn]
                                              ]
        """
        tempAu=[]
        tempBu=[]
        for i in Matrix:
            templist=[]
            flag=0
            for j in i[position0:]:
                if flag == position1:
                    tempBu.append(-1*(j%1))
                else:
                    templist.append(-1*(j%1))
                flag+=1
            tempAu.append(templist)
        a=np.zeros((position1,position1))
        au=np.hstack((au,a))
        au=np.vstack((au,np.hstack((np.zeros((position0,position0)),np.array(tempAu)))))
        bu=np.hstack((bu,np.array(tempBu)))
        res=linprog(C,au,bu,Aeq,Beq,newb)
        if not res.success:return res#规划失败
    return res

if __name__=='__main__':
    C = [-3,-4] 
    A = [[2,1],[1,3]]
    b = [40,30]
    X0_bounds = [0,float('inf')]
    X1_bounds = [0,float('inf')]
    res=simplexMethod(C,A,b,putValue=True,putForm=True)
    print(res)
    


