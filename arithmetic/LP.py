#线性规划算法库
from scipy.optimize import linprog
import sympy
import numpy as np
import copy
#基础线性规划为
# linprog(c,Au,Bu,Aeq,Beq,b)
###########整数规划##############

#分支定界法#####################################################################
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
#####################################################################################


#割平面法#############################################################################
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
    ######################
    res=linprog(C,au,bu,Aeq,Beq,b)
    newb=copy.deepcopy(b)
    if not res.success:return res#无解
    """
    接下来是割平面求最优解
    """
    if au:
        Matrix=np.ones((au.shape[0],au.shape[0]))
        Matrix=np.hstack((au,Matrix))
        Matrix=np.hstack((Matrix,Beq))
        """
        Matrix:
            [[a11,a12,a13,...,one1,0,0,...,b1]
             [a21,a22,a23,...,0,one2,0,...,b2]
             ...
             ...
             [an1,an2,an3,...,0,0,...,onen,bn]
                                              ]
        """
        position=au.shape[1]+1
    else:#无整数解
        pass
    while not all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x):
        """
        分割平面
        """
        Matrix=np.array(sympy.Matrix(Matrix).rref[0])
        """
        Matrix:
            [[A1,0,0,...,B11,B12,B12,...,Nb1]
             [0,A2,0,...,B21,B22,B23,...,Nb2]
             ...
             ...
             [0,0,...,An,BN1,BN2,BN3,...,Nbn]
                                              ]
        """
        
        res=linprog(C,au,bu,Aeq,Beq,newb)
        if not res.success:return res#规划失败
    return res

#######################################################################################



if __name__=='__main__':
    c=[-5,-8]
    au=[[1,1],[5,9],[1,0],[0,1]]
    bu=[6,45,float('inf'),float('inf')]
    x0=[0,float('inf')]
    x1=[0,float('inf')]
    #res=brandBoundLP(c,au,bu,b=(x0,x1),putValue=True)
    res=linprog(c,au,bu)
    print(res)

