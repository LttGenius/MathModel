#线性规划算法库
from re import I
from scipy.optimize import linprog
import numpy as np
import copy
#基础线性规划为
# linprog(c,Au,Bu,Aeq,Beq,b)

###########整数规划##############
#分支定界法
def brandBoundLP(c,Au=None,Bu=None,Aeq=None,Beq=None,b=None,limitT=1e-7,putValue=False):#返回[最优值，最优解]
    res=linprog(c,Au,Bu,Aeq,Beq,b)
    optimalValue=res
    if not res.success:return 
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
        return False

if __name__=='__main__':
    c=[-40,-90]
    au=[[9,7],[7,20]]
    bu=[56,70]
    x0=[0,float('inf')]
    x1=[0,float('inf')]
    res=brandBoundLP(c,au,bu,b=(x0,x1),putValue=True)
    print(res)

