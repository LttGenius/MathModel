"""
线性规划算法库
Arithmetic:
    Basic:
        松弛问题线性规划 linprog
        单纯形法 simplexMethod
    Integer LP:
        分支定界法 brandBoundLP
        割平面法 cuttingPlaneApproach
        匈牙利法 
"""
from scipy.optimize import linprog
import sympy
import numpy as np
import copy
"""
带精度比较函数
    需要保证传入为一维
"""
def Com_pre(comObject,minOrMax=False,returnValue='value',limitPrecision=1e-5)->'str':
    """
    内置比较函数
    comObject:一维数组,否则最后结果可能会出错
    minOrMax=False:最小查找
             True:最大查找
    returnValue='value':输出结果为查找值
                'index':输出结果为查找位置
    limitPrecision:比较阈值
    """

    temp=np.sort(comObject)
    if minOrMax:
        temp=temp[::-1]
    stand=temp[0]
    flag=0
    for i in temp:
        if abs(stand-i)<=limitPrecision:flag+=1
        else:break
    if returnValue=='value':
        return temp[0:flag]
    elif returnValue=='index':
        if minOrMax:
            return np.where(comObject>=temp[flag-1])
        else:
            return np.where(comObject<=temp[flag-1])
#松弛线性规划
# linprog(c,Au,Bu,Aeq,Beq,b)

#单纯形法
"""
Model:
    Min(Max) cX
    S.T.
        Au<=Bu
    需要保证X>=0
    Max和Min由参数putValue确认:当c对应是Max时,putValue应为True,否则为false
    putForm:True时输出表格,否则只输出x和最优值
1）当所有非基变量的检验数都小于零，则原问题有唯一最优解；
2）当所有非基变量的检验数都小于等于零，注意有等于零的检验数，则有无穷多个最优解；
3）当任意一个大于零的非基变量的检验数，其对应的ajk（求最小比值的分母）都小于等于零时，则原问题有无界解；
4）添加人工变量后的问题，当所有非基变量的检验数都小于等于零，而基变量中有人工变量时，则原问题无可行解。
PS:情况1和情况2并未区分，如需分别请将参数putForm设置为True，查看表格分别
"""
def simplexMethod(c,Au=None,Bu=None,putValue=False,putForm=False)->'str':
    """
    线性规划算法之单纯形法
    Model:
        Min(Max) C @ X 
        S.T.
            Au @ X <= Bu
            X >= 0
    C:一维数组，线性规划中需要求值的函数系数
    Au:二维数组，不等式的系数
    Bu:一维数组，不等式的值
    putValue=True:模型是求最大值，会对C进行自动求负
             False:模型求最小值
    putForm=True:输出单纯形最优表格矩阵
            False:不输出，默认不输出
    Return:如果putForm为False:
            [x,fun]
           如果putForm为True:
            输出矩阵
    """
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
            return [[float('inf') for _ in range(len(c))],[float('inf')]]
        position.append(int(np.argwhere(B==(min(B)))))#查找到主元素位置
        #接下里进行初等行变换
        #将主元素这一行除以主元素
        B=simplexMatrix[position[1]]=simplexMatrix[position[1]]/simplexMatrix[position[1],position[0]]
        #其余行减去这一行
        temp=np.array(simplexMatrix[position[1]])
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
def brandBoundLP(C,Au=None,Bu=None,Aeq=None,Beq=None,b=None,limitT=1e-7,putValue=False)->'str':
    """
    整数线性规划算法之分支定界法
    Model:
        Min(Max) C @ X 
        S.T.
            Au @ X <= Bu
            Aeq @ X=Beq
            lb <= X <= ub
    C:一维数组，线性规划中需要求值的函数系数
    Au:二维数组，不等式的系数
    Bu:一维数组，不等式的值
    Aeq:二维数组，等式的系数
    Beq:一维数组，等式的值
    b:未知数的上下限 b=(lb,ub)
    putValue=True:模型是求最大值，会对C进行自动求负
             False:模型求最小值
    limitT:误差阈值,默认为1e-5
    Return:(形式如linprog)
    x:在满足约束的情况下将目标函数最小化的决策变量的值
        数据类型:一维数组
    fun:目标函数的最佳值（c T x）
        数据类型:浮点数
    slack:不等式约束的松弛值（名义上为正值）bub-Aub x
        数据类型:一维数组
    con:等式约束的残差（名义上为零）beq-Aeq x
        数据类型:一维数组
    success:当算法成功找到最佳解决方案时为 True
        数据类型:布尔值
    status:表示算法退出状态的整数
        数据类型:整数
        0 : 优化成功终止
        1 : 达到了迭代限制
        2 : 问题似乎不可行
        3 : 问题似乎是不收敛
        4 : 遇到数值困难
    nit:在所有阶段中执行的迭代总数
        数据类型:整数
    message:算法退出状态的字符串描述符
        数据类型:字符串
    """

    if putValue:
        c=np.array([-1*i for i in C])
    else:
        c=np.array(C)
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
    newb0=np.array(list(b))
    newb1=np.array(newb0)
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
                    optimalValue=res
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
                newTemp=np.array(temp)
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
Model:
    Min(Max) CX 
    S.T.
        AuX<=Bu
        AeqX=Beq
        lb<=X<=ub(b=(lb,ub))
putValue=True:模型是求最大值，会对C进行自动求负
         False:模型求最小值
limitT:误差阈值
PS:半成品，依然有bug需要完善，部分测试不通过
    还需完成只有等式约束情况
"""
def cuttingPlaneApproach(c,Au=None,Bu=None,Aeq=None,Beq=None,b=None,limitT=1e-5,putValue=False)->'str':
    """
    整数线性规划算法之割平面法
    Model:
        Min(Max) C @ X 
        S.T.
            Au @ X <= Bu
            Aeq @ X=Beq
            lb <= X <= ub
    C:一维数组，线性规划中需要求值的函数系数
    Au:二维数组，不等式的系数
    Bu:一维数组，不等式的值
    Aeq:二维数组，等式的系数
    Beq:一维数组，等式的值
    b:未知数的上下限 b=(lb,ub)
    putValue=True:模型是求最大值，会对C进行自动求负
             False:模型求最小值
    limitT:误差阈值,默认为1e-5
    Return:(形式如linprog)
    x:在满足约束的情况下将目标函数最小化的决策变量的值
        数据类型:一维数组
    fun:目标函数的最佳值（c T x）
        数据类型:浮点数
    slack:不等式约束的松弛值（名义上为正值）bub-Aub x
        数据类型:一维数组
    con:等式约束的残差（名义上为零）beq-Aeq x
        数据类型:一维数组
    success:当算法成功找到最佳解决方案时为 True
        数据类型:布尔值
    status:表示算法退出状态的整数
        数据类型:整数
        0 : 优化成功终止
        1 : 达到了迭代限制
        2 : 问题似乎不可行
        3 : 问题似乎是不收敛
        4 : 遇到数值困难
    nit:在所有阶段中执行的迭代总数
        数据类型:整数
    message:算法退出状态的字符串描述符
        数据类型:字符串
    """
    ###拷贝###############
    if putValue:
        C=[-1*i for i in c]
    else:
        C=np.array(c)
    xposition=len(c)#记录输出的解个数
    #复制拷贝
    au=np.array(Au)
    bu=np.array(Bu)
    newb=np.array(b)
    ######################
    res=linprog(C,au,bu,Aeq,Beq,newb,method='simplex')#第一次求解
    if not res.success:return res#无解
    """
    接下来是割平面求最优解
    """
    #构建初始矩阵
    """
    PS:编写割平面时还未完成单纯形算法，采用linprog求解，如果有解并且解不为整数，构建矩阵模拟单纯形法矩阵
    """
    simplexTable=np.hstack((au,np.eye(len(au))))
    simplexTable=np.array(sympy.Matrix(simplexTable).rref()[0].tolist())
    simplexTable=np.hstack((simplexTable,res.x.reshape(res.x.shape[0],1)))
    """
    矩阵形式(初等变化之后):
    array:[[1,0,0,...,a11,a12,a13,b1]
           [0,1,0,...,a21,a22,a23,b2]
           [0,0,1,...,a31,a32,a33,b3]
           [0,0,0,...,a41,a42,a43,b4]
           ...
           [0,0,...,1,an1,an2,an3,bn]]
    """
    while not all(((x-np.floor(x))<limitT or (np.ceil(x)-x)<limitT) for x in res.x[0:xposition]):
        #寻找最小小数行作为添加条件行
        #最简矩阵 保证为二维
        simplexTable=np.array(sympy.Matrix(simplexTable).rref()[0].tolist())
        copySimplexTable=copy.deepcopy(simplexTable)
        simplexTable%=1
        temp=simplexTable[:,-1]
        position=Com_pre(temp,minOrMax=True,returnValue='index')#找到矩阵b那一列最大分数位置
        #注意temp可能一维可能二维
        temp=copy.deepcopy(simplexTable[position])#保证了temp始终为二维数组
        temp=np.sum(temp,axis=1)#为一维数组
        temp=Com_pre(temp,returnValue='index')[0][0]#如果重复取所有中第一个 #标记2
        position=position[0][temp]#确定合适的等式进行分割
        #更新约束
        updateMatrix_newLine=-1*simplexTable[position,:]#一维
        #依次更新
        C=np.append(C,[0 for _ in range(len(au))])
        newb=np.vstack((newb,[[0,float('inf')] for _ in range(len(au))]))
        au=-1*simplexTable[position:position+1,0:-1]#只有一个不等式 au此时为2维数组
        bu=-1*simplexTable[position:position+1,-1]
        aeq=copy.deepcopy(copySimplexTable[:,0:-1])
        beq=copy.deepcopy(copySimplexTable[:,-1])
        #进行线性规划
        res=linprog(C,au,bu,aeq,beq,newb,method='simplex')
        #更新矩阵
        simplexTable=np.vstack((copySimplexTable,updateMatrix_newLine))#添加新的不等式
        temp=[[0] for _ in range(len(simplexTable))]
        temp[-1]=[1]
        simplexTable=np.hstack((simplexTable,temp))#将不等式转化为等式，此时矩阵为线性规划之后更新的模拟单纯形法矩阵
        if not res.success:return res
    if putValue:
        res.fun*=-1
    return res

if __name__=='__main__':
    C = [-5,-8] 
    A = [[1,1],[5,9]]
    bu = [6,45]
    X0_bounds = [0,float('inf')]
    X1_bounds = [0,float('inf')]
    #test=np.array([1.33333,1.33333,2,3,4])
    #res=Com_pre(test)
    res=cuttingPlaneApproach(C,A,bu,b=(X0_bounds,X1_bounds))
    print(res)
    


