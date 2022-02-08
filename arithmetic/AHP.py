"""
层次分析法
"""
import numpy as np
import copy
#####归一化#####
def uniformization(m,way='column',):
    if way=='column':
        return m/np.sum(m,axis=0)
    elif way=='row':
        return (m.T/np.sum(m,axis=1)).T
    elif way=='all':
        return m/np.sum(m)
    else:
        raise

#####层次分析法#####
"""
层次分析法(AHP):
    1、构建判断矩阵
    2、构建方案对判断因素的矩阵
    3、对判断矩阵进行特征向量提取和一致性检验
    4、对每个方案矩阵进行特征向量提取和归一化检验
    5、对总排序进行归一化检验
    6、返回分析结果
"""
def ahp(a:np.ndarray,B:dict,returnMatrix=False,allReal=False,returnCICR=False,returnRank=False):
    """
    层次分析法(AHP):
        1、构建判断矩阵
        2、构建方案对判断因素的矩阵
        3、对判断矩阵进行特征向量提取和一致性检验
        4、对每个方案矩阵进行特征向量提取和归一化检验
        5、对总排序进行归一化检验
        6、返回分析结果
    
    输入参数
        a:判断矩阵,二维数组,请确保是方阵
        B:方案矩阵,键确保为字符串或者数字,不同元素表示不同方案对判断因素的矩阵
    返回结果
        默认返回最大值结果和最大值位置
        returnMatrix=False:不返回结果矩阵(默认)
                     True:返回结果矩阵
        allReal=False:不将结果全部转化为实数(默认)
                True:将结果全部转化为实数
        returnCICR=False:不返回所有矩阵一致性检验表(默认)
                   True:返回一致性检验表
                   返回形式为字典{'CI':[],'CR':[]}
        returnRank=False:不返回排序结果(默认)
                   True:返回排序结果
                   返回形式为列表,返回为列表,降序排序的下标
    """
    RI=[0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54,1.56,1.58,1.59,1.60,1.61,1.615,1.62,1.63]#RI表
    n=a.shape[0]
    eigenvalues,featurevector=np.linalg.eig(a)#获取特征值特征向量
    eigenvalue=max(eigenvalues)#最大特征值
    CICR={'CI':[],'CR':[]}
    temp=(eigenvalue-n)/(n-1)
    assert temp/RI[n]<0.1,'判断矩阵一致性检验失败'
    CICR['CI'].append(temp)
    temp=temp/RI[n]#检验CR
    CICR['CR'].append(temp)
    resultMatrix=np.append(featurevector[:,np.argmax(eigenvalues)],temp)#构建判断矩阵权重向量
    #开始对方案进行特征值特征向量提取并检验
    tempMatrix=[]
    for i in B.items():
        temp=np.array(i[1])
        eigenvalues,featurevector=np.linalg.eig(temp)
        eigenvalue=max(eigenvalues)
        n=temp.shape[0]
        temp=(eigenvalue-n)/(n-1)
        assert temp/RI[n]<0.1,'方案矩阵'+str(i[0])+'一致性检验失败'
        CICR['CI'].append(temp)
        tempMatrix.append(np.append(featurevector[:,np.argmax(eigenvalues)],np.array([eigenvalue,temp])))
        temp=temp/RI[n]
        CICR['CR'].append(temp)
    tempMatrix=np.array(tempMatrix).T#转置为列
    #归一化处理
    resultMatrix[0:-1]=uniformization(resultMatrix[0:-1])
    tempMatrix[0:-2,:]=uniformization(tempMatrix[0:-2,:])
    #总排序一致性检验
    assert np.dot(tempMatrix[-1,:],resultMatrix[0:-1])/np.sum(resultMatrix[0:-1]*RI[n])<0.1,'总排序一致性检验不通过'
    temp=np.array([np.dot(i,resultMatrix[0:-1]) for i in tempMatrix[0:-2,:]])#权重相乘计算
    temp=np.hstack((temp,np.array([np.max(temp),np.argmax(temp)])))#合并权重结果和最大值以及最大值位置
    #开始合并结果矩阵
    tempMatrix=np.hstack((tempMatrix,temp[:,np.newaxis]))
    resultMatrix=np.vstack((resultMatrix,tempMatrix))
    elseReturn=[]
    #返回判断
    if returnCICR:
        elseReturn.append(CICR)
    if returnRank:
        elseReturn.append(np.argsort(-resultMatrix[1:-2,-1]))
    if allReal:
        resultMatrix=resultMatrix.astype(float)
    if returnMatrix:
        if returnCICR or returnRank:
            return (resultMatrix,elseReturn)
    if returnCICR or returnRank:
        return ((resultMatrix[1+int(resultMatrix[-1,-1]),:],resultMatrix[-1,-1]),elseReturn)
    else:
        return (resultMatrix[1+int(resultMatrix[-1,-1]),:],resultMatrix[-1,-1])


if __name__ == '__main__':
    A=np.array([[1,1/2,4,3,3],[1,1,7,5,5],[1,1,1,1/2,1/3],[1,1,1,1,1],[1,1,1,1,1]])
    A+=1/A.T-np.ones(A.shape)
    B={}
    temp=np.array([[1,2,5],[1,1,2],[1,1,1]])
    temp=temp+1/temp.T-np.ones(temp.shape)
    B[1]=temp
    temp=np.array([[1,1/3,1/8],[1,1,1/3],[1,1,1]])
    temp=temp+1/temp.T-np.ones(temp.shape)
    B[2]=temp
    temp=np.array([[1,1,3],[1,1,3],[1,1,1]])
    temp=temp+1/temp.T-np.ones(temp.shape)
    B[3]=temp
    temp=np.array([[1,3,4],[1,1,1],[1,1,1]])
    temp=temp+1/temp.T-np.ones(temp.shape)
    B[4]=temp
    temp=np.array([[1,1,1/4],[1,1,1/4],[1,1,1]])
    temp=temp+1/temp.T-np.ones(temp.shape)
    B[5]=temp
    res=ahp(A,B,allReal=True,returnCICR=True,returnMatrix=True,returnRank=True)
    print(res)