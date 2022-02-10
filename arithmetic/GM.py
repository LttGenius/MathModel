"""
灰色预测算法:
    grayPrediction
"""
import numpy as np
def grayPrediction(initData:np.array,alpha=0.5,checkWay='re'):
    """
    灰色预测模型算法GM(1,1):
        
    """
    #参数检验
    inPara=('re','ru','Both')
    assert checkWay in inPara,'参数checkWay错误'
    # 1 数据检验
    labd=initData[0:-1]/initData[1:]
    n=len(initData)
    assert np.all(labd>np.exp(-2/(n+1))) and np.all(labd<np.exp(2/(n+1))),'级比不在可覆盖区间'
    # 2 建立模型
        #加权
    B=[]
    temp=np.cumsum(initData)
    j=temp[0]
    temp=temp[1:]
    for i in temp:
        B.append(-1*(alpha*i+(1-alpha)*j))
        j=i
        #构建矩阵
    Y=initData[1:]
    B=np.array(B).reshape(-1,1)
    B=np.hstack((B,np.ones((B.shape[0],1))))
        #解出u
    U=np.dot(B.T,B)
    U=np.dot(np.linalg.inv(U),B.T)
    U=np.dot(U,Y)
        #构建微分方程求解
    """
    temp=lambda k:(initData[0]-U[1]/U[0])*np.exp(-U[0]*k)+U[1]/U[0] if k>0 else initData[0]
    returnFun=lambda k:temp(k)-(temp(k-1) if k>0 else 0)
    """
    def returnFun(k):
        if k<=1:
            if k<1:
                return initData[0]
            else:
                return (initData[0]-U[1]/U[0])*np.exp(-U[0]*k)+U[1]/U[0]-initData[0]
        return (initData[0]-U[1]/U[0])*np.exp(-U[0]*k)+U[1]/U[0]-((initData[0]-U[1]/U[0])*np.exp(-U[0]*(k-1))+U[1]/U[0])
    # 3 检验预测值
    checkList={'re':[],'ru':None}
        #残差检验
    if checkWay == 're' or checkWay == 'Both':
        j=0
        for i in initData:
            tt=returnFun(j)
            checkList['re'].append((i-returnFun(j))/i)
            j+=1
        #级比检验
    if checkWay == 'ru' or checkWay == 'Both':
        checkList['ru']=list(1-((1-0.5*alpha)/(1+0.5*alpha))*labd)
    return (checkList,returnFun)

if __name__ == '__main__':
    Y=np.array([71.1,72.4,72.4,72.1,71.4,72,71.6])
    res=grayPrediction(Y)
    print(res[1])