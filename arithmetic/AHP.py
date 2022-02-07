"""
层次分析法
"""
import numpy as np

"""
层次分析法(AHP):
"""
def ahp(A:dict,B:dict,returnMatrix=False,allReal=False):
    RI=[0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54,1.56,1.58,1.59,1.60,1.61,1.615,1.62,1.63]
    a=np.array(list(A.values()))
    n=a.shape[0]
    eigenvalues,featurevector=np.linalg.eig(a)
    eigenvalue=max(eigenvalues)
    CICR={'CI':[],'CR':[]}
    temp=(eigenvalue-n)/n-1
    assert temp/RI[n]<0.1,'判断矩阵一致性检验失败'
    CICR['CI'].append(temp)
    temp=temp/RI[n]
    CICR['CR'].append(temp)
    resultMatrix=np.append(featurevector[:,np.argmax(eigenvalues)],temp)
    tempMatrix=[]
    for i in B.items():
        temp=np.array(i[1])
        eigenvalues,featurevector=np.linalg.eig(a)
        eigenvalue=max(eigenvalues)
        temp=(eigenvalue-n)/n-1
        assert temp/RI[n]<0.1,'方案矩阵'+str(i[0])+'一致性检验失败'
        CICR['CI'].append(temp)
        tempMatrix.append(np.append(featurevector[:,np.argmax(eigenvalues)],np.array([eigenvalue,temp])))
        temp=temp/RI[n]
        CICR['CR'].append(temp)
    tempMatrix=np.array(tempMatrix).T
    assert np.dot(tempMatrix[-1,:],resultMatrix[0:-1])/np.sum(resultMatrix[0:-1]*RI[n])<0.1,'总排序一致性检验不通过'
    temp=np.array([np.dot(i,j) for i,j in zip(tempMatrix[:,0:-2],resultMatrix[0:-1])])
    temp=np.vstack((temp,np.array([np.max(temp),np.argmax(temp)]).reshape(-1,1)))
    tempMatrix=np.hstack((tempMatrix,temp))
    resultMatrix=np.vstack((resultMatrix,tempMatrix))
    if allReal:
        resultMatrix=resultMatrix.astype(float)
    if returnMatrix:
        return resultMatrix
    return (resultMatrix[1+int(resultMatrix[-1,-1]),:],resultMatrix[-1,-1])


if __name__ == '__main__':
    A={1:[1,3,5],2:[0.33,1,3],3:[0.2,0.33,1]}
    B={1:[[1,2],[0.5,1]],2:[[1,2],[0.5,1]],3:[[1,2],[0.5,1]]}
    res=ahp(A,B,allReal=True)
    print(res)