"""
非线性规划算法库
Arithmetic:
"""
from scipy.optimize import minimize
import numpy as np
"""
非线性规划函数minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
其中:
fun:目标函数，返回单值，
x0:初始迭代值，
args:要输入到目标函数中的参数
method:求解的算法，目前可选的有
        'Nelder-Mead'
        'Powell' 
        'CG' 
        'BFGS' 
        'Newton-CG' 
        'L-BFGS-B'
        'TNC'
        'COBYLA' 
        'SLSQP' 
        'dogleg' 
        'trust-ncg' 
        以及在 version 0.14.0，还能自定义算法
        以上算法的解释和相关用法见 minimize 函数的官方说明文档，一般求极值多用 'SLSQP'算法
jac:目标函数的雅可比矩阵。可选项，仅适用于CG，BFGS，Newton-CG，L-BFGS-B，TNC，SLSQP，dogleg，trust-ncg。如果jac是布尔值并且为True，则假定fun与目标函数一起返回梯度。如果为False，将以数字方式估计梯度。jac也可以返回目标的梯度。此时，它的参数必须与fun相同。
hess，hessp:可选项，目标函数的Hessian（二阶导数矩阵）或目标函数的Hessian乘以任意向量p。仅适用于Newton-CG，dogleg，trust-ncg。
bounds:可选项，变量的边界（仅适用于L-BFGS-B，TNC和SLSQP）。以（min，max）对的形式定义 x 中每个元素的边界。如果某个参数在 min 或者 max 的一个方向上没有边界，则用 None 标识。如（None, max）
constraints:约束条件（只对 COBYLA 和 SLSQP）。dict 类型。
    type : str， 'eq' 表示等于0，'ineq' 表示不小于0
    fun : 定义约束的目标函数
    jac : 函数的雅可比矩阵 (只用于 SLSQP)，可选项。
    args : fun 和 雅可比矩阵的入参，可选项。
tol:迭代停止的精度。
callback(xk):每次迭代要回调的函数，需要有参数 xk
options:其他选项
    maxiter :  最大迭代次数
    disp :  是否显示过程信息   
"""
if __name__ == '__main__':
    def testFun(x):
        return x[0]**2+x[1]**2+x[2]**2+8
    cons=({'type':'ineq','fun':lambda x:x[0]**2-x[1]+x[2]**2},
          {'type':'ineq','fun':lambda x:-1*(x[0]+x[1]**2+x[2]**2-20)},
          {'type':'eq','fun':lambda x:-1*x[0]-1*x[1]**2+2},
          {'type':'eq','fun':lambda x:x[1]+2*x[2]**2-3})
    x0=np.array([1,1,1])
    bnds=[(0,None) for _ in range(3)]
    res=minimize(testFun,x0,method='SLSQP', bounds=bnds, constraints=cons)
    print(res)