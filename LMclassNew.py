import numpy as np

#from numba import njit,float64,int32
#from numba.experimental import jitclass

#TODO: make the computation of the Jacobian multithreaded

"""
Uses the Levenberg Marquardt alorithm to attempt to find an input x to F that will produce a result as close to yData as possible (minimum least squared error)

Parameters:
    F (float[])->float[]: the function to be optimized
    bounds (float[]): a set of bounds to constrain possible x guesses, arguments should come in pairs ((lower bound,upper bound),(lower bound,upper bound),...) use empty list for each unbounded input.
    x0 (float[]): the first input guess. the input guess should ideally be as close to the true optimum as possible
    yData (float[]): the optimal output that Levenberg Marquardt will attempt to match
    acceptThresh (float): a parameter to Levenberg Marquardt used to determine whether a guess will be accepted by the algorithm as an improvement
    jacobianDelta (float): the step size that will be used to compute the jacobian via finite differences
    maxIters (float): the maximum number of iterations before the algorithm automatically stops
    errorStopThresh (float): if the output least squared error falls below this threshold, the program will stop automatically
    jacUpdateInterval (int): how many accepted steps between updates of the jacobian, 1 is the default Levenberg Marquardt algorithm behavior
    scalerInit (float): initial value for the lambda parameter to Levenberg Marquardt, which dictates how much the algorithm will follow gradient descent vs gauss newton by default
    scaleUpFactor (float): factor to increase the lambda parameter by every time Levenberg Marquardt does not find a good trajectory, which makes its behavior more similar to gradient descent
    scaleDownFactor (float): factor to decrease the lambda parameter by every time Levenberg Marquardt finds a good trajectory, which makes it behavior more similar to gauss newton
    JacobianFunction (float[])->float[]: If a jacobian function is provided, it will be called after every call to F to get the local jacobian. Every i,j entry of the jacobian matrix should correspond to the partial derivative of output i with respect to input j.
    executor (cocurrent.futures.executor): optional, used to mulithread computation of the jacobian. will not work with ipython notebooks
"""
class LevenbergMarquardt():
    def __init__(self,F,bounds,x0,yData,scalerInit=1,scaleUpFactor=11,scaleDownFactor=9,maxInitialDelta=0.1,jacobianDelta=1e-4,acceptThresh=0,jacUpdateInterval=1,badRunJacIntervalWeight=0.25,JacobianFunction=None):
        self.F=F
        self.bounds=np.array(bounds)
        self.x=np.array(x0,dtype=np.float64,copy=True).reshape(-1)
        self.x0=np.copy(self.x)
        self.bestX=np.copy(self.x)
        self.yData=np.array(yData,dtype=np.float64,copy=True).reshape(-1)
        self.scaler=float(scalerInit)
        self.scaleUpFactor=scaleUpFactor
        self.scaleDownFactor=scaleDownFactor
        self.maxInitialDelta=maxInitialDelta
        self.jacobianDelta=jacobianDelta
        self.acceptThresh=acceptThresh
        self.jacUpdateInterval=jacUpdateInterval
        self.badRunJacIntervalWeight=badRunJacIntervalWeight
        self.JacobianFunction=JacobianFunction
        self.nCalls=0
        BoundInputs(self.x,self.bounds,throwError=True)
        self.nIters=0
        self.errorRecord=[]
        self.improvementRecord=[]
        self.bestY=None
        self.localSens=None
        self.deltas=None
        self.stuckCount=0
        self.jacUpdate=0
        self.yErrorBest=None
        self.yNextError=0
        self.xNext=None
        self.yNext=None
        self._acceptedStep=False

    def Step(self,executor=None):
        if self.nIters==0:
            self.yNext,self.J=GetJacobianPlusY(self.F,self.x,self.jacobianDelta,self.bounds,self.JacobianFunction,executor)
            self.bestY=self.yNext
            self.y=self.yNext.reshape(-1)
            self.yErrorBest=GetError(self.y,self.yData)
            self.errorRecord.append(self.yErrorBest)
            self.localSens=LocalSensitivityAnalysis(self.J)
            self.nCalls+=self.x.size+1
            self.H=self.J.T @ self.J
        elif self._acceptedStep:
            self._acceptedStep=False
            self.jacUpdate+=1
            if(self.jacUpdate-self.jacUpdateInterval>=0):
                self.J = GetJacobian(self.F,self.xNext,self.yNext,self.jacobianDelta,self.bounds,self.JacobianFunction,executor)
                self.nCalls+=self.x.size
                self.jacUpdate=0
            self.y=self.yNext
            self.bestY=self.yNext
            self.x=self.xNext
            self.yErrorBest=self.yNextError
            self.scaler/=self.scaleDownFactor
            self.H=self.J.T @ self.J
            self.stuckCount=0
            self.improvementRecord.append(self.yNextError)
        else:
            self.stuckCount+=1
            self.jacUpdate+=self.badRunJacIntervalWeight
            self.scaler*=self.scaleUpFactor
        looping=True
        while(looping):
            if (~np.isfinite(self.yData)).any():print("yData matrix contains nan!")
            if (~np.isfinite(self.y)).any():print("y matrix contains nan!")
            if (~np.isfinite(self.J)).any(): print("jacobian matrix contains nan!")
            if (~np.isfinite(self.H)).any(): print("hessian matrix contains nan!")
            self.deltas=np.linalg.lstsq((self.H+self.scaler*np.diag(np.diag(self.H))).astype(float),(self.J.T @ (self.yData-self.y)).astype(float),rcond=-1)[0]
            if self.nIters==0 and self.maxInitialDelta is not None and np.max(np.abs(self.deltas)/(self.bounds[:,1]-self.bounds[:,0]))>self.maxInitialDelta: self.scaler*=self.scaleUpFactor
            else: looping=False

        self.xNext=self.x+self.deltas
        BoundInputs(self.xNext,self.bounds)
        self.yNext=self.F(self.xNext).reshape(-1)
        self.nCalls+=1
        self.yNextError=GetError(self.yNext,self.yData)
        self.errorRecord.append(self.yNextError)
#        yield {'fit':xNext,'iterations':i+1,'functionCalls':nCalls,'error':errorRecord[-1],'lastOut':lastOut,"scaler":scaler}
        # LM may return early if the error falls below the acceptable threshold
        self.nIters+=1
        if Metric(self.yErrorBest,self.yNextError,self.J,self.deltas,self.scaler,self.yData,self.yNext)>self.acceptThresh: 
            self.bestY=self.yNext
            self.bestX=self.xNext
            self._acceptedStep=True
            self.yErrorBest=self.yNextError

    def DefaultRun(self,errorComparisonStop,errorComparisonInterval=10,stuckStopThresh=25,maxIters=100,executor=None)->str:
        for i in range(maxIters):
            step=self.Step(executor)
            if self.stuckCount==stuckStopThresh:
                return "stuck"
            if len(self.improvementRecord)>=errorComparisonInterval and self.improvementRecord[-errorComparisonInterval] - self.improvementRecord[-1] < errorComparisonStop:
                return 'errorComparison'
        return 'maxIters'


#force inputs to be within bounds if they were provided
#@njit
def BoundInputs(x,bounds,throwError=False):
    for i in range(x.size):
        currBounds=bounds[i]
        if len(currBounds)>0:
            if x[i]<currBounds[0]:
                if throwError:raise Exception(f"input entry {i}: {x[i]} out of bound: {currBounds}")
                x[i]=currBounds[0]
            elif x[i]>currBounds[1]:
                if throwError:raise Exception(f"input entry {i}: {x[i]} out of bound: {currBounds}")
                x[i]=currBounds[1]

#not used
#@njit
def BroydenUpdate(J,y,yNext,deltas):
    return J+(np.outer(yNext - y - J @ deltas,deltas))/(deltas.T @ deltas)
    

#taken from The Levenberg-Marquardt method for nonlinear least squares curve-fitting problem, @Henri P. Gavin
#@njit
def Metric(yErrorBest,yNextError,J,deltas,scaler,yData,yNext):
    return yErrorBest-yNextError
#    return (yError-yNextError)/(2*deltas.T @ (scaler * deltas + J.T @ (yData-yNext)))

def LocalSensitivityAnalysis(J):
    return (J*J).sum(0)

#@njit
def GetError(y,yData):
    solutionDeltas=yData-y
    return solutionDeltas.T @ solutionDeltas

def DeltasToJac(deltaY,deltaX,y,direction):
    if direction==1:
        return (deltaY-y)/deltaX
    else:
        return (y-deltaY)/deltaX

def JacobianSingleVar(inputArr):
    F=inputArr[0]
    x=np.copy(inputArr[1])
    y=inputArr[2]
    if (~np.isfinite(y)).any():
        print(f"function applied to {x} results in nan")
    delta=inputArr[3]
    bounds=inputArr[4]
    i=inputArr[5]
    currBounds=bounds[i]
    boundsRange=np.abs(currBounds[0]-currBounds[1])
    delta*=boundsRange
    jaci=np.zeros((y.size),dtype=np.float64)
    if len(currBounds)>0 and x[i]+delta<currBounds[1] and x[i]+delta>currBounds[0]:
        x[i]+=delta
        deltaY=F(x).reshape(-1)
        if (~np.isfinite(deltaY)).any():
            print(f"function applied to {x} results in nan")
        jaci=(deltaY-y)/delta
    #if the Jacobian calculation would put the computation of F out of bounds, reverse the gradient
    else:
        x[i]-=delta
        deltaY=F(x).reshape(-1)
        if (~np.isfinite(deltaY)).any():
            print(f"function applied to {x} results in nan")
        jaci=(y-deltaY)/delta
    return jaci


#@njit
def GetJacobianPlusY(F,x,delta,bounds,JacobianFunction,executor):
    if JacobianFunction!=None:return F(x),JacobianFunction(x)
    if executor!=None:
        deltaYs=executor.map(JacobianPlusY,[(F,x,delta,bounds,i) for i in range(-1,len(x))])
#        jobs=[executor.submit(JacobianPlusY,(F,x,delta,bounds,i)) for i in range(-1,len(x))]
#        deltaYs=[x.result() for x in jobs]
        y=deltaYs[0]
        Jacs=[DeltasToJac(deltaY,delta,y.reshape(-1),direction) for deltaY,direction in deltaYs[1:]]
        return y,np.stack(Jacs).T
    else: 
        y=F(x)
        return y,np.stack([JacobianSingleVar((F,x,y.reshape(-1),delta,bounds,i)) for i in range(len(x))]).T

def GetJacobian(F,x,y,delta,bounds,JacobianFunction,executor):
    if JacobianFunction!=None:return JacobianFunction(x)
    if executor!=None:
        return np.stack(executor.map(JacobianSingleVar,[(F,x,y,delta,bounds,i) for i in range(len(x))]))
#        jobs=[executor.submit(JacobianSingleVar,(F,x,y,delta,bounds,i)) for i in range(len(x))]
#        return np.stack([x.result() for x in jobs]).T
    else: return np.stack([JacobianSingleVar((F,x,y,delta,bounds,i)) for i in range(len(x))]).T


def JacobianPlusY(inputArr):
    F=inputArr[0]
    x=np.copy(inputArr[1])
    delta=inputArr[2]
    bounds=inputArr[3]
    i=inputArr[4]
    if i==-1: return F(x)
    currBounds=bounds[i]
    boundsRange=np.abs(currBounds[0]-currBounds[1])
    delta*=boundsRange
    if len(currBounds)>0 and x[i]+delta<currBounds[1] and x[i]+delta>currBounds[0]:
        direction=1
        x[i]+=delta
        deltaY=F(x).reshape(-1)
        if (~np.isfinite(deltaY)).any():
            print(f"function applied to {x} results in nan")
    #if the Jacobian calculation would put the computation of F out of bounds, reverse the gradient
    else:
        direction=-1
        x[i]-=delta
        deltaY=F(x).reshape(-1)
        if (~np.isfinite(deltaY)).any():
            print(f"function applied to {x} results in nan")
    return deltaY,direction
