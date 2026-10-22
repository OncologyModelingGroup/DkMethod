#%% IMPORTS AND DEFINES
import os
import matplotlib.pyplot as plt
from LMclassNew import LevenbergMarquardt
from functools import partial
import math
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
from dataclasses import dataclass,replace
from multiprocessing import Pool
import os
from PDEgrid import PDEgrid

@dataclass
class ModelParams:d:float;k:float;steps:float;sideLen:int;dx:float=1;dt:float=1;com:np.ndarray=None
@dataclass
class ModelResults:params:ModelParams;finalState:np.ndarray;dimensons:int;intermediateStates:np.ndarray=None;saveFreq:int=None;weights:np.ndarray=None;id:str=None;noiseStd:float=None
def RunModel3D(params:ModelParams,saveFreq=None,stopDensity=None,return3D=False)->ModelResults:
#    mask=GenRadialMask([params.sideLen,params.sideLen,params.sideLen])
    model=SetupModel(1E-6,[params.dx,params.dx,params.dx],[params.sideLen,params.sideLen,params.sideLen],params.com)
    tFrac,tWhole=math.modf(params.steps)
    intermediateStates=[] if saveFreq is not None else None
    for t in range(int(tWhole)): 
        if saveFreq is not None and t % saveFreq==0: intermediateStates.append(Model3Dto1D(model,return3D))
        #if stopDensity is not None and GetDensity1D(Model3Dto1D(model))>stopDensity: return ModelResults(replace(params,steps=t),Model3Dto1D(model,return3D),3,intermediateStates,saveFreq)
        if stopDensity is not None and GetDensity3D(model._field)>stopDensity: return ModelResults(replace(params,steps=t),Model3Dto1D(model,return3D),3,intermediateStates,saveFreq)
        StepModelNoDrug(model,None,params.d,params.k)
    if tFrac>0: StepModelNoDrug(model,None,params.d,params.k,tFrac)
    if saveFreq is not None: intermediateStates=np.array(intermediateStates)
    return ModelResults(replace(params),Model3Dto1D(model,return3D),3,intermediateStates,saveFreq)

def SetupModel(initDensity,dims,shape,com=None):
    if com is None:com=np.array((int(shape[0]//2),int(shape[1]//2),int(shape[2]//2)))
    model=PDEgrid(tuple(shape))
    model.SetDxDyDzDt(dims[0],dims[1],dims[2],1)
    model.Set3D(com[0],com[1],com[2],initDensity/model.GetVoxelVol())
    return model

@njit
def StepModelNoDrug(model:PDEgrid,mask,d,k,stepFrac=1):
    if mask==None: model.Diffusion(d*stepFrac)
    else: model.DiffusionBc(d*stepFrac,mask)
    for x in range(model.Xdim()):
        for y in range(model.Ydim()):
            for z in range(model.Zdim()):
                pop=model.Get3D(x,y,z)
                model.Add3D(x,y,z,(pop*(k*(1-pop)))*model.Dt()*stepFrac)
    model.Update()

def RunModelSaveDensities3D(params:ModelParams,saveDensities,return3D=False)->ModelResults:
    model=SetupModel(1E-6,[params.dx,params.dx,params.dx],[params.sideLen,params.sideLen,params.sideLen],params.com)
    tFrac,tWhole=math.modf(params.steps)
    intermediateStates=[]
    for t in range(int(tWhole)): 
        if GetDensity3D(model._field)>saveDensities[len(intermediateStates)]:
            intermediateStates.append(Model3Dto1D(model,return3D))
        if len(intermediateStates)==len(saveDensities): 
            return ModelResults(replace(params,steps=t),saveDensities,3,intermediateStates,None)
        StepModelNoDrug(model,None,params.d,params.k)
    if tFrac>0: StepModelNoDrug(model,None,params.d,params.k,tFrac)
    return ModelResults(replace(params),saveDensities,3,intermediateStates,None)

@njit
def GetModelSum(model:PDEgrid):
    sum=0
    for i in range(len(model)): sum+=model.Get(i)
    return sum

#@njit
def GenRadialMask(shapeOut):
    com=(shapeOut[0]//2,shapeOut[1]//2,shapeOut[2]//2)
    maxRad=shapeOut[0]//2+1
    out=np.zeros((shapeOut[0],shapeOut[1],shapeOut[2]))
    for x in range(shapeOut[0]):
        for y in range(shapeOut[1]):
            for z in range(shapeOut[2]):
                xDisp=com[0]-x
                yDisp=com[1]-y
                zDisp=com[2]-z
                dist=np.sqrt(xDisp*xDisp+yDisp*yDisp+zDisp*zDisp)
                idx=int(dist)
                frac=dist-idx
                if frac==0 and idx<maxRad:out[x,y,z]=1
                elif idx<maxRad-1: out[x,y,z]=1
                else: out[x,y,z]=-1
    return out

#@njit
def SetupModel1D(sideLen,dims,dt,densityInit)->PDEgrid:
    model=PDEgrid((sideLen,))
    model.SetDxDyDzDt(dims[0],dims[1],dims[2],dt)
    model.Set(0,densityInit/(dims[0]*dims[1]*dims[2]))
    return model

#def _DiffusionSpherical1DfvTerm(self,rateConstant,vi,ai,cm,cp,dr,dt):
#    return ((rateConstant*ai*dt)/(vi*dr))*((cp-cm))
#
#def DiffusionSpherical1Dfv(self,rateConstant):
#    v0=(4/3)*np.pi*0.5**3
#    a0=4*np.pi*0.5**2
#    self._deltas[0]+=_DiffusionSpherical1DfvTerm(self,rateConstant,v0,a0,self._field[0],self._field[1],self._dx,self._dt)
#    for i in range(1,self.Xdim()-1):
#        vi=(4/3)*np.pi*((i+0.5)**3-(i-0.5)**3)
#        ap=4*np.pi*(i+0.5)**2
#        am=4*np.pi*(i-0.5)**2
#        self._deltas[i]+=_DiffusionSpherical1DfvTerm(self,rateConstant,vi,ap,self._field[i],self._field[i+1],self._dx,self._dt)-_DiffusionSpherical1DfvTerm(self,rateConstant,vi,am,self._field[i-1],self._field[i],self._dx,self._dt)
#    vf=(4/3)*np.pi*(self.Xdim()-0.5)**3-(self.Xdim()-1.5)**3
#    af=4*np.pi*(self.Xdim()-1.5)**3
#    self._deltas[self.Xdim()-1]+=_DiffusionSpherical1DfvTerm(self,rateConstant,vf,af,self.field[self.Xdim()-2],self.field[self.Xdim()-1],self._dx,self._dt)

def DiffusionSpherical1D(self,rateConstant):
    self._deltas[0]+=(6*rateConstant*(self._field[1]-self._field[0]))*self._dtdxSq
    for x in range(1,self.Xdim()-1):
        lap=(self._field[x+1]-2*self._field[x]+self._field[x-1])*self._dtdxSq
        deriv=(self._field[x+1]-self._field[x-1])*self._dtdxHalf
        self._deltas[x]+=rateConstant*lap+((2*rateConstant)/(x*self._dx))*deriv
    lap=self._field[self.Xdim()-2]-self._field[self.Xdim()-1]*self._dtdxSq
    deriv=self._field[self.Xdim()-1]-self._field[self.Xdim()-2]*self._dtdxHalf
    self._deltas[self.Xdim()-1]+=rateConstant*lap+((2*rateConstant)/(x*self._dx))*deriv

@njit
def StepModel1D(model:PDEgrid,d,k,stepFrac=1):
    model.DiffusionSpherical1Dfv(d*stepFrac)
    #model.DiffusionSpherical1Dfv(d*stepFrac)
    for i in range(len(model)):
        n=model.Get(i)
        model.Add(i,n*k*stepFrac*model.Dt()*(1-n))
    model.Update()

def Model1Dto3D(model:PDEgrid,shapeOut=None,com=None):
    if shapeOut==None: shapeOut=np.array([len(model)*2-1,len(model)*2-1,len(model)*2-1])
    if com==None: com=np.array([shapeOut[0]//2,shapeOut[1]//2,shapeOut[2]//2])
    return _Model1Dto3D(model,shapeOut,com)

@njit
def Vec1Dto2D(radVec:np.ndarray):
    shapeOut=(len(radVec)*2-1,len(radVec)*2-1)
    com=(len(radVec)-1,len(radVec)-1)
    out=np.zeros(shapeOut)
    for x in range(shapeOut[0]):
        for y in range(shapeOut[1]):
            xDisp=com[0]-x
            yDisp=com[1]-y
            dist=np.sqrt(xDisp*xDisp+yDisp*yDisp)
            idx=int(dist)
            frac=dist-idx
            if frac==0 and idx<len(radVec):out[x,y]=radVec[idx]
            elif idx<len(radVec)-1: out[x,y]=radVec[idx]*(1-frac)+radVec[idx+1]*(frac)
    return out

def PlotVec2D(radVec:np.ndarray):
    plt.imshow(Vec1Dto2D(radVec),vmin=0,vmax=1)

@njit
def _Model1Dto3D(model:PDEgrid,shapeOut,com):
    out=np.zeros((shapeOut[0],shapeOut[1],shapeOut[2]))
    for x in range(shapeOut[0]):
        for y in range(shapeOut[1]):
            for z in range(shapeOut[2]):
                xDisp=com[0]-x
                yDisp=com[1]-y
                zDisp=com[2]-z
                dist=np.sqrt(xDisp*xDisp+yDisp*yDisp+zDisp*zDisp)
                idx=int(dist)
                frac=dist-idx
                if frac==0 and idx<len(model):out[x,y,z]=model[idx]
                elif idx<len(model)-1: out[x,y,z]=model[idx]*(1-frac)+model[idx+1]*(frac)
    return out

@njit
def _Model3Dto1D(model:PDEgrid):
    out=np.zeros(model.Xdim()//2+1)
    center=np.array([model.Xdim()//2,model.Ydim()//2,model.Zdim()//2])
    for x in range(center[0],model.Xdim()):
        out[x-center[0]]=model.Get3D(x,center[1],center[2])
    return out

def Model3Dto1D(model:PDEgrid,return3D=False):
    if return3D:return model.GetFieldCopy3D()
    else: return _Model3Dto1D(model)

def Field3Dto1D(field:np.ndarray):
    return field[field.shape[0]//2,field.shape[1]//2,field.shape[2]//2:field.shape[2]]


def GetVoxelAvgs(field:np.ndarray):
    maxDist=field.shape[0]//2+1
    center=np.array([field.shape[0]/2,field.shape[0]/2,field.shape[0]/2])
    return GetVoxelAvgsInner(field,center,maxDist)

#@njit
def GetVoxelAvgsInner(field:np.ndarray,center:np.ndarray,maxDist):
    avgs=np.zeros(maxDist)
    counts=np.zeros(maxDist)
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            for z in range(field.shape[2]):
                dist=np.sqrt(((x+0.5)-center[0])**2+((y+0.5)-center[1])**2+((z+0.5)-center[2])**2)
                idist=int(dist+0.5)
                if idist<maxDist:
                    avgs[idist]+=field[x,y,z]
                    counts[idist]+=1
    return avgs/counts


def RunModel1D(params:ModelParams,saveFreq=None,stopDensity=None)->ModelResults:
#    shape=[params.sideLen,params.sideLen,params.sideLen]
    model=SetupModel1D(params.sideLen//2+1,[params.dx,params.dx,params.dx],params.dt,1E-6)
    tFrac,tWhole=math.modf(params.steps)
    intermediateStates=[] if saveFreq is not None else None
    for t in range(int(tWhole)): 
        if saveFreq is not None and t % saveFreq==0: intermediateStates.append(model.GetFieldCopy1D())
#        if stopDensity is not None and GetDensity3D(Model1Dto3D(model))>stopDensity: return ModelResults(replace(params,steps=t),model.GetFieldCopy1D(),3,intermediateStates,saveFreq)
        if stopDensity is not None and GetDensity1D(model.GetFieldCopy1D())>stopDensity: return ModelResults(replace(params,steps=t),model.GetFieldCopy1D(),3,intermediateStates,saveFreq)
        StepModel1D(model,params.d,params.k,1)
    if tFrac>0: StepModel1D(model,params.d,params.k,tFrac)
    if saveFreq is not None: intermediateStates=np.array(intermediateStates)
    return ModelResults(replace(params),model.GetFieldCopy1D(),1,intermediateStates,saveFreq)

def ResampleField(field:np.ndarray, newLength):
    step=len(field)
    original_indices = np.linspace(0, len(field) - 1, num=len(field))
    new_indices = np.linspace(0, len(field) - 1, num=newLength)
    return np.interp(new_indices, original_indices, field)

def RescaleDx(res:ModelResults,newSideLen):
    oldDx=res.params.dx
    oldLen=res.params.sideLen
    newTumor=ResampleField(res.finalState,newSideLen//2+1)
    newDx=(oldLen+1)/(newSideLen+1)*oldDx
    newParams=replace(res.params,dx=newDx)
    return replace(res,params=newParams,finalState=newTumor)

@njit
def GetShellVolArr(length):
    out=np.zeros(length)
    for i in range(length):
        out[i]=GetShellVol(i)
    return out

@njit
def GetShellVol(iRad):
    if iRad==0: return (4/3)*np.pi*0.5**3
    else: return (4/3)*np.pi*((iRad+0.5)**3-(iRad-0.5)**3)

def GetError(field1,field2,weights): 
    if weights is None:return np.sum((field1-field2)**2)
    return np.sum((field1-field2)**2*weights)

@njit
def GetDensity1D(radVec:np.ndarray):
    densityTotal=radVec[0]*(4/3)*np.pi*(0.5**3)
    for i in range(1,len(radVec)): 
        densityTotal+=radVec[i]*(4/3)*np.pi*((i+0.5)**3-(i-0.5)**3)
    return densityTotal/((4/3)*np.pi*(len(radVec)-0.5)**3)

@njit
def GetDensity3D(field:np.ndarray):
    fieldFlat=np.ravel(field)
    densitySum=0
    for i in range(len(fieldFlat)):densitySum+=fieldFlat[i]
    return densitySum/len(fieldFlat)


#def GetDensity3D(field:np.ndarray): return np.sum(field)/field.size

def LmRun(x,startParams:ModelParams,weights=None):
    inputParams=replace(startParams,d=10**x[0],steps=x[1])
    return ApplyWeightsSqrt(RunModel1D(inputParams).finalState,weights)

def LmRunK(x,startParams:ModelParams,weights=None):
    inputParams=replace(startParams,k=10**x[0],steps=x[1])
    return ApplyWeightsSqrt(RunModel1D(inputParams).finalState,weights)

#@njit
def ApplyWeightsSqrt(radVec,weights):
    if weights is None: return radVec
    out=np.zeros(len(radVec))
    for i in range(len(radVec)):
        out[i]=radVec[i]*np.sqrt(weights[i])
    return out


def GridSearch(truth:ModelResults,grid:list[ModelResults])->ModelParams:
    bestError=np.finfo(np.float64).max
    bestRes:ModelResults=None
    bestiState=-1
    for g in grid:
        for i,state in enumerate(g.intermediateStates):
            error=GetError(truth.finalState,state,truth.weights)
            if error<bestError:
                bestRes=g
                bestiState=i
                bestError=error
    return replace(bestRes.params,steps=bestiState*bestRes.saveFreq)

def LmSearch(truth:ModelResults,start:ModelParams,bounds)->ModelResults:
    lm=LevenbergMarquardt(partial(LmRun,startParams=start,weights=truth.weights),bounds,(np.log10(start.d),start.steps),ApplyWeightsSqrt(truth.finalState,truth.weights))
    lm.DefaultRun(0.0001)
    return ModelResults(replace(start,d=10**lm.bestX[0],steps=lm.bestX[1]),lm.bestY,dimensons=len(lm.bestY),weights=truth.weights,id=truth.id,noiseStd=truth.noiseStd)

def LmSearchK(truth:ModelResults,start:ModelParams,bounds)->ModelResults:
    lm=LevenbergMarquardt(partial(LmRunK,startParams=start,weights=truth.weights),bounds,(np.log10(start.k),start.steps),ApplyWeightsSqrt(truth.finalState,truth.weights))
    lm.DefaultRun(0.0001)
    return ModelResults(replace(start,k=10**lm.bestX[0],steps=lm.bestX[1]),lm.bestY,dimensons=len(lm.bestY),weights=truth.weights,id=truth.id)

def GridLmSearch(truth:ModelResults,grid:list[ModelResults],bounds)->ModelResults:
    print(truth.id)
    start=GridSearch(truth,grid)
    return LmSearch(truth,start,bounds)

def GridLmSearchK(truth:ModelResults,grid:list[ModelResults],bounds)->ModelResults:
    print(truth.id)
    start=GridSearch(truth,grid)
    return LmSearchK(truth,start,bounds)

def KScalingSweepRun(k_dScalar):
    k=k_dScalar[0]
    dScalar=k_dScalar[1]
    return RunModel1D(ModelParams(k*dScalar,k,10//k,200))

def GenGrid(gridParams:list[ModelParams],outPath,saveFreq=1000):
    with Pool() as pool: results=pool.map(partial(RunModel1D,saveFreq=saveFreq),gridParams)
    pd.to_pickle(results,outPath)

def GenScaledGroundTruth(sideLen,dx,outPath):
    print(f"starting {outPath}")
    if not os.path.exists(outPath):
        truthParams=[ModelParams(d,k=1E-2,steps=20000,sideLen=sideLen,dx=dx) for d in np.logspace(-3,-1,10)]
        with Pool() as pool: results=pool.map(partial(RunModel3D,stopDensity=0.1,return3D=True),truthParams)
        pd.to_pickle(results,outPath)

def ViewScaledGroundTruths(data:list[list[ModelResults]]):
    for i in range(len(data)):
        for j in range(len(data[i])):
            plt.title(f"scale {i}, run {j}")
            plt.imshow(data[i][j].finalState[data[i][j].finalState.shape[0]//2],vmin=0,vmax=1)
            plt.show()

def GenScaledGrids(lengths,outputPath):
    lengthGridParams=[[ModelParams(d,k=1E-2,steps=1E6,sideLen=nl-1,dx=200/nl,dt=1E-2) for d in np.logspace(-4,0,100)] for nl in lengths]
    grids=[]
    for gridParams in lengthGridParams: 
        print(f"generating grid for dx {gridParams[0].dx}")
        with Pool() as pool: results=pool.map(partial(RunModel1D,saveFreq=10000),gridParams,chunksize=20)
        grids.append(results)
    pd.to_pickle(grids,outputPath)

def RunLmScaled(lenGrids:list[list[ModelResults]],lenTruths:list[list[ModelResults]],outPath):
    lmRes=[]
    for i in range(len(lenGrids)):
        print(f"running lm for dx {lenGrids[i][0].params.dx}")
        with Pool() as pool: 
            results=pool.map(partial(GridLmSearch,grid=lenGrids[i],bounds=((-4,0),(0,1E6))),[replace(t,finalState=Field3Dto1D(t.finalState)) for t in lenTruths[i]])
            #results=pool.map(partial(GridSearch,grid=lenGrids[i]),lenTruths[i])
            lmRes.append(results)
    pd.to_pickle(lmRes,outPath)

def DkAccuracyDf(truths:list[ModelResults],res:list[ModelResults]):
    df=pd.DataFrame({'D True':truths[i].params.d,'k True':truths[i].params.k,'T True':truths[i].params.steps, 'dx True':truths[i].params.dx,
                     'D':res[i].params.d,'k':res[i].params.k,'T':res[i].params.steps, 'dx':res[i].params.dx ,'noiseStd':res[i].noiseStd} for i in range(len(truths)))
    df['D/k True']=df['D True']/df['k True']
    df['D/k']=df['D']/df['k']
    df['D/k percent difference']=(df['D/k']-df['D/k True'])*100/df['D/k True']
    return df

def DkAccuracyDfAndFigure(truths:list[list[ModelResults]],res:list[list[ModelResults]],outPath,col='dx',kwargs={}):
    df=pd.concat([DkAccuracyDf(truths[i],res[i])for i in range(len(truths))])
    DkAccuracyFigure(df,outPath,col,kwargs)

def DkAccuracyFigure(df,outPath,col='dx',kwargs={}):
    sns.lineplot(data=df,x='D/k True',y='D/k percent difference',hue=col,palette='bright',**kwargs)
    plt.xlabel('$\\it{D}$/$\\it{k}$ true')
    plt.ylabel('$\\it{D}$/$\\it{k}$ percent error')
    plt.xscale('log')
#    fg=sns.FacetGrid(data=df,col=col)
#    fg.set(xscale='log',**kwargs)
#    fg.map_dataframe(sns.lineplot,x='D/k True',y='D/k percent difference')
    plt.tight_layout()
    plt.legend().remove()
    plt.savefig(outPath)
    plt.clf()


def GenNoiseField(field,stdDev):
    noise=np.random.normal(loc=0,scale=stdDev,size=field.shape)
    out=np.copy(field)+noise
    out=np.clip(out,0,1)
    return out

def NoiseFitting(fields:list[ModelResults],stds,grid:list[ModelResults],repeats)->list[ModelResults]:
    print(f"running noise fitting for grid dims: {grid[0].dimensons}")
    fields=[replace(field,finalState=GenNoiseField(field.finalState,std),noiseStd=std) for i in range(repeats) for std in stds for field in fields]
    with Pool() as pool: results=pool.map(partial(GridLmSearch,grid=grid,bounds=((-4,0),(0,1E6))),[replace(t,finalState=Field3Dto1D(t.finalState)) for t in fields])
    return results

def NoiseDf(truths,baseRes,noiseRes,outPath):
    baseRes=DkAccuracyDf(truths,baseRes)
    baseRes['noiseStd']=0
    dfs=[baseRes]
    for i in range(len(noiseRes)//len(truths)):
        dfs.append(DkAccuracyDf(truths,noiseRes[i*len(truths):(i+1)*len(truths)]))
    pd.to_pickle(pd.concat(dfs),outPath)

def PlotNoise(df,outPath):
    df['noise standard dev.']=df['noiseStd']
    fg=sns.FacetGrid(data=df,col='noise standard dev.',col_wrap=3)
    df['$\\it{D}$/$\\it{k}$ true']=df['D/k True']
    df['$\\it{D}$/$\\it{k}$ percent error']=df['D/k percent difference']
    fg.map_dataframe(sns.scatterplot,data=df,x='$\\it{D}$/$\\it{k}$ true',y='$\\it{D}$/$\\it{k}$ percent error',s=5)
#    fg.set(ylim=(-100,100))
    fg.set(xscale="log")
    plt.gcf().set_size_inches(7.5,2.5)
    plt.savefig(outPath)
    plt.clf()

def PlotNoNoise(df,outPath):
    df=df[df['noiseStd']==0]
    sns.scatterplot(data=df,x='D/k True',y='D/k percent difference')
    plt.ylim(-10,10)
    plt.xscale('log')
    plt.gcf().set_size_inches(5,5)
    plt.tight_layout()
    plt.savefig(outPath)
    plt.clf()

def GenGroundTruths(outDir):
    truthNames=["ScaleTruths19_10_3D_StopDensity1.p","ScaleTruths49_4_3D_StopDensity1.p","ScaleTruths99_2_3D_StopDensity1.p","ScaleTruths199_1_3D_StopDensity1.p"]
    GenScaledGroundTruth(19,10,outDir+truthNames[0])
    GenScaledGroundTruth(49,4,outDir+truthNames[1])
    GenScaledGroundTruth(99,2,outDir+truthNames[2])
    GenScaledGroundTruth(199,1,outDir+truthNames[3])
    scaledTruths=[pd.read_pickle(outDir+name) for name in truthNames]
    pd.to_pickle(scaledTruths,outDir+"MultipleLengthTruths.p")

def RunLmDtScaled(lenGrids:list[list[ModelResults]],lenTruths:list[list[ModelResults]],allBounds,outPath):
    lmRes=[]
    for i in range(len(lenGrids)):
        print(f"running lm for k {lenGrids[i][0].params.k}")
        with Pool() as pool: 
            results=pool.map(partial(GridLmSearch,grid=lenGrids[i],bounds=allBounds[i]),[replace(t,finalState=Field3Dto1D(t.finalState)) for t in lenTruths[i]])
            #results=pool.map(partial(GridSearch,grid=lenGrids[i]),lenTruths[i])
            lmRes.append(results)
    pd.to_pickle(lmRes,outPath)
