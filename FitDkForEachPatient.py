#%% IMPORTS AND DEFINES
import nibabel as nib
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import ndarray
from numba import njit
import pandas as pd
from dataclasses import dataclass,replace
import ProcessPatientData as ppd
from Model3Dand1D import ModelParams,ModelResults,GridLmSearch,RunModel1D,GetError,GetShellVol,GetShellVolArr,RunModel3D,GridLmSearchK
from multiprocessing import Pool
from functools import partial
from matplotlib.colors import LogNorm
from PDEgrid import PDEgrid
from lifelines import KaplanMeierFitter
import matplotlib.patches as patches
from scipy.stats import pearsonr


EMPTY,NON,ENH=0,5,6

@dataclass
class DisplayPt():id:str;densityDisplayField:ndarray;tumor:ppd.PtField;counts:ppd.PtRadialCounts;radial:ppd.PtRadial
from ProcessPatientData import PtRadial,PtField,PtRadialCounts

def GenGridSearchDf(truth:ModelResults,grid:list[ModelResults])->pd.DataFrame:
    rows=[]
    for g in grid:
        for i,state in enumerate(g.intermediateStates):
            error=GetError(truth.finalState,state,truth.weights)
            rows.append({"error":error,"T":i*g.saveFreq,"D":g.params.d,"k":g.params.k})
    return pd.DataFrame(rows)

def GenPtGridDf(pt:ppd.PtRadial,grid:list[ModelResults])->pd.DataFrame:
    ptDens=PtRadialToModelResults(pt,"radial")
    df=GenGridSearchDf(ptDens,grid)
    df['D/k']=df['D']/df['k']
    df['D/k']=np.round(df['D/k'],4)
    return df

def GridSearchHeatmap(errorDf,ptId,cbar=True):
    errorDf['t']=errorDf['T']
    errorDf['log (D/k)']=np.log10(errorDf['D/k'])
    iMin=errorDf['error'].idxmin()
    piv=pd.pivot(data=errorDf,index='t',columns='log (D/k)',values='error')
    coordMinDk=errorDf['t'][iMin]
    coordMinT=errorDf['log (D/k)'][iMin]
#    ax=sns.heatmap(piv,norm=LogNorm(),cmap='viridis_r',xticklabels=200,yticklabels=200,cbar=cbar,cbar_kws={'label':'error'})
    plt.imshow(piv,cmap='viridis_r',extent=[piv.columns[0],piv.columns[-1],piv.index[0],piv.index[-1]],norm=LogNorm(vmin=np.min(piv),vmax=np.max(piv)),aspect='auto',origin='lower')
    if cbar:
        bar=plt.colorbar()
        bar.set_label("sum squared error")
    plt.scatter([coordMinT],[coordMinDk],s=5,c='r')
    plt.title(f"pt: {ptId}")
    plt.gca().set_xlabel("log ($\\it{D}$/$\\it{k}$)")
    plt.gca().set_ylabel("$\\it{t}$")
#    plt.gca().invert_yaxis()

def DkErrorPlot(errorDf:pd.DataFrame,res:ModelResults,ptId):
    errorDf['log ($\\it{D}$/$\\it{k}$)']=np.log10(errorDf['D/k'])
    idxs=errorDf.groupby(by='log ($\\it{D}$/$\\it{k}$)')['error'].idxmin()
    mins=errorDf.loc[idxs]
    sns.lineplot(data=mins,x='log ($\\it{D}$/$\\it{k}$)',y='error')
    plt.axvline(x=np.log10(res.params.d/res.params.k),c='r')
    plt.xticks([-1,0,1,2])
#    plt.xscale("log")

def GetDensityDist(res:ModelResults): return res.finalState/np.sqrt(res.weights) if res.weights is not None else res.finalState

def CompareDensities(pt:ppd.PtRadial,res:ModelResults,ptId):
    plt.plot(pt.tumor)
    plt.plot(res.finalState/np.sqrt(res.weights))
    plt.xlabel("radius (mm)")
    plt.ylabel("density (normalized)")
    plt.title(f"pt: {ptId}")
    
def GenRadialPt(radialCounts:ppd.PtRadialCounts,enhDensity,nonDensity,length=None)->ppd.PtRadial:
    counts=np.where(radialCounts.counts==0,1,radialCounts.counts)
    tumor=(radialCounts.enh*enhDensity+radialCounts.non*nonDensity)/counts
    if length!=None: tumor=np.concatenate((tumor,np.zeros(length-len(tumor))))
    return ppd.PtRadial(tumor,enhDensity,nonDensity,radialCounts.id,counts)

#draw point on heatmap, draw vertical line on D/k vs error line plot, draw best fit line plot to compare to ground truth
def VerifyPatientResults(pt:ppd.PtRadial,res:ModelResults,grid:list[ModelResults],ptId):
    pt=GenRadialPt(pt,0.9,0.45,101)
    df=GenPtGridDf(pt,grid)
    #GridSearchHeatmap(df,ptId)
    #plt.show()
    plt.subplot(1,2,1)
    DkErrorPlot(df,res,ptId)
    plt.subplot(1,2,2)
    CompareDensities(pt,res,ptId)
    plt.tight_layout()
    plt.show()

def PtRadialToModelResults(pt:ppd.PtRadial,scaling)->ModelResults: 
    if scaling=="unscaled":weights=None
    elif scaling=="count": weights=pt.counts
    elif scaling=="radial": weights=GetShellVolArr(len(pt.counts))
    else: raise Exception("scaling option must be one of unscaled, count or radial not: "+scaling)
    return ModelResults(None,pt.tumor,None,None,None,weights,pt.id)

def GenRunName(enhMax,isScaled): return f"sweepRes_{float(enhMax)}_{enhMax/2}_{isScaled}.p"

#scaling options: "unscaled", "scaled", "counts"
def RunFitting(grid:list[ModelResults],pts:dict[str,ppd.PtRadialCounts],enhMax,scaling,voxelCounts=None):
    ptDens=[PtRadialToModelResults(GenRadialPt(pts[ptId],enhMax,enhMax/2),scaling) for ptId in pts.keys()]
    with Pool() as pool:
        results=pool.map(partial(GridLmSearch,grid=grid,bounds=((-5,-2),(0,1E6))),ptDens)
    return results
    
def GenResDf(res,pts,enhMax,isScaled):
    ids=list(pts.keys())
#    res:list[ModelResults]=pd.read_pickle(DATA_PATH+"testGridResults_1_05.p")
    resDf=pd.DataFrame([{"id":ids[i],"D":res[i].params.d,"k":res[i].params.k,"T":res[i].params.steps} for i in range(len(res))])
    resDf['D/k']=resDf['D']/resDf['k']
    resDf['enhMax']=enhMax
    resDf['isScaled']=isScaled
    return resDf
def PlotComRadial(comX,comY,pt:ppd.PtField,radial:ppd.PtRadial,counts:ppd.PtRadialCounts):
    cmap=plt.cm.viridis
    norm=plt.Normalize(-0.2,0.9)
    for i in range(100,0,-1):
        color=cmap(norm(radial.tumor[i]))
        circ=patches.Circle(xy=(comX,comY),radius=0.5+i,edgecolor='none',facecolor=color)
        plt.gca().add_patch(circ)
    circ=patches.Circle(xy=(comX,comY),radius=100.5,edgecolor='k',facecolor="none")
    plt.gca().add_patch(circ)
    circ=patches.Circle(xy=(comX,comY),radius=1.5,edgecolor='r',facecolor='r')
    plt.gca().add_patch(circ)

def PlotSlice(tumor,pt,radial,counts): 
    plt.imshow(tumor,interpolation='nearest',origin='lower',vmin=-0.2,vmax=0.9)
    plt.gca().set_xlabel("x (mm)")
    plt.gca().set_ylabel("y (mm)")
def PlotComOnly(comX,comY,pt,raidal,counts):
    circ=patches.Circle(xy=(comX,comY),radius=1.5,edgecolor='r',facecolor='r')
    plt.gca().add_patch(circ)
def PlotComLines(comX,comY,pt,raidal,counts): 
    for i in range(0,101,10):
        circ=patches.Circle(xy=(comX,comY),radius=0.5+i,edgecolor='k',facecolor='none')
        plt.gca().add_patch(circ)
    circ=patches.Circle(xy=(comX,comY),radius=1.5,edgecolor='r',facecolor='r')
    plt.gca().add_patch(circ)

def GetSaggitalSlice(field,x): return field[int(x)]
def GetCoronalSlice(field,y): return field[:,int(y)]
def GetTransverseSlice(field,z): return field[:,:,int(z)]

def PlotSaggital(pt:DisplayPt,PlotSlice,PlotCom):
    com=pt.tumor.com
    PlotSlice(pt.densityDisplayField[int(com[0])].T,pt.tumor,pt.radial,pt.counts)
    PlotCom(com[1],com[2],pt.tumor,pt.radial,pt.counts)

def PlotCoronal(pt:DisplayPt,PlotSlice,PlotCom):
    com=pt.tumor.com
    PlotSlice(pt.densityDisplayField[:,int(com[1])].T,pt.tumor,pt.radial,pt.counts)
    PlotCom(com[0],com[2],pt.tumor,pt.radial,pt.counts)

def PlotTransverse(pt:DisplayPt,PlotSlice,PlotCom):
    com=pt.tumor.com
    PlotSlice(pt.densityDisplayField[:,:,int(com[2])].T,pt.tumor,pt.radial,pt.counts)
    PlotCom(com[0],com[1],pt.tumor,pt.radial,pt.counts)

def GetDensityDist(res:ModelResults): return res.finalState/np.sqrt(res.weights) if res.weights is not None else res.finalState
def GetDensityDist(res:ModelResults): return res.finalState/np.sqrt(res.weights) if res.weights is not None else res.finalState

def GetSpatialError(pt:ppd.PtRadialCounts,ptRes:ModelResults,maxEnh):
    densitiesOut=GetDensityDist(ptRes)
    densitiesIn=GenRadialPt(pt,maxEnh,maxEnh/2,101).tumor
    if ptRes.weights is None: err=(densitiesOut-densitiesIn)
    else: err=(densitiesOut-densitiesIn)*ptRes.weights
    return err

def ErrorLinePlotDf(data:dict[str,ppd.PtRadialCounts],res:list[ModelResults],enhMax,isScaled):
    ids=list(data.keys())
    rows=[]
    for i in range(len(ids)):
        pt=data[ids[i]]
        ptRes=res[i]
        errors=GetSpatialError(pt,ptRes,enhMax)
        for i in range(len(errors)):
            rows.append({"error":errors[i],"error^2":errors[i]**2,"id":pt.id,"radius (mm)":i,"enhMax":enhMax,"isScaled":isScaled,'weight':ptRes.weights[i]})
            #rows.append({"avg error / enhMax (density)":errors[i],"avg error^2 / enhMax (density^2)":errors[i]**2,"id":pt.id,"radius (mm)":i,"enhMax":enhMax,"isScaled":isScaled,'weight':ptRes.weights[i]})
    df=pd.DataFrame(rows)
    return df

def ErrorLinePlot(errorDf,abs=False):
    if abs:sns.lineplot(errorDf,y='avg error^2 / enhMax (density^2)',x='radius (mm)',hue='isScaled',style='enhMax')#,hue_order=['radial','count','unscaled'],style_order=enhMaxs,errorbar=None)
    else :sns.lineplot(errorDf,y='avg error / enhMax (density)',x='radius (mm)',hue='isScaled',style='enhMax')#,hue_order=['radial','count','unscaled'],style_order=enhMaxs,errorbar=None)


#def CompareSubsets(dataPath,enhMax1,isScaled1,enhMax2,isScaled2):
#    subset1=GenResDf(pd.read_pickle(dataPath+GenRunName(enhMax1,isScaled1)),data,enhMax1,isScaled1)
#    subset2=GenResDf(pd.read_pickle(dataPath+GenRunName(enhMax2,isScaled2)),data,enhMax2,isScaled2)
#    df=pd.concat([subset1,subset2],ignore_index=True)
#    plt.plot([subset1['D/k'],subset2['D/k']],[subset1['T'],subset2['T']],c='k',zorder=1)
#    sns.scatterplot(df,x='D/k',y='T',hue='isScaled',style='enhMax',hue_order=['radial','count','unscaled'],palette='bright',style_order=[1,0.8,0.6],zorder=2)
#    plt.xscale('log')

def PlotKM(subset,label):
    kmf=KaplanMeierFitter()
    kmf.fit(durations=subset['OS_MONTHS'],event_observed=subset['eventCode'],label=label)
    kmf.plot_survival_function(ci_show=True)

def GenSubsetsLabels(df,cutCol,nCuts):
    df=df.sort_values(by=cutCol)
    subsets=[]
    labels=[]
    for i in range(0,nCuts):
        idxLen=len(df.index)
        labelVal1=np.round(df.iloc[idxLen//nCuts*i]['D/k'],nCuts)
        labelVal2=np.round(df.iloc[idxLen//nCuts*(i+1)]['D/k'],nCuts)
        subsets.append(df[idxLen//nCuts*i:idxLen//nCuts*(i+1)])
        labels.append(f'{labelVal1} <= D/k < {labelVal2}')
    return subsets,labels

def GenDisplayPt(ptPaths,ptId,ComCalc,root)->DisplayPt:
    pt=ppd.GenPt(ptPaths[ptId],ComCalc,root)
    counts=ppd.GenRadialCounts(pt)
    radial=GenRadialPt(counts,0.9,0.45,101)
    scaledTumor=np.where(pt.tumor==6,0.9,pt.tumor)
    scaledTumor=np.where(scaledTumor==5,0.45,scaledTumor)
    scaledTumor=np.where(pt.mask==-1,-0.2,scaledTumor)
    return DisplayPt(ptId,scaledTumor,pt,counts,radial)

def GenPatientDkResDf(res:list[ModelResults],paths=None,outPath=None):
    df=pd.DataFrame([{"id":entry.id,"D":entry.params.d,"k":entry.params.k,"t":entry.params.steps} for entry in res])
    if paths!=None: df['root']=[paths[k]['root'].replace("\\","/").split("/")[-2] for k in paths.keys()]
    df['D/k']=df['D']/df['k']
    if outPath is not None: df.to_csv(outPath,index=None)
    return df


def CalcKd(volEnh,volNon): return (4*np.pi/3)**(2/3)*(6.106/(volNon**(1/3)-volEnh**(1/3)))**2

def GenPatientDks(data:dict[str,ppd.PtRadialCounts],grid:list[ModelResults],outPath)->list[ModelResults]:
    ptDens=[PtRadialToModelResults(GenRadialPt(data[ptId],0.9,0.45,101),"radial") for ptId in data.keys()]
    with Pool() as pool: results=pool.map(partial(GridLmSearch,grid=grid,bounds=((-5,-2),(0,1e6))),ptDens)
    pd.to_pickle(results,outPath)
    return results

def GenPatientDksFlexibleK(data:dict[str,ppd.PtRadialCounts],grid:list[ModelResults],outPath)->list[ModelResults]:
    ptDens=[PtRadialToModelResults(GenRadialPt(data[ptId],0.9,0.45,101),"radial") for ptId in data.keys()]
    with Pool() as pool: results=pool.map(partial(GridLmSearchK,grid=grid,bounds=((-5,-2),(0,1e6))),ptDens)
    pd.to_pickle(results,outPath)

def DkResScatterplot(patientDkResDf):
    sns.scatterplot(patientDkResDf,x='D/k',y='t',size=2,legend=False)
##    plt.xlim(1E-5/1E-4,1E-2/1E-4) #    plt.ylim(0,1E6)
    plt.xscale('log')

def FixedKvsFixedDFigure(fixedKRes:list[ModelResults],fixedDRes:list[ModelResults],outPath):
    fixedDDf=GenPatientDkResDf(fixedDRes)
    fixedKDf=GenPatientDkResDf(fixedKRes)
    DkResScatterplot(fixedKDf)
    plt.gcf().set_size_inches(2.5,2.5)
    plt.xlabel("$\\it{D}$/$\\it{k}$ ($\\it{k}$ fixed)")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(outPath+"fixedKSweep.svg")
    plt.clf()
    DkResScatterplot(fixedDDf)
    plt.gcf().set_size_inches(2.5,2.5)
    plt.xlabel("$\\it{D}$/$\\it{k}$ ($\\it{D}$ fixed)")
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(outPath+"fixedDSweep.svg")
    plt.clf()
    sns.scatterplot(x=fixedDDf['D/k'],y=fixedKDf['D/k'],size=2,legend=False)
#    pcc,pval=pearsonr(fixedDDf['log(D/k)'],fixedKDf['log(D/k)'])
    plt.xlabel("$\\it{D}$/$\\it{k}$ ($\\it{D}$ fixed)")
    plt.ylabel("$\\it{D}$/$\\it{k}$ ($\\it{k}$ fixed)")
    plt.xscale('log')
    plt.yscale('log')
    plt.axline((0,0),(1,1),c='k')
    plt.gcf().set_size_inches(2.5,2.5)
    plt.tight_layout()
    plt.savefig(outPath+"fixedDvsFixedK.svg")
    plt.clf()

def PlotAllIndividualPatientResults(data:dict[str,ppd.PtRadialCounts],grid:list[ModelResults],res:list[ModelResults]):
    ids=list(data.keys())
    for i in range(len(ids)):
        pt=data[ids[i]]
        ptRes=res[i]
        VerifyPatientResults(pt,ptRes,grid,pt.id)

def MethodsFigure(ptPaths,fits:list[ModelResults],ptFeatures,grid:list[ModelResults],outPath,root):
    ptIds=ptFeatures['ID']
    fits=dict([(ptIds[i][-4:],fits[i]) for i in range(len(fits))])
    for iPt,ptId in enumerate(['1456','5417','6193']):
        pt=GenDisplayPt(ptPaths,ptId,ppd.GetLargestCom,root)
        fit=fits[ptId]
        counts=ppd.GenRadialCounts(pt.tumor)
        radial=GenRadialPt(counts,0.9,0.45,101)
        df=GenPtGridDf(radial,grid)
        scaledTumor=np.where(pt.tumor.tumor==6,0.9,pt.tumor)
        scaledTumor=np.where(scaledTumor==5,0.45,scaledTumor)
        scaledTumor=np.where(pt.tumor.mask==-1,-0.2,scaledTumor)

        plt.subplot(4,3,iPt+1)
        PlotTransverse(pt,PlotSlice,PlotComLines)
        if iPt==2:
            cbar=plt.colorbar(fraction=0.05,ticks=[-0.2,0,0.45,0.9])
            cbar.ax.set_yticklabels(["Mask","0:NT","0.45:NE","0.9:EN"])

        plt.subplot(4,3,iPt+4)
        PlotTransverse(pt,PlotSlice,PlotComRadial)
        if iPt==2:
            cbar=plt.colorbar(fraction=0.05,ticks=[-0.2,0,0.45,0.9])
            cbar.ax.set_yticklabels(["Mask","0:NT","0.45:NE","0.9:EN"])

        plt.subplot(4,3,iPt+7)
        DkErrorPlot(df,fit,ptId)
        plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.gca().set_ylabel("min error for all t")

        plt.subplot(4,3,iPt+10)
        plt.plot(radial.tumor,label='data')
        plt.plot(GetDensityDist(fit),label='PDE',linestyle='--')
        plt.xlabel("radius (mm)")
        plt.ylabel("normalized density")
        if iPt==2: plt.legend()


    plt.gcf().set_size_inches(7.5,7.5)
    plt.tight_layout()
    plt.savefig(outPath+"Methods3.svg");plt.clf()

def SupplementalGridHeatmaps(ptPaths,fits:list[ModelResults],ptFeatures,grid:list[ModelResults],outPath,root):
    ptIds=ptFeatures['ID']
    fits=dict([(ptIds[i][-4:],fits[i]) for i in range(len(fits))])
    for iPt,ptId in enumerate(['1456','5417','6193']):
        pt=GenDisplayPt(ptPaths,ptId,ppd.GetLargestCom,root)
        fit=fits[ptId]
        counts=ppd.GenRadialCounts(pt.tumor)
        radial=GenRadialPt(counts,0.9,0.45,101)
        df=GenPtGridDf(radial,grid)
        scaledTumor=np.where(pt.tumor.tumor==6,0.9,pt.tumor)
        scaledTumor=np.where(scaledTumor==5,0.45,scaledTumor)
        scaledTumor=np.where(pt.tumor.mask==-1,-0.2,scaledTumor)

        plt.subplot(1,3,iPt+1)
        if iPt==2: GridSearchHeatmap(df,ptId,True)
        else: GridSearchHeatmap(df,ptId,False)
    plt.gcf().set_size_inches(7.5,2.5)
    plt.tight_layout()
    plt.savefig(outPath+"SupplementalGridHeatmaps.svg");plt.clf()
            

