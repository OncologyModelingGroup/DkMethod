#%% DEFINES
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
from multiprocessing import Pool
from functools import partial

EMPTY,NON,ENH=0,5,6

@dataclass
class PtField:tumor:ndarray;mask:ndarray;com:ndarray;maxTumorRad:float;id:str;tumorDisp:ndarray=None
@dataclass
class PtRadialCounts:counts:ndarray;empty:ndarray;non:ndarray;enh:ndarray;com:ndarray;id:str=None;com:np.ndarray
@dataclass
class PtRadial:tumor:ndarray;enhDensity:float;nonDensity:float;id:str;counts:ndarray
@dataclass
class RawPt:tumor:PtField;T1pre:ndarray;T1post:ndarray;T2:ndarray;FLAIR:ndarray

import FitDkForEachPatient as fitDk

def GenPt(ptPaths,ComCalc,enhVal=ENH,nonVal=NON,maskVal=0,dispEnh=None,dispNon=None,dispNormal=None,dispMask=None)->PtField:
#    print(f"pulling pt {ptPaths['pt']}")
    seg=nib.load(ptPaths['root']+ptPaths['seg']).get_fdata()
    mask=nib.load(ptPaths['root']+ptPaths['mask']).get_fdata()
    mask=np.where(mask>0,1,0)
    seg=np.where((seg==1)|(seg==4),enhVal,seg)
    seg=np.where(seg==2,nonVal,seg)
    mask=np.where(mask>0,1,-1)
    seg=np.where(mask>0,seg,maskVal)
    if dispEnh!=None: 
        disp=np.where(mask>0,dispNormal,dispMask)
        disp=np.where(seg==enhVal,dispEnh,disp)
        disp=np.where(seg==nonVal,dispNon,disp)
        pt=PtField(seg,mask,np.array([-1,-1,-1]),-1,ptPaths['pt'],disp)
    else: pt=PtField(seg,mask,np.array([-1,-1,-1]),-1,ptPaths['pt'])
    pt=replace(pt,com=ComCalc(pt.tumor,pt.mask))
    return pt

def GenRawPt(ptPaths,ComCalc)->RawPt:
    flair=None;t2=None;t1Pre=None;t1Post=None
    for file in os.listdir(ptPaths['root']):
        if "_flair" in file: flair=nib.load(ptPaths['root']+file).get_fdata()
        elif "_t2" in file: t2=nib.load(ptPaths['root']+file).get_fdata()
        elif "_t1ce" in file: t1Post=nib.load(ptPaths['root']+file).get_fdata()
        elif "_t1Gd" in file: t1Post=nib.load(ptPaths['root']+file).get_fdata()
        elif "_t1" in file: t1Pre=nib.load(ptPaths['root']+file).get_fdata()
    pt=GenPt(ptPaths,ComCalc,rootDir,ENH,NON,EMPTY,0.9,0.45,0,-0.2)
    return RawPt(pt,t1Pre,t1Post,t2,flair)

@njit
def GetContiguousRegion(field:ndarray,inside,iRegion,x,y,z):
    v=np.zeros((field.size,3),dtype=np.int32)
    v[0,0]=x; v[0,1]=y; v[0,2]=z
    field[x,y,z]=iRegion
    i=0
    iEnd=1
    shape=field.shape
    while(i<iEnd):
        if v[i,0]+1<shape[0] and field[v[i,0]+1,v[i,1],v[i,2]]==inside: 
            field[v[i,0]+1,v[i,1],v[i,2]]=iRegion 
            v[iEnd,0]=v[i,0]+1; v[iEnd,1]=v[i,1]; v[iEnd,2]=v[i,2]
            iEnd+=1
        if v[i,0]-1>=0 and field[v[i,0]-1,v[i,1],v[i,2]]==inside: 
            field[v[i,0]-1,v[i,1],v[i,2]]=iRegion 
            v[iEnd,0]=v[i,0]-1; v[iEnd,1]=v[i,1]; v[iEnd,2]=v[i,2]
            iEnd+=1

        if v[i,1]+1<shape[1] and field[v[i,0],v[i,1]+1,v[i,2]]==inside: 
            field[v[i,0],v[i,1]+1,v[i,2]]=iRegion 
            v[iEnd,0]=v[i,0]; v[iEnd,1]=v[i,1]+1; v[iEnd,2]=v[i,2]
            iEnd+=1
        if v[i,1]-1>=0 and field[v[i,0],v[i,1]-1,v[i,2]]==inside: 
            field[v[i,0],v[i,1]-1,v[i,2]]=iRegion 
            v[iEnd,0]=v[i,0]; v[iEnd,1]=v[i,1]-1; v[iEnd,2]=v[i,2]
            iEnd+=1

        if v[i,2]+1<shape[2] and field[v[i,0],v[i,1],v[i,2]+1]==inside: 
            field[v[i,0],v[i,1],v[i,2]+1]=iRegion 
            v[iEnd,0]=v[i,0]; v[iEnd,1]=v[i,1]; v[iEnd,2]=v[i,2]+1
            iEnd+=1
        if v[i,2]-1>=0 and field[v[i,0],v[i,1],v[i,2]-1]==inside: 
            field[v[i,0],v[i,1],v[i,2]-1]=iRegion 
            v[iEnd,0]=v[i,0]; v[iEnd,1]=v[i,1]; v[iEnd,2]=v[i,2]-1
            iEnd+=1
        i+=1

@njit
def LabelContiguousRegions(field:ndarray,inside,iStart):
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            for z in range(field.shape[2]):
                if field[x,y,z]==inside:
                    GetContiguousRegion(field,inside,iStart,x,y,z)
                    iStart+=1
    return field

def GetLargestMassIndex(labeled,iStart,returnCount=False):
    iMasses,counts=np.unique(labeled,return_counts=True)
    iFirstMass=np.argmax(iMasses>=iStart)
    iMasses=iMasses[iFirstMass:]
    counts=counts[iFirstMass:]
    iMaxCount=counts.argmax()
    if returnCount:return counts.max()
    return iMasses[iMaxCount]

def GetLongestDiamCom(field,mask,returnDiamPoints=False):
    labeled=LabelContiguousRegions(field.copy(),ENH,7)
    idx=GetLargestMassIndex(labeled,iStart=7)
    edge=GetEdge(labeled,idx)
    diam=FindLongestDiam(edge)
    if returnDiamPoints:return np.array([diam[0],diam[1],(diam[0]+diam[1])/2])
    return (diam[0]+diam[1])/2

def GetLongest2DiamCom(field,mask,returnDiamPoints=False):
    labeled=LabelContiguousRegions(field.copy(),ENH,7)
    idx=GetLargestMassIndex(labeled,iStart=7)
    edge=GetEdge(labeled,idx)
    p1,q1=FindLongestDiam(edge)
    out=FindLongestPerpIntersectionPoint(edge,p1,q1,5*(np.pi/180),1)
    if returnDiamPoints:return np.stack((p1,q1,out[0],out[1],out[2]))
    return out[2]


def GetLargestCom(field,mask):
    labeled=LabelContiguousRegions(field.copy(),ENH,7)
    idx=GetLargestMassIndex(labeled,iStart=7)
    return GetComEnh(labeled,idx)

def GetLargestVol(field,inside,iStart):
    labeled=LabelContiguousRegions(field.copy(),ENH,7)
    return GetLargestMassIndex(labeled,iStart,True)

def GetComComposite(field,mask,radiusFracCutoff):
    com=GetLargestCom(field,mask)
    dist=GetDistToMask(mask,com)
    vol=GetLargestVol(field,ENH,7)
    radius=((vol*3)/(4*np.pi))**(1/3)
    if dist<radiusFracCutoff*radius: 
        com=GetLongestDiamCom(field,mask)
    return com

def GetComComposite2Diam(field,mask,radiusFracCutoff):
    com=GetLargestCom(field,mask)
    dist=GetDistToMask(mask,com)
    vol=GetLargestVol(field,ENH,7)
    radius=((vol*3)/(4*np.pi))**(1/3)
    if dist<radiusFracCutoff*radius: 
        com=GetLongest2DiamCom(field,mask)
    return com


def GetComEnh(field:ndarray,EnhLabel):
    enh=np.where(field==EnhLabel,1,0)
    com=GetComCont(enh)
    return com

@njit
def GetComCont(tumor):
    totalMass=0
    centerZ=0
    centerY=0
    centerX=0
    for z in range(tumor.shape[0]):
        for y in range(tumor.shape[1]):
            for x in range(tumor.shape[2]):
                dens=tumor[z,y,x]
                if dens>0:
                    totalMass+=dens
                    centerZ+=z*dens
                    centerY+=y*dens
                    centerX+=x*dens
    out=np.array([centerZ/totalMass,centerY/totalMass,centerX/totalMass])
    return out

@njit
def GetMaxTumorRad(t,com):
    maxDistSq=0
    for x in range(t.shape[0]):
        for y in range(t.shape[1]):
            for z in range(t.shape[2]):
                if t[x,y,z]>0:
                    distSq=((x)-com[0])**2+((y)-com[1])**2+((z)-com[2])**2
                    if distSq>maxDistSq:
                        maxDistSq=distSq
    return np.sqrt(maxDistSq)


def GetRadialDist100(t,mask,com):
    return GetRadialDist(t,mask,com,99.5)

@njit
def GetRadialDist(t,mask,com,maxRad):
    I_COUNT,I_EMPTY,I_NON,I_ENH=0,1,2,3
    sums=np.zeros((4,int(maxRad+1.5)))
    for x in range(t.shape[0]):
        for y in range(t.shape[1]):
            for z in range(t.shape[2]):
                if mask is None or mask[x,y,z]>=0:
                    dist=np.sqrt((x-com[0])**2+(y-com[1])**2+(z-com[2])**2)
                    iDist=int(dist+0.5)
                    if iDist<sums.shape[1]:
                        sums[I_COUNT,iDist]+=1
                        voxVal=t[x,y,z]
                        if voxVal==EMPTY:sums[I_EMPTY,iDist]+=1
                        elif voxVal==NON:sums[I_NON,iDist]+=1
                        elif voxVal==ENH:sums[I_ENH,iDist]+=1
    if sums[I_COUNT,0]==0:
        voxVal=t[int(com[0]+0.5),int(com[1]+0.5),int(com[2]+0.5)]
        sums[I_COUNT,0]=1
        if voxVal==0:sums[I_EMPTY,0]+=1
        elif voxVal==NON:sums[I_NON,0]+=1
        elif voxVal==ENH:sums[I_ENH,0]+=1
    return sums

@njit
def FindClosestDist(maskEdge,com):
    minDistSq=np.finfo(np.float32).max
    for i in range(len(maskEdge)):
        distSq=(com[0]-maskEdge[i][0])**2+(com[1]-maskEdge[i][1])**2+(com[2]-maskEdge[i][2])**2
        if minDistSq>distSq:
            minDistSq=distSq
    return np.sqrt(minDistSq)

def GetDistToMask(mask,com):
    labeled=LabelContiguousRegions(np.copy(mask),-1,2)
    idx=GetLargestMassIndex(labeled,2)
    maskEdge=GetEdge(labeled,idx)
    return FindClosestDist(maskEdge,com)


@njit
def GetEdge(vol:np.ndarray,onEdgeVal):
    i=0
    outVoxels=np.zeros((vol.size,3))
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            for z in range(vol.shape[2]):
                if vol[x,y,z]==onEdgeVal and ((not x+1 >= vol.shape[0] and vol[x+1,y,z]!=onEdgeVal) or (not x-1 < 0 and vol[x-1,y,z]!=onEdgeVal) or (not y+1 >= vol.shape[1] and vol[x,y+1,z]!=onEdgeVal) or (not y-1 < 0 and vol[x,y-1,z]!=onEdgeVal) or (not z+1 >= vol.shape[2] and vol[x,y,z+1]!=onEdgeVal) or (not z-1 < 0 and vol[x,y,z-1]!=onEdgeVal)):
                    outVoxels[i,0]=x;outVoxels[i,1]=y;outVoxels[i,2]=z
                    i+=1
    return outVoxels[0:i]

@njit
def FindLongestDiam(edgeVoxels:np.ndarray):
    out=np.zeros((2,3))
    longestDistSq=0
    for i in range(len(edgeVoxels)):
        for j in range(i+1,len(edgeVoxels)):
            distSq=(edgeVoxels[i,0]-edgeVoxels[j,0])**2+(edgeVoxels[i,1]-edgeVoxels[j,1])**2+(edgeVoxels[i,2]-edgeVoxels[j,2])**2
            if distSq>longestDistSq:
                longestDistSq=distSq
                out[0]=edgeVoxels[i]
                out[1]=edgeVoxels[j]
    return out

@njit
def FindLongestPerpIntersectionPoint(edgeVoxels:np.ndarray,p1,q1,angleThresh,distThresh):
    intersectionPt=np.zeros(3)
    longestDistSq=0
    longestPoints=None
    for i in range(len(edgeVoxels)):
        for j in range(i+1,len(edgeVoxels)):
            distSq=(edgeVoxels[i,0]-edgeVoxels[j,0])**2+(edgeVoxels[i,1]-edgeVoxels[j,1])**2+(edgeVoxels[i,2]-edgeVoxels[j,2])**2
            if distSq>longestDistSq and np.abs(GetAngle(p1,q1,edgeVoxels[i],edgeVoxels[j])-np.pi/2)<angleThresh:
                shortestDist,pOut,_=ShortestDist(p1,q1,edgeVoxels[i],edgeVoxels[j])
                if shortestDist<distThresh:
                    longestDistSq=distSq
                    intersectionPt=pOut
                    longestPoints=[edgeVoxels[i],edgeVoxels[j]]
    return np.stack((longestPoints[0],longestPoints[1],intersectionPt))

def FindSegPath(files):
    for file in files:
        if "_ManuallyCorrected." in file: return file
    for file in files:
        if "_GlistrBoost." in file: return file

def FindSegPath2(files):
    for file in files:
        if "_seg." in file: return file

def FindMaskPath(files):
    for file in files:
        if "_flair" in file: return file
    for file in files:
        if "_t2" in file: return file
    for file in files:
        if "_t1ce" in file: return file
    for file in files:
        #if "_t1" in file: return file[:12]+"/"+file
        if "_t1" in file: return file

def FindMaskPath2(files):
    for file in files:
        if "_flair" in file: return file
    for file in files:
        if "_t2" in file: return file
    for file in files:
        if "_t1Gd" in file: return file
    for file in files:
        if "_t1" in file: return file


def GenBratsPaths(bratsPath,outPath,FindSegPath=FindSegPath,FindMaskPath=FindMaskPath,idBegin=8,idEnd=12):
    paths={}
    for root,dirs,files in os.walk(bratsPath):
        segPath=FindSegPath(files)
        if segPath!=None:
            ptId=segPath[idBegin:idEnd]
            maskPath=FindMaskPath(files)
            paths[ptId]={"seg":segPath,"mask":maskPath,"pt":ptId,"root":root+"/"}
    pd.to_pickle(paths,outPath)

def GenRadialCounts(pt:PtField)->PtRadialCounts:
    sums=GetRadialDist100(pt.tumor,pt.mask,pt.com)
    return PtRadialCounts(sums[0],sums[1],sums[2],sums[3],pt.com,pt.id)


def ProcessPt(ptId,paths,ComCalc)->PtRadialCounts:
        print(ptId)
        pt=GenPt(paths[ptId],ComCalc)
        radial=GenRadialCounts(pt)
        return radial

def ProcessPts(paths,outPath,ComCalc)->list[PtRadialCounts]:
    with Pool() as pool:results=pool.map(partial(ProcessPt,paths=paths,ComCalc=ComCalc),paths.keys())
    pd.to_pickle(dict(zip(paths.keys(),results)),outPath)
    return results

@njit
def GetAngle(p1,q1,p2,q2):
    v1=q1-p1
    v2=q2-p2
    dotted=np.dot(v1,v2)
    norm=np.linalg.norm(v1)*np.linalg.norm(v2)
    frac=dotted/norm
    if frac<-1:frac=-1
    elif frac>1:frac=1
    return np.arccos(frac)

@njit
def ShortestDist(p1, q1, p2, q2):
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a * c - b * b
    if denom == 0.0:
        # Segments are parallel
        s = 0.0
        t = d / b if b != 0 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        if s<0.0:s=0.0
        elif s>1.0:s=1.0
        if t<0.0:t=0.0
        elif t>1.0:t=1.0
        if s<0:s=0
        elif s>1:s=1
        if t<0:t=0
        elif t>1:t=1

    closest_point_on_seg1 = p1 + s * u
    closest_point_on_seg2 = p2 + t * v
    distance = np.linalg.norm(closest_point_on_seg1 - closest_point_on_seg2)

    return (distance, closest_point_on_seg1, closest_point_on_seg2)

def GenDispImg(img,cmap):
    plt.imshow(img.T,cmap=cmap)
    plt.gca().set_axis_off()


def PlotExampleScans(id,paths,rootPath,outDir):
    rawPt=GenRawPt(paths[id],GetLargestCom,rootPath)
    iShow=rawPt.tumor.com[2]
    if rawPt.FLAIR is not None:
        GenDispImg(fitDk.GetTransverseSlice(rawPt.FLAIR,z=iShow),'Greys_r')
        plt.savefig(outDir+"FLAIR.png",bbox_inches='tight',pad_inches=0.0)
    if rawPt.T1pre is not None:
        GenDispImg(fitDk.GetTransverseSlice(rawPt.T1pre,z=iShow),'Greys_r')
        plt.savefig(outDir+"T1pre.png",bbox_inches='tight',pad_inches=0.0)
    if rawPt.T1post is not None:
        GenDispImg(fitDk.GetTransverseSlice(rawPt.T1post,z=iShow),'Greys_r')
        plt.savefig(outDir+"T1post.png",bbox_inches='tight',pad_inches=0.0)
    if rawPt.T2 is not None:
        GenDispImg(fitDk.GetTransverseSlice(rawPt.T2,z=iShow),'Greys_r')
        plt.savefig(outDir+"T2.png",bbox_inches='tight',pad_inches=0.0)
    GenDispImg(fitDk.GetTransverseSlice(rawPt.tumor.tumorDisp,z=iShow),'viridis')
    plt.savefig(outDir+"Segmentation.png",bbox_inches='tight',pad_inches=0.0)
