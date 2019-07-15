# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:15:44 2019

@author: RicardoSan and Gonzalo Paz

Fast Marching 

"""

import numpy as np
from numpy import array
from numpy import dot
import math
import operator
import bisect


def updateNode(nodeTarget, costMap, Tmap, nbT, nbNodes, closedMap):

    for i in range(1,6+1):
        if i == 1:
            nodeChild = np.add(nodeTarget,[0,0,-1])
        elif i == 2:
            nodeChild = np.add(nodeTarget,[0,0,1])
        elif i == 3:
            nodeChild = np.add(nodeTarget,[-1,0,0])
        elif i == 4:
            nodeChild = np.add(nodeTarget,[1,0,0])
        elif i == 5:
            nodeChild = np.add(nodeTarget,[0,1,0])
        elif i == 6:
            nodeChild = np.add(nodeTarget,[0,-1,0])
                
        if closedMap[nodeChild[1],nodeChild[0],nodeChild[2]] == 0:
            C = costMap[nodeChild[1],nodeChild[0],nodeChild[2]]
            Tx1 = Tmap[nodeChild[1],nodeChild[0]-1,nodeChild[2]]
            Tx2 = Tmap[nodeChild[1],nodeChild[0]+1,nodeChild[2]]
            Ty1 = Tmap[nodeChild[1]-1,nodeChild[0],nodeChild[2]]
            Ty2 = Tmap[nodeChild[1]+1,nodeChild[0],nodeChild[2]]
            Tz1 = Tmap[nodeChild[1],nodeChild[0],nodeChild[2]-1]
            Tz2 = Tmap[nodeChild[1],nodeChild[0],nodeChild[2]+1]
            
            if Tx1<Tx2:
                Tx = Tx1
            else:
                Tx = Tx2
                
            if Ty1<Ty2:
                Ty = Ty1
            else:
                Ty = Ty2
                
            if Tz1<Tz2:
                Tz = Tz1
            else:
                Tz = Tz2
                
            Tarray = [Tx,Ty,Tz]
            Tr = np.inf
            
            while Tr == np.inf:
                n = len(Tarray)
                Tmax = max(Tarray)
                
                sumT = 0
                for a in range(0,n):
                    sumT = sumT + (Tmax-Tarray[a])**2
                    
                if (C**2) > sumT:
                    Tr = ((sumlist(Tarray))+np.sqrt(n*C**2+(sumlist(Tarray))**2-n*(sumlist(array(Tarray)**2))))/n
                    
                Tarray.remove(Tmax)
            
            T = Tr
            
            if np.isinf(Tmap[nodeChild[1],nodeChild[0],nodeChild[2]]):
                nIndex = bisect.bisect_left(nbT,T)
                nbT.insert(nIndex,T)
                nbNodes.insert(nIndex, nodeChild)
                Tmap[nodeChild[1],nodeChild[0],nodeChild[2]] = T
#                Tmap[nodeChild[1],nodeChild[0],nodeChild[2]] = T
#                nodeRes = [nodeChild[0], nodeChild[1],nodeChild[2], T]
#                narrowBand.append(nodeRes)
            else:
                if T < Tmap[nodeChild[1],nodeChild[0],nodeChild[2]]:
                    tempT = Tmap[nodeChild[1],nodeChild[0],nodeChild[2]]
                    nIndex = bisect.bisect_left(nbT,tempT)
                    nIndex = next(x for x,n in enumerate(nbNodes[nIndex-1:],nIndex) if np.array_equal(nodeChild,nbNodes[x]))
                    del nbT[nIndex]
                    del nbNodes[nIndex]
                    nIndex = bisect.bisect_left(nbT,T)
                    nbT.insert(nIndex,T)
                    nbNodes.insert(nIndex, nodeChild)
                    Tmap[nodeChild[1],nodeChild[0],nodeChild[2]] = T
#                    narrowBand.remove([nodeChild[0], nodeChild[1],nodeChild[2], Tmap[nodeChild[1],nodeChild[0],nodeChild[2]]])
#                    Tmap[nodeChild[1],nodeChild[0],nodeChild[2]] = T
#                    nodeRes = [nodeChild[0], nodeChild[1], T]
#                    nodeRes = [nodeChild[0], nodeChild[1],nodeChild[2], T]
#                    narrowBand.append(nodeRes)                
    return Tmap, nbT, nbNodes

def sumlist(listNum):
   if len(listNum) == 1:
        return listNum[0]
   else:
        return listNum[0] + sumlist(listNum[1:])
    
def getMinNB(nbT,nbNodes):
#    #Tvalues = [i[0] for i in narrowBand]
#    nodeMin = min(narrowBand, key=operator.itemgetter(3))
#    narrowBand.remove(nodeMin)
#    #narrowBand = [i for i in narrowBand if not np.array_equal(i[0],nodeMin)]
#    nodeTarget = nodeMin[0:3]
##    nodeMin = narrowBand.pop(0)
##    nodeTarget = nodeMin[1]
#    return nodeTarget, narrowBand
    nodeTarget = nbNodes.pop(0)
    del nbT[0]
#    nodeMin = min(narrowBand, key=operator.itemgetter(2))
#    narrowBand.remove(nodeMin)
    #narrowBand = [i for i in narrowBand if not np.array_equal(i[0],nodeMin)]
    return nodeTarget, nbT, nbNodes


def computeTmap(costMap,goal,start):
    closedMap = np.zeros_like(costMap)
    closedMap[np.where(costMap == np.inf)] = 1
    Tmap = np.ones_like(costMap)*np.inf
    nbT = []
    nbNodes = []
    Tmap[goal[1],goal[0],goal[2]] = 0
    closedMap[goal[1],goal[0],goal[2]] = 1
    [Tmap, nbT, nbNodes] = updateNode(goal, costMap, Tmap, nbT, nbNodes, closedMap)
    #iter = 1
    #size = np.size(costMap)
    while len(nbT) > 0:
        nodeTarget, nbT, nbNodes = getMinNB(nbT, nbNodes)
        closedMap[nodeTarget[1],nodeTarget[0],nodeTarget[2]] = 1
        Tmap, nbT, nbNodes = updateNode(nodeTarget, costMap, Tmap, nbT, nbNodes, closedMap)
        if np.array_equal(nodeTarget,start):
            break
        #iter = iter + 1
        #print('Completed: ' + "{0:.2f}".format(100*iter/size) + ' %')
    return Tmap

# =============================================================================
# def biComputeTmap(costMap,goal,start):
#     # Define nodeTargets
#     nodeTargetG = [goal[0],goal[1],goal[2]]
#     nodeTargetS = [start[0],start[1],start[2]]
#     # Both closedMaps are created
#     closedMapG = np.zeros_like(costMap)
#     closedMapG[nodeTargetG[1],nodeTargetG[0],nodeTargetG[2]] = 1
#     closedMapG[np.isinf(costMap)] = 1
#     closedMapS = np.zeros_like(costMap)
#     closedMapS[nodeTargetS[1],nodeTargetS[0],nodeTargetS[2]] = 1
#     closedMapS[np.isinf(costMap)] = 1
#     # Narrow Bands are initialized
#     narrowBandG = []
#     narrowBandS = []
#     # Both Tmaps are initialized
#     TmapG = np.ones_like(costMap)*np.inf
#     TmapG[nodeTargetG[1],nodeTargetG[0],nodeTargetG[2]] = 0
#     [TmapG, narrowBandG] = updateNode(nodeTargetG, costMap, TmapG, narrowBandG, closedMapG)
#     TmapS = np.ones_like(costMap)*np.inf
#     TmapS[nodeTargetS[1],nodeTargetS[0],nodeTargetS[2]] = 0
#     [TmapS, narrowBandS] = updateNode(nodeTargetS, costMap, TmapS, narrowBandS, closedMapS)
# 
#     #iter = 1
#     #size = np.size(costMap)
#     
#     while narrowBandG or narrowBandS:
#         if narrowBandG:
#             nodeTargetG, narrowBandG = getMinNB(narrowBandG)
#             closedMapG[nodeTargetG[1],nodeTargetG[0],nodeTargetG[2]] = 1
#             TmapG, narrowBandG = updateNode(nodeTargetG, costMap, TmapG, narrowBandG, closedMapG)
#         if narrowBandS:
#             nodeTargetS, narrowBandS = getMinNB(narrowBandS)
#             closedMapS[nodeTargetS[1],nodeTargetS[0],nodeTargetS[2]] = 1
#             TmapS, narrowBandS = updateNode(nodeTargetS, costMap, TmapS, narrowBandS, closedMapS)
#         if closedMapS[nodeTargetG[1],nodeTargetG[0],nodeTargetG[2]] == 1:
#             nodeJoin = nodeTargetG
#             break
#         if closedMapG[nodeTargetS[1],nodeTargetS[0],nodeTargetS[2]] == 1:
#             nodeJoin = nodeTargetS
#             break
#         #iter = iter + 2
#         #print('Completed: ' + "{0:.2f}".format(100*iter/size) + ' %')
#         
#     TmapG[np.isnan(TmapG)] = np.inf
#     TmapS[np.isnan(TmapS)] = np.inf
#     nodeJoin = np.uint32(nodeJoin)
#     return TmapG, TmapS, nodeJoin
# =============================================================================


def getPathGDM(totalCostMap,initWaypoint,endWaypoint,tau):
    
    G2, G1, G3 = np.gradient(totalCostMap)
    
    gamma = np.empty([0,3])
    gamma = np.vstack((gamma,initWaypoint.T))

    nearN = np.zeros(3)
    
    for k in range(0,int(round(15000/tau))):
        dx = interpolatePoint(gamma[-1,:],G1)
        dy = interpolatePoint(gamma[-1,:],G2)
        dz = interpolatePoint(gamma[-1,:],G3)
        
        if (np.isnan(dx)) or (np.isnan(dy) or np.isnan(dz)): #or (np.isnan(interpolatePoint(gamma[-1,:]-[dx,dy],totalCostMap))):
            nearN[0] = int(round(gamma[-1,0]))
            nearN[1] = int(round(gamma[-1,1]))
            nearN[2] = int(round(gamma[-1,2]))
            nearN = np.uint32(nearN)
            while np.isinf(totalCostMap[nearN[1],nearN[0],nearN[2]]):
                gamma = gamma[:-1]
                nearN[0] = int(round(gamma[-1,0]))
                nearN[1] = int(round(gamma[-1,1]))
                nearN[2] = int(round(gamma[-1,2]))
                nearN = np.uint32(nearN)
            if gamma.size > 0:
                while (np.linalg.norm(gamma[-1,:]-nearN)) < 1:
                    gamma = gamma[:-1]
                    if gamma.size == 0:
                        break
                    
            gamma = np.vstack((gamma,array(nearN).T))
            currentT = totalCostMap[nearN[1],nearN[0],nearN[2]]
            for i in range(1,6+1):
                if i == 1:
                    nodeChild = np.int32(nearN + [0,-1,0])
                elif i == 2:
                    nodeChild = np.int32(nearN + [0,1,0])
                elif i == 3:
                    nodeChild = np.int32(nearN + [-1,0,0])
                elif i == 4:
                    nodeChild = np.int32(nearN + [1,0,0])
                elif i == 5:
                    nodeChild = np.int32(nearN + [0,0,-1])
                elif i == 6:
                    nodeChild = np.int32(nearN + [0,0,1])
                elif i == 7:
                    nodeChild = np.int32(nearN + [-1,1,0])
                elif i == 8:
                    nodeChild = np.int32(nearN + [1,-1,0])
            
                if totalCostMap[nodeChild[1],nodeChild[0],nodeChild[2]] < currentT:
                    currentT = totalCostMap[nodeChild[1],nodeChild[0],nodeChild[2]]
                    dx = (nearN[0]-nodeChild[0])/tau
                    dy = (nearN[1]-nodeChild[1])/tau
                    dz = (nearN[2]-nodeChild[2])/tau
        
        norm = math.sqrt(dx**2+dy**2+dz**2)
        if norm < 0.01:
            dnx = dx/norm
            dny = dy/norm
            dnz = dz/norm
            add = gamma[-1,:] - dot(tau,[dnx,dny,dnz])
            gamma = np.vstack((gamma,array(add).T))
        else:
            add = gamma[-1,:] - dot(tau,[dx,dy,dz])
            gamma = np.vstack((gamma,array(add).T))
        
        if np.linalg.norm(gamma[-1,:]-endWaypoint)<1.5:
            break
    
    gamma = np.vstack((gamma,array(endWaypoint).T))

    return gamma            
    
    

def interpolatePoint(point,mapI):
    i = np.uint32(np.fix(point[0]))
    j = np.uint32(np.fix(point[1]))
    k = np.uint32(np.fix(point[2]))
    a = point[0] - i
    b = point[1] - j
    c = point[2] - k
    
    a0 = mapI[j,i,k]
    a1 = mapI[j,i+1,k] - mapI[j,i,k]
    a2 = mapI[j+1,i,k] - mapI[j,i,k]
    a3 = mapI[j,i,k+1] - mapI[j,i,k]
    a4 = mapI[j+1,i+1,k] + mapI[j,i,k] - mapI[j,i+1,k] - mapI[j+1,i,k]
    a5 = mapI[j,i+1,k+1] + mapI[j,i,k] - mapI[j,i+1,k] - mapI[j,i,k+1]
    a6 = mapI[j+1,i,k+1] + mapI[j,i,k] - mapI[j+1,i,k] - mapI[j,i,k+1]
    a7 = mapI[j+1,i+1,k+1] + mapI[j,i,k] - mapI[j+1,i,k] - mapI[j,i,k+1] - mapI[j,i+1,k]
    
    m,n,o = np.uint32(mapI.shape)
    
    if i == n:
        if j == m:
            if k == o:
                I = mapI[j,i,k]
            else:
                I = c*mapI[j,i,k+1] + (1-c)*mapI[j,i,k]
        else:
            if k == o:
                I = b*mapI[j+1,i,k] + (1-b)*mapI[j,i,k]
            else:
                I = a0 + a1*a + a2*b + a3*c + a4*a*b + a5*a*c + a6*b*c + a7*a*b*c
    else:
        if j == m:
            if k == o:
                I = a*mapI[j,i+1,k] + (1-a)*mapI[j,i,k]
            else:
                I = a0 + a1*a + a2*b + a3*c + a4*a*b + a5*a*c + a6*b*c + a7*a*b*c
        else:
            I = a0 + a1*a + a2*b + a3*c + a4*a*b + a5*a*c + a6*b*c + a7*a*b*c
                    
    return I