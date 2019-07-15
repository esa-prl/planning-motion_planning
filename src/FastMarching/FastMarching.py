# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:15:44 2019

@author: RicardoSan and Gonzalo Paz

Fast Marching

"""

import numpy as np
import operator
import math
import bisect
from time import time

def getEikonal(Thor,Tver,cost):
    if np.isinf(Thor):
        if np.isinf(Tver):
            return np.inf
        else:
            return Tver+cost
    if np.isinf(Tver):
        return Thor+cost
    else:
        if cost < np.abs(Thor-Tver):
            return np.minimum(Thor,Tver)+cost
        else:
            return .5*(Thor+Tver+math.sqrt(2*np.power(cost,2) - np.power(Thor-Tver,2)))

def getNeighbours(nodeTarget, closedMap):
    nList = []
    nTij = nodeTarget[0:2]
    neighbours = [[-1,0],
                  [1,0],
                  [0,-1],
                  [0,1]]
    for ni in neighbours:
        nN = np.add(nTij,ni)
        if closedMap[nN[1], nN[0]] == 0:
            nList.append(nN)
    return nList

def updateNode(nodeTarget, costMap, Tmap, nbT, nbNodes, closedMap):
    #nList = getNeighbours(nodeTarget, closedMap)
    for i in range(1,4+1):
        if i == 1:
            nodeChild = np.add(nodeTarget,[0,-1])
        elif i == 2:
            nodeChild = np.add(nodeTarget,[0,1])
        elif i == 3:
            nodeChild = np.add(nodeTarget,[-1,0])
        elif i == 4:
            nodeChild = np.add(nodeTarget,[1,0])

        if closedMap[nodeChild[1],nodeChild[0]] == 0:
            Thor1 = Tmap[nodeChild[1],nodeChild[0]+1]
            Thor2 = Tmap[nodeChild[1],nodeChild[0]-1]
            Thor = np.minimum(Thor1,Thor2)
            Tver1 = Tmap[nodeChild[1]+1,nodeChild[0]]
            Tver2 = Tmap[nodeChild[1]-1,nodeChild[0]]
            Tver = np.minimum(Tver1,Tver2)
            T = getEikonal(Thor,Tver,costMap[nodeChild[1],nodeChild[0]])
            if np.isinf(Tmap[nodeChild[1],nodeChild[0]]):
                nIndex = bisect.bisect_left(nbT,T)
                nbT.insert(nIndex,T)
                nbNodes.insert(nIndex, nodeChild)
                Tmap[nodeChild[1],nodeChild[0]] = T
            else:
                if T < Tmap[nodeChild[1],nodeChild[0]]:
                    tempT = Tmap[nodeChild[1],nodeChild[0]]
                    nIndex = bisect.bisect_left(nbT,tempT)
                    nIndex = next(x for x,n in enumerate(nbNodes[nIndex-1:],nIndex) if np.array_equal(nodeChild,nbNodes[x]))
                    del nbT[nIndex]
                    del nbNodes[nIndex]
                    nIndex = bisect.bisect_left(nbT,T)
                    nbT.insert(nIndex,T)
                    nbNodes.insert(nIndex, nodeChild)
                    Tmap[nodeChild[1],nodeChild[0]] = T
    return Tmap, nbT, nbNodes

def getMinNB(nbT,nbNodes):
    #Tvalues = [i[0] for i in narrowBand]
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
    Tmap[goal[1],goal[0]] = 0
    closedMap[goal[1],goal[0]] = 1
    nodeTarget = [goal[0],goal[1]]
    [Tmap, nbT, nbNodes] = updateNode(nodeTarget, costMap, Tmap, nbT, nbNodes, closedMap)
    #iter = 1
    #size = np.size(costMap)
    while nbT:
        nodeTarget, nbT, nbNodes = getMinNB(nbT, nbNodes)
        closedMap[nodeTarget[1],nodeTarget[0]] = 1
        Tmap, narrowBand = updateNode(nodeTarget, costMap, Tmap, nbT, nbNodes, closedMap)
        if np.array_equal(nodeTarget,start):
            break
        #iter = iter + 1
        #print('Completed: ' + "{0:.2f}".format(100*iter/size) + ' %')
    return Tmap

def biComputeTmap(costMap,goal,start):
    # Define nodeTargets
    nodeTargetG = [goal[0],goal[1]]
    nodeTargetS = [start[0],start[1]]
    # Both closedMaps are created
    closedMapG = np.zeros_like(costMap)
    closedMapG[nodeTargetG[1],nodeTargetG[0]] = 1
    closedMapG[np.isinf(costMap)] = 1
    closedMapS = np.zeros_like(costMap)
    closedMapS[nodeTargetS[1],nodeTargetS[0]] = 1
    closedMapS[np.isinf(costMap)] = 1
    # Narrow Bands are initialized
    nbTG = []
    nbNodesG = []
    nbTS = []
    nbNodesS = []
    # Both Tmaps are initialized
    TmapG = np.ones_like(costMap)*np.inf
    TmapG[nodeTargetG[1],nodeTargetG[0]] = 0
    [TmapG, nbTG, nbNodesG] = updateNode(nodeTargetG, costMap, TmapG, nbTG, nbNodesG, closedMapG)
    TmapS = np.ones_like(costMap)*np.inf
    TmapS[nodeTargetS[1],nodeTargetS[0]] = 0
    [TmapS, nbTS, nbNodesS] = updateNode(nodeTargetS, costMap, TmapS, nbTS, nbNodesS, closedMapS)

    #iter = 1
    #size = np.size(costMap)

    while nbTG or nbTS:
        if nbTG:
            nodeTargetG, nbTG, nbNodesG = getMinNB(nbTG, nbNodesG)
            closedMapG[nodeTargetG[1],nodeTargetG[0]] = 1
            TmapG, nbTG, nbNodesG = updateNode(nodeTargetG, costMap, TmapG, nbTG, nbNodesG, closedMapG)
        if nbTS:
            nodeTargetS, nbTS, nbNodesS = getMinNB(nbTS, nbNodesS)
            closedMapS[nodeTargetS[1],nodeTargetS[0]] = 1
            TmapS, nbTS, nbNodesS = updateNode(nodeTargetS, costMap, TmapS, nbTS, nbNodesS, closedMapS)
        if closedMapS[nodeTargetG[1],nodeTargetG[0]] == 1:
            nodeJoin = nodeTargetG
            break
        if closedMapG[nodeTargetS[1],nodeTargetS[0]] == 1:
            nodeJoin = nodeTargetS
            break
        #iter = iter + 2
        #print('Completed: ' + "{0:.2f}".format(100*iter/size) + ' %')

    TmapG[np.isnan(TmapG)] = np.inf
    TmapS[np.isnan(TmapS)] = np.inf
    nodeJoin = np.uint32(nodeJoin)
    return TmapG, TmapS, nodeJoin

def getPathGDM(totalCostMap,initWaypoint,endWaypoint,tau):

    #G2, G1 = np.gradient(totalCostMap)

    gamma = np.empty([0,2])
    gamma = np.vstack((gamma,initWaypoint.T))

    nearN = [0,0]

    for k in range(0,round(15000/tau)):
        G1,G2 = computeGradient(totalCostMap,gamma[-1,:])
        dx = interpolatePoint(gamma[-1,:],G1)
        dy = interpolatePoint(gamma[-1,:],G2)

        if (np.isnan(dx)) or (np.isnan(dy)):
            try:
                nearN[0] = int(round(gamma[-1,0]))
                nearN[1] = int(round(gamma[-1,1]))
                while np.isinf(totalCostMap[nearN[1],nearN[0]]):
                    gamma = np.delete(gamma,-1,axis=0)
                    nearN[0] = int(round(gamma[-1,0]))
                    nearN[1] = int(round(gamma[-1,1]))

                if gamma.size > 0:
                    while (np.linalg.norm(gamma[-1,:]-nearN)) < 1:
                        gamma = np.delete(gamma,-1,axis=0)
                        if gamma.size == 0:
                            break

                gamma = np.vstack((gamma,np.array(nearN).T))
                currentT = totalCostMap[nearN[1],nearN[0]]
                for i in range(1,5):
                    if i == 1:
                        nodeChild = np.uint32(nearN + [0,-1])
                    elif i == 2:
                        nodeChild = np.uint32(nearN + [0,1])
                    elif i == 3:
                        nodeChild = np.uint32(nearN + [-1,0])
                    elif i == 4:
                        nodeChild = np.uint32(nearN + [1,0])
                    elif i == 5:
                        nodeChild = np.uint32(nearN + [-1,-1])
                    elif i == 6:
                        nodeChild = np.uint32(nearN + [1,1])
                    elif i == 7:
                        nodeChild = np.uint32(nearN + [-1,1])
                    elif i == 8:
                        nodeChild = np.uint32(nearN + [1,-1])

                    if totalCostMap[nodeChild[1],nodeChild[0]] < currentT:
                        currentT = totalCostMap[nodeChild[1],nodeChild[0]]
                        dx = (nearN[0]-nodeChild[0])/tau
                        dy = (nearN[1]-nodeChild[1])/tau
            except:
                return gamma

        if np.linalg.norm([dx,dy]) < 0.01:
            dnx = dx/math.sqrt(dx**2+dy**2)
            dny = dy/math.sqrt(dx**2+dy**2)
            add = gamma[-1,:] - np.dot(tau,[dnx,dny])
            gamma = np.vstack((gamma,np.array(add).T))
        else:
            dx = dx/math.sqrt(dx**2+dy**2)
            dy = dy/math.sqrt(dx**2+dy**2)
            add = gamma[-1,:] - np.dot(tau,[dx,dy])
            gamma = np.vstack((gamma,np.array(add).T))

        if np.linalg.norm(gamma[-1,:]-endWaypoint)<1.5:
            break

    gamma = np.vstack((gamma,np.array(endWaypoint).T))

    return gamma

# =============================================================================
#    Use this gradient computation instead of np.gradient, since this acknow-
#    ledges cost discontinuities
# =============================================================================
def computeGradient(cost, point=[]):

    m,n = cost.shape

    if len(point) == 0:
        jmax,imax = m,n
        jmin,imin = 0,0
    else:
        jmax,imax = min(m,int(point[1])+3),min(n,int(point[0])+3)

        jmin,imin = max(int(0),int(point[1]-3)),max(int(0),int(point[0]-3))


    Gx = np.zeros_like(cost)
    Gy = np.zeros_like(cost)
    Gnx = np.zeros_like(cost)
    Gny = np.zeros_like(cost)

    for i in range(imin,imax):
        for j in range(jmin, jmax):
            if j == 0:
                Gy[0,i] = cost[1,i]-cost[0,i]
            else:
                if j == m-1:
                    Gy[j,i] = cost[j,i]-cost[j-1,i]
                else:
                    if np.isinf(cost[j+1,i]):
                        if np.isinf(cost[j-1,i]):
                            Gy[j,i] = 0
                        else:
                            Gy[j,i] = cost[j,i]-cost[j-1,i]
                    else:
                        if np.isinf(cost[j-1,i]):
                            Gy[j,i] = cost[j+1,i]-cost[j,i]
                        else:
                            Gy[j,i] = (cost[j+1,i]-cost[j-1,i])/2

            if i == 0:
                Gx[j,0] = cost[j,1]-cost[j,0]
            else:
                if i == n-1:
                    Gx[j,i] = cost[j,i]-cost[j,i-1]
                else:
                    if np.isinf(cost[j,i+1]):
                        if np.isinf(cost[j,i-1]):
                            Gx[j,i] = 0
                        else:
                            Gx[j,i] = cost[j,i]-cost[j,i-1]
                    else:
                        if np.isinf(cost[j,i-1]):
                            Gx[j,i] = cost[j,i+1]-cost[j,i]
                        else:
                            Gx[j,i] = (cost[j,i+1]-cost[j,i-1])/2

            Gnx[j,i] = Gx[j,i]/math.sqrt(Gx[j,i]**2+Gy[j,i]**2)
            Gny[j,i] = Gy[j,i]/math.sqrt(Gx[j,i]**2+Gy[j,i]**2)


    return Gnx,Gny




def interpolatePoint(point,mapI):
    i = np.uint32(np.fix(point[0]))
    j = np.uint32(np.fix(point[1]))
    a = point[0] - i
    b = point[1] - j


    m,n = np.uint32(mapI.shape)

    if i == n:
        if j == m:
            I = mapI[j,i]
        else:
            I = b*mapI[j+1,i] + (1-b)*mapI[j,i]
    else:
        if j == m:
            I = a*mapI[j,i+1] + (1-a)*mapI[j,i]
        else:
            a00 = mapI[j,i]
            a10 = mapI[j,i+1] - mapI[j,i]
            a01 = mapI[j+1,i] - mapI[j,i]
            a11 = mapI[j+1,i+1] + mapI[j,i] - mapI[j,i+1] - mapI[j+1,i]
            if a == 0:
                if b == 0:
                    I = a00
                else:
                    I = a00 + a01*b
            else:
                if b == 0:
                    I = a00 + a10*a
                else:
                    I = a00 + a10*a + a01*b + a11*a*b

    return I