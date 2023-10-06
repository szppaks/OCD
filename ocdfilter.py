
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021.09.16

@author: Peter Szutor


Octree based "radius-outlier-like" point cloud filter
Based on  'Fast Radius Outlier Filter Variant for Large Point Clouds by Péter Szutor and Marianna Zichar'
https://www.mdpi.com/2306-5729/8/10/149

Copyright 2021 Peter Szutor

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import math
from numba import jit, njit

def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht))

#@jit
def octreecodes(ppoints,pdepht,maxx,minx,maxy,miny,maxz,minz):  #kiszámolja az octree kódokat.
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=(np.searchsorted(xletra,ppoints[:,0],side='right')-1).astype(np.int16)
    otcodey=(np.searchsorted(yletra,ppoints[:,1],side='right')-1).astype(np.int16)
    otcodez=(np.searchsorted(zletra,ppoints[:,2],side='right')-1).astype(np.int16)
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    return (ki,minx,maxx,miny,maxy,minz,maxz)

    
def ockodossze(x,y,z,pdepht):
    ki=np.left_shift(x,pdepht*2)+np.left_shift(y,pdepht)+z
    return ki




#@jit
def obfilter(ppoints,cubesize,ocnum,ncnum,verbose=False): 
    '''
    

    Parameters
    ----------
    ppoints : numpy array (n,3)
        the xyz koordinates
    cubesize : float
        size of the cube -> like radius in ROL filter
    ocnum : int
        Own cell count -> 1-20
    ncnum : int
        neighbour cells count -> 0-1000
    verbose : The default is False.

    Returns
    -------
    filtpoints : numpy array (n,3)
        the points without outliers

    '''
    sajatpontdarab=ocnum
    szomszedpontdarab=ncnum
    kozvszomszedsuly=6
    elszomszedsuly=18
    #kiterjedészámolás
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    sizes=np.array([maxx-minx,maxy-miny,maxz-minz])
    #print(sizes)
    maxsize=np.amax(sizes)# LEgnagyobb kioterjedés, ehhez méretezem az octree mélységet
    ocdepht=math.floor(math.log(maxsize/cubesize))+1 # a kiterjedésből és a kockamérteből kiszámolom, milyen mélységű octree kell. Max 9 lehet memóriaproblémák miatt
    
    #print('ocdepht:',ocdepht)
    if ocdepht>10:
        print("A cubesize nem jo, fentebb kell veszem a minimális méretre:",2**(ocdepht-9)*cubesize)
        ocdepht=10
    if verbose:    
        print('Ocdepht',ocdepht)
    #kiszámolom minden ponthoz az octree kódokat
    occ=octreecodes(ppoints,ocdepht,maxx,minx,maxy,miny,maxz,minz)
    #print('maxoccode',np.amax(occ[0]))
    #print('octree ready')
    #occsorted=occ[occ[:,0].argsort()]
    #ebbe a tommbe gyujtom a darabszamokat. Pont olyan méretű, amekkora octree van, 2**(depth*3) ha depth >8, az baj. Az index az octreekód
    #saját darabszám, szomszéd darabszám
    gyujto=np.zeros((2**(ocdepht*3),2),dtype='single')
    ockod, darab = np.unique(occ[0], return_counts=True) # az octree kódokból kiszűröm az ismétlődést, illetve megadja, hogy hány darab van egy octree kódból.
    darab=darab.astype(np.int32)
    #begyűjtöm a darabszámokat és szomszéd darabszámokat
    if verbose:
        print('octree values:',len(ockod))
    for i in range(0,len(ockod)):
        aktockod=ockod[i]
        gyujto[aktockod,0]=darab[i]
    for i in range(0,len(ockod)):
        xmask=''.join('1'*ocdepht)+''.join('0'*ocdepht*2)
        ymask=''.join('0'*ocdepht)+''.join('1'*ocdepht)+''.join('0'*ocdepht)
        zmask=''.join('0'*ocdepht*2)+''.join('1'*ocdepht)
        
        xcode=np.right_shift(np.bitwise_and(aktockod,int(xmask,2 )),ocdepht*2)
        ycode=np.right_shift(np.bitwise_and(aktockod,int(ymask,2 )),ocdepht)
        zcode=np.bitwise_and(aktockod,int(zmask,2 ))
        
        aktgyujtopot=ockodossze(xcode,ycode,zcode,ocdepht)
        #közvetlen szomszédok
        eltol=ockodossze(xcode+1,ycode,zcode,ocdepht)
        if eltol<len(gyujto):
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
        eltol=ockodossze(xcode-1,ycode,zcode,ocdepht)
        if eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
        eltol=ockodossze(xcode,ycode+1,zcode,ocdepht)
        if eltol<len(gyujto):
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
        eltol=ockodossze(xcode,ycode-1,zcode,ocdepht)
        if eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
        eltol=ockodossze(xcode,ycode,zcode+1,ocdepht)
        if eltol<len(gyujto):
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
        eltol=ockodossze(xcode,ycode,zcode-1,ocdepht)
        if eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/kozvszomszedsuly
            
        #élszomszédok
        eltol=ockodossze(xcode+1,ycode+1,zcode,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode-1,zcode,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode+1,zcode,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode-1,zcode,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
            
        #sarokszomszédok
        eltol=ockodossze(xcode+1,ycode+1,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode-1,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode+1,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode+1,ycode+1,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode+1,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode-1,zcode+1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode+1,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
        eltol=ockodossze(xcode-1,ycode-1,zcode-1,ocdepht)
        if eltol<len(gyujto) and eltol>=0:
            gyujto[aktgyujtopot,1]=gyujto[aktgyujtopot,1]+gyujto[eltol,0]/elszomszedsuly
            
    #print('darab kesz',len(ockod))
    ockod=None
    darab=None
    marad=np.zeros((len(ppoints)),dtype=bool)   
    darabtomb=np.take(gyujto,occ[0],axis=0)[:,:] # az összes ponthoz megkeresem a gyűjtő tömbben octree kód alapján, hogy mennyi az "indexe", azaz a kiszámolt szomszédossági mutató
    occ=None
    gyujto=None

    marad=np.logical_or(darabtomb[:,0]>sajatpontdarab,darabtomb[:,1]>szomszedpontdarab)
    darabtomb=None
    filtpoints=ppoints[marad,:3]
    marad=None
    return filtpoints

