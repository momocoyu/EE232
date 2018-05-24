#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:53:47 2018

@author: yangyang
"""

from numpy import *  
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as pl
import copy

#Q1-hotmap
rw1 = np.zeros((10, 10), dtype=np.double)
rw1[9][9]=1.0
sc=plt.pcolor(rw1,cmap='hot')
plt.colorbar(sc)
plt.gca().invert_yaxis()
plt.show()

rw2 = np.zeros((10, 10), dtype=np.double)
rw2[1][4]=-100.0
rw2[2][4]=-100.0
rw2[3][4]=-100.0
rw2[4][4]=-100.0
rw2[5][4]=-100.0
rw2[6][4]=-100.0
rw2[1][5]=-100.0
rw2[1][6]=-100.0
rw2[2][6]=-100.0
rw2[3][6]=-100.0
rw2[7][6]=-100.0
rw2[8][6]=-100.0
rw2[3][7]=-100.0
rw2[7][7]=-100.0
rw2[3][8]=-100.0
rw2[4][8]=-100.0
rw2[5][8]=-100.0
rw2[6][8]=-100.0
rw2[7][8]=-100.0
rw2[9][9]=10.0
sc=plt.pcolor(rw2,cmap='hot')
plt.colorbar(sc)
plt.gca().invert_yaxis()
plt.show()


#Q2-create environment

class Environment:
    
    def __init__(self):

        #reward function
        self.rw = np.zeros((10, 10), dtype=np.double)
        self.rw[9][9]=1.0
        
        self.w=0.1
        
        #discount factor
        self.discount_factor=0.8
        
        
        #state space
        self.state_space=np.zeros((10, 10), dtype=np.double)
        k=0
        for i in range(10):
            for j in range(10):
                self.state_space[j][i]=k
                k=k+1
        
        
        #action set
        self.action=['up','down','left','right']
        
    # transition probabilities
    def transition(self, action,next_state):
        edgeup=np.arange(10,90,10)
        edgedown=np.arange(19,99,10)
        edgeleft=np.arange(1,9,1)
        edgeright=np.arange(91,99,1)
    
        if self.state==0:
            if next_state ==10 or next_state== 1 or next_state==0:
                
                if action == 'right':
                    if next_state == 10:
                        p= 1-self.w+self.w/4.0
                    if next_state == 0: 
                        p= self.w/2.0
                    if next_state == 1: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == 1:
                        p= 1-self.w+self.w/4.0
                    if next_state == 0: 
                        p= self.w/2.0
                    if next_state == 10: 
                        p= self.w/4.0
                
                else:
                    if next_state == 0:
                        p= 1-self.w+self.w/2
                    else:
                        p= self.w/4.0
            else: p=0
        
        elif self.state==9:
            if next_state ==9 or next_state== 8 or next_state==19:
                
                if action == 'right':
                    if next_state == 19:
                        p= 1-self.w+self.w/4.0
                    if next_state == 9: 
                        p= self.w/2.0
                    if next_state == 8: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == 8:
                        p= 1-self.w+self.w/4.0
                    if next_state == 9: 
                        p= self.w/2.0
                    if next_state == 19: 
                        p= self.w/4.0
                
                else:
                    if next_state == 9:
                        p= 1-self.w+self.w/2.0
                    else:
                        p= self.w/4.0
            else: p=0
             
        elif self.state==90:
            if next_state ==90 or next_state== 80 or next_state==91:
                
                if action == 'left':
                    if next_state == 80:
                        p= 1-self.w+self.w/4.0
                    if next_state ==90: 
                        p= self.w/2.0
                    if next_state == 91: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == 91:
                        p= 1-self.w+self.w/4.0
                    if next_state == 90: 
                        p= self.w/2.0
                    if next_state == 80: 
                        p= self.w/4.0
                
                else:
                    if next_state == 90:
                        p= 1-self.w+self.w/2.0
                    else:
                        p= self.w/4.0
            else: p=0
            
        elif self.state==99:
            if next_state ==98 or next_state== 89 or next_state==99:
                
                if action == 'left':
                    if next_state == 89:
                        p= 1-self.w+self.w/4.0
                    if next_state == 99: 
                        p= self.w/2.0
                    if next_state == 98: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == 98:
                        p= 1-self.w+self.w/4.0
                    if next_state == 99: 
                        p= self.w/2.0
                    if next_state == 89: 
                        p= self.w/4.0
                
                else:
                    if next_state == 99:
                        p= 1-self.w+self.w/2.0
                    else:
                        p= self.w/4.0
            else: p=0
            
        elif self.state in edgeup:
            if next_state ==self.state-10 or next_state== self.state+10 or next_state==self.state+1 or next_state==self.state:
                if action == 'up':
                    if next_state == self.state:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == self.state+1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'left':
                    if next_state == self.state-10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'right':
                    if next_state == self.state+10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
            else: p=0
            
        elif self.state in edgedown:
            if next_state ==self.state-10 or next_state== self.state+10 or next_state==self.state-1 or next_state==self.state:
                if action == 'down':
                    if next_state == self.state:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == self.state-1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'left':
                    if next_state == self.state-10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'right':
                    if next_state == self.state+10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
            else: p=0
            
        elif self.state in edgeleft:
            if next_state ==self.state-1 or next_state== self.state+1 or next_state==self.state+10 or next_state==self.state:
                if action == 'left':
                    if next_state == self.state:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'right':
                    if next_state == self.state+10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == self.state-1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == self.state+1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
            else: p=0
            
        elif self.state in edgeright:
            if next_state ==self.state-1 or next_state== self.state+1 or next_state==self.state-10 or next_state==self.state:
                if action == 'right':
                    if next_state == self.state:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'left':
                    if next_state == self.state-10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == self.state-1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == self.state+1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
            else: p=0
            
        else:
            if next_state ==self.state-1 or next_state== self.state+1 or next_state==self.state-10 or next_state==self.state+10:
                if action == 'right':
                    if next_state == self.state+10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'left':
                    if next_state == self.state-10:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'up':
                    if next_state == self.state-1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
                
                elif action == 'down':
                    if next_state == self.state+1:
                        p= 1-self.w+self.w/4.0
                    else: 
                        p= self.w/4.0
            else: p = 0
            
        return p



obj = Environment()
rwf=[0]*100
rwf[99] = 1.0
import math
V=np.zeros(shape=(10,10))
Vt=np.zeros(shape=(10,10))
A = np.empty(shape=(10,10), dtype='object')
delta=math.inf

while delta > 0.01:
    delta = 0
    m=0
    Vt=copy.deepcopy(V)
    for i in range(10):
        for j in range(10):
            val = V[j][i]
            k = i*10+j
            obj.state = m
            m=m+1
            
            
            tempu =0
            tempd = 0
            templ = 0
            tempr = 0
            for q in range(100):
                    
                t=q//10
                z=q-t*10
                tempu = tempu + obj.transition('up',q)*(rwf[q]+obj.discount_factor*Vt[z][t])
                tempd = tempd + obj.transition('down',q)*(rwf[q]+obj.discount_factor*Vt[z][t])
                templ= templ + obj.transition('left',q)*(rwf[q]+obj.discount_factor*Vt[z][t])
                tempr = tempr + obj.transition('right',q)*(rwf[q]+obj.discount_factor*Vt[z][t])
                
            V[j][i]= max(tempu,tempd,templ,tempr)
            if V[j][i] == tempu:
                A[j][i] = '↑'
            elif V[j][i] == tempd:
                A[j][i] = '↓'
            elif V[j][i] == templ:
                A[j][i] = '←'
            elif V[j][i] == tempr:
                A[j][i] = '→'
            delta=max(delta,abs(val-V[j][i]))

#print table
pl.figure()
tb = pl.table(cellText=V, loc=(0,0), cellLoc='center')

tc = tb.properties()['child_artists']
for cell in tc: 
    cell.set_height(0.25)
    cell.set_width(0.25)

ax = pl.gca()
ax.set_xticks([])
ax.set_yticks([])
pl.show()

#arrow
pl.figure()
tb = pl.table(cellText=A, loc=(0,0), cellLoc='center')

tc = tb.properties()['child_artists']
for cell in tc: 
    cell.set_height(0.1)
    cell.set_width(0.1)

ax = pl.gca()
ax.set_xticks([])
ax.set_yticks([])
pl.show()

#print hot map 
sc=plt.pcolor(V,cmap='hot')
plt.colorbar(sc)
plt.gca().invert_yaxis()
plt.show()
                
                        
        