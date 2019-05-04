#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:14:36 2019

@author: linjunqi
"""


from __future__ import division
from sympy import *
import numpy as np

data = np.array([1,2,3,4,2,3,5,6])

def line(data):
    """
    fitting line
    """
    length = len(data)
    result = np.zeros(length)
    start = 0
    end = length-1
    mid = int((end-start)/2)

    start_num = data[start]
    mid_num = data[mid]
    end_num = data[end]
    p_len = mid - start
    l_len = end -mid
    p_delta = (data[mid]-data[start])/p_len
    l_delta = (data[end]-data[mid])/l_len
    for i in range(p_len):
        result[i] = data[start] + p_delta * i
    for i in range(l_len+1):
        result[i+p_len] = data[mid] + l_delta * i
    return result 
    



def equation(data):
    """
    fitting curve line
    """
    length = len(data)
    start = 0
    end = length - 1
    q_mid = int((start+end)/3)
    h_mid = int(q_mid * 2)
    x = np.array([start,q_mid,h_mid,end])
    y = np.array([data[start],data[q_mid],data[h_mid],data[end]])
    
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    
    f =  solve([a * x[0]**3 + b * x[0]**2 + c* x[0] +d - y[0], 
             a * x[1]**3 + b * x[1]**2 + c* x[1] +d - y[1],
              a * x[2]**3 + b * x[2]**2 + c* x[2] +d - y[2],
               a * x[3]**3 + b * x[3]**2 + c* x[3] +d - y[3]],[a, b, c, d])
    

    degree = 1000
    z = np.zeros(degree)
    h = np.zeros(degree)
    result = np.zeros(length)
    temp = 0
    delta = len(data)/degree
    for i in range(degree):
        h[i] = temp
        temp += delta
    
    for i in range(degree):
        z[i] = f[a]*h[i]**3 + f[b]*h[i]**2 + f[c]*h[i] + f[d] 
    
    for i in range(length):
        result[i] = z[int((i/length)*degree)]
    return h,result















