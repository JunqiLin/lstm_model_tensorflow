#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:25:28 2019

@author: linjunqi
"""

"""
mdoel data generation
"""
import pymongo
import pandas as pd
import numpy as np
from CONSTANT import TIME_STEPS,BATCH_SIZE,DB_NAME


def get_top(num):
    strnum = list(str(num))
    for i,item in enumerate(strnum):
        if i >0:
            strnum[i] = '0'
    result = int("".join(strnum))
    return result
    

    
class DataGenerator(object):
    def __init__(self, symbol,batch_size,steps,split):
        self.symbol = symbol
        self.batch_size = batch_size
        self.steps = steps
        self.split = split
        self.start = 0
        self.tstart = 0
        
        
        client = pymongo.MongoClient('localhost', 27017)
        db = client[DB_NAME]
        table = db[symbol]
        data = pd.DataFrame(list(table.find()))
        
        data.set_index(["datetime"], inplace=True)
        data = data[['symbol','open','close','high','low','volume']]
        
        self.data = data[:6000]
#        self.data = data
        self.length = get_top(len(self.data))
        self.train_len = int(round(self.length*split))
        self.test_len = self.length - self.train_len

        
        self.open = self.data['open']
        self.close = self.data['close']
        self.high = self.data['high']
        self.low = self.data['low']
        self.volume = self.data['volume']
        
        self.norm_close = np.array([])
        self.train_data = np.array([])
        self.test_data = np.array([])
        self._normolize_data()
        self._generate_train_data()
        self._generate_test_data()
        
    def _normolize_data(self):
        np_close = np.array(self.close)
        norm_close = (np_close-np.mean(np_close))/np.std(np_close)
        self.norm_close = norm_close[:self.length]
    

    def _generate_train_data(self):
        self.train_data = self.norm_close[:self.train_len]
        

    def _generate_test_data(self):
        self.test_data =  self.norm_close[-self.test_len:]
    

    def get_batch_from_train(self):
        BATCH_START = self.start
        trainDataX = self.train_data[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE].reshape((BATCH_SIZE,TIME_STEPS))
        trainDataY = self.train_data[BATCH_START+1:BATCH_START+TIME_STEPS*BATCH_SIZE+1].reshape((BATCH_SIZE,TIME_STEPS))
        self.start += TIME_STEPS
        return [trainDataX[:,:,np.newaxis],trainDataY[:,:,np.newaxis]]
    
    def get_batch_from_test(self):
        BATCH_START = self.tstart
        testDataX = self.test_data[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE].reshape((BATCH_SIZE,TIME_STEPS))
        testDataY = self.test_data[BATCH_START+1:BATCH_START+TIME_STEPS*BATCH_SIZE+1].reshape((BATCH_SIZE,TIME_STEPS))
        self.tstart += TIME_STEPS
        return [testDataX[:,:,np.newaxis],testDataY[:,:,np.newaxis]]
    
#    def get_batch(self):
#        BATCH_START = self.start
#        testDataX = self.norm_close[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE].reshape((BATCH_SIZE,TIME_STEPS))
#        testDataY = self.norm_close[BATCH_START+1:BATCH_START+TIME_STEPS*BATCH_SIZE+1].reshape((BATCH_SIZE,TIME_STEPS))
#        self.start += TIME_STEPS
#        return [testDataX[:,:,np.newaxis],testDataY[:,:,np.newaxis]]
    
    def get_batch(self):
        BATCH_START = self.start
        x_batch = []
        y_batch=[]
        for i in range(BATCH_SIZE):
            xrow_data = self.norm_close[BATCH_START:BATCH_START+TIME_STEPS]
            yrow_data = self.norm_close[BATCH_START+1:BATCH_START+TIME_STEPS+1]
            x_batch.append(xrow_data)
            y_batch.append(yrow_data)
            BATCH_START += 1
        
        testDataX = np.array(x_batch).reshape([-1]).reshape((BATCH_SIZE,TIME_STEPS))    
        testDataY = np.array(y_batch).reshape([-1]).reshape((BATCH_SIZE,TIME_STEPS))   
        self.start += BATCH_SIZE
        return [testDataX[:,:,np.newaxis],testDataY[:,:,np.newaxis]]
    
#dg = DataGenerator('add',50,20,0.8)
#print(dg.data.head())
#r = dg.get_batch()
#r = dg.get_batch()
#print(r)
#print(dg.length)
#
##for i in range(dg.train_len):
##    print(dg.norm_close[i])
##    print(dg.train_data[i])
#
#for i in range(dg.test_len):
#    print(dg.norm_close[i+dg.train_len])
#    print(dg.test_data[i])
#    print("##")


