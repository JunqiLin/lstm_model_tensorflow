#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:01:05 2019

@author: linjunqi
"""

"""
data_process
"""

"""
Store csv file data into Mongodb 
"""


from time import time
from datetime import datetime
import csv
import pymongo
import os
import pandas as pd 
from CONSTANT import DB_NAME
#DB_NAME = "SP500_TOP10"



class BarData(object):
    """
    standard data format designed
    """
    def __init__(self):
        """"""
        self.symbol = 0
        self.datetime=0
        self.volume= 0
        self.open = 0
        self.high = 0
        self.low = 0
        self.close = 0

def loadCsvFromYahoo(fileName, dbName, symbol):
    print("Begin to load csv files %s to %s 's %s")%(fileName, dbName, symbol)
    
    client = pymongo.MongoClient()
    collection = client[dbName][symbol]
    collection.ensure_index([('datetime',pymongo.ASCENDING)],unique = True)
    
    with open(fileName,'r') as f:
#        reader = csv.DictReader(f)
        reader = pd.read_csv(f)
        reader.dropna(axis=0, how='any', inplace=True)
        for index, d in reader.iterrows():
            bar = BarData()
            bar.symbol = symbol
            bar.open = float(d['Open'])
            bar.high = float(d['High'])
            bar.low = float(d['Low'])
            bar.close = float(d['Close'])
            bar.datetime = datetime.strptime(d['Date'],'%Y/%m/%d')
            bar.volume = d['Volume']
            
            flt = {'datetime':bar.datetime}
            collection.update_one(flt,{'$set':bar.__dict__},upsert = True)
    print(u'Insert finished')
    
if __name__ =="__main__":
    flag="all"
    if flag =="all":
        path = os.path.abspath(os.path.dirname(os.getcwd()))+'/data'
        filenames = os.listdir(path)[2:]
#        filenames=['add.csv']
        for f in filenames:
            print(f)
            fn = path+'/'+str(f)
            loadCsvFromYahoo(fn, DB_NAME, str(f)[:-4])
        print('All Data has been inserted')
#        
        
    
    
            

