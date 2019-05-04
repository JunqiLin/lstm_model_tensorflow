#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Mon Apr 22 23:01:49 2019

@author: linjunqi
"""

"""
lstm model
"""

import numpy as np
import pandas as pd
from data_generator import DataGenerator
import smooth as sm
from CONSTANT import TIME_STEPS,BATCH_SIZE,INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,LR
import tensorflow as tf
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import math
tf.reset_default_graph()

class LSTM(object):
    """
    lstm model
    """
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
        self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size ], name='ys')
        
        self.n_steps =n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        self.prediction = self.build_model()['pred']
        self.optimizer = self.train_lstm(self.prediction)
    """
    generate random weights and biases
    """    
    @staticmethod 
    def _weight_variable(shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer,name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    @staticmethod 
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))
    ##mean-square function to calculate error
    
    
    def build_model(self):
        """
        build input hidden layer
        """
        with tf.variable_scope('in_hidden'):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            bs_in = self._bias_variable([self.cell_size,])
            
            with tf.name_scope('Wx_plus_in_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
    
        """
        build lstm cell and feed data from input hidden layer into it
        """
        with tf.variable_scope('LSTM_cell'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope('initial_state'):
                cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                    lstm_cell, l_in_y, initial_state=cell_init_state, time_major=False)
    
        """
        build output hidden layer
        """
        with tf.variable_scope('out_hidden'):
            l_out_x = tf.reshape(cell_outputs, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            with tf.name_scope('Wx_plus_out_b'):
                pred = tf.matmul(l_out_x, Ws_out) + bs_out
        

            
        model={'pred':pred,'final_state':cell_final_state}   
        return model
    
    def train_lstm(self, pred):
        """
        calculate loss
        """
        with tf.name_scope('cost'):
            losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(pred, [-1], name='reshape_pred')],
                [tf.reshape(self.ys, [-1], name='reshape_target')],
                [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=self.ms_error,
                name='losses'
            )
            with tf.name_scope('average_cost'):
                cost = tf.div(
                    tf.reduce_sum(losses, name='losses_sum'),
                    self.batch_size,
                    name='average_cost')
                tf.summary.scalar('cost', cost)
        """
        AdamOptimizer training
        """
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(LR).minimize(cost)
        result={'cost':cost,'train_op':train_op}
        return result


if __name__ == '__main__':
    
    lstm= LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    split = 0.8
    generator = DataGenerator('add',BATCH_SIZE, TIME_STEPS, split)
    
    
    seq_len = BATCH_SIZE+TIME_STEPS-1
    train_times = int(math.floor((generator.train_len - seq_len)/BATCH_SIZE))
    times = int(math.floor((generator.length - seq_len)/BATCH_SIZE))
    test_times = times-train_times
    

    drawtest=[]
    drawtest=[]
    drawtrend=[]
    
    P_STEP = 1
    """
    training
    """
    for i in range(train_times):
        index = TIME_STEPS-1
        seq, res = generator.get_batch(P_STEP)
        feed_dict = {
                    lstm.xs: seq,
                    lstm.ys: res
            }

        _, cost,  pred = sess.run(
            [lstm.optimizer['train_op'],lstm.optimizer['cost'], lstm.prediction],
            feed_dict=feed_dict)

        print('cost: ', round(cost, 4))
        result = sess.run(merged, feed_dict)
        writer.add_summary(result, i)
        writer.flush()
        
        
        
    end_pre = (train_times-1)*BATCH_SIZE + seq_len -1
    print(end_pre)
            
    print("Test Set Cost")
    
    """
    testing
    """
    for i in range(test_times):
        index = TIME_STEPS-1
        test_seq, test_res = generator.get_batch(P_STEP)
        feed_dict = {lstm.xs: test_seq, lstm.ys: test_res}
        
        cost, pred = sess.run([lstm.optimizer['cost'],lstm.prediction], feed_dict = feed_dict)
        print('cost: ', round(cost, 4))
        
        for j in range(BATCH_SIZE):
            drawtest.append(pred[index])
            index+=TIME_STEPS      
#    
    true_data = generator.norm_close[end_pre:]
    ##real data
    pd_or = pd.DataFrame(generator.norm_close[end_pre:])
    
    ##prediction sequence
    t = 0
    for _ in range(test_times):
        data = drawtest[t:t+BATCH_SIZE]
        data = sm.line(data)
        drawtrend.append(data)
        t += BATCH_SIZE
    """
    output result
    """
    ##prediction trend curve
    drawtrend = np.array(drawtrend).reshape([-1])

    ##prediction curve
    drawtest = np.array(drawtest).reshape([-1])
    
    right = 0
    wrong = 0

    for l in range(len(drawtest)):
        if l == len(drawtest)-1:
            continue
        if (drawtest[l+1]>drawtest[l] and true_data[l+1]>true_data[l]) or (drawtest[l+1]<drawtest[l] and true_data[l+1]<true_data[l]):
            right+=1
        if (drawtest[l+1]>drawtest[l] and true_data[l+1]<true_data[l]) or (drawtest[l+1]<drawtest[l] and true_data[l+1]>true_data[l]):
            wrong +=1
            
    print("Right:%s"%(right))
    print("Wrong:%s"%(wrong))
    accuracy = right / (wrong + right)
    print("Accuracy Percentage is %s"%(accuracy))
    pd_drawtrend = pd.DataFrame(drawtrend)
    pd_drawpred = pd.DataFrame(drawtest)
    
    
    train_pic = go.Scatter(x=pd_drawpred.index, y=pd_drawpred[0], name='test_pre_point')
    trend_pic = go.Scatter(x=pd_drawtrend.index, y=pd_drawtrend[0], name='test_pre_trend')
    origin_pic = go.Scatter(x=pd_or.index,y = pd_or[0],name='real data')
    
    r = [train_pic,origin_pic,trend_pic]
    fig = go.Figure(data=r)
    py.plot(fig)
##    
#    
    
