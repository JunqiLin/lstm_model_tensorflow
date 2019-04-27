#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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
from CONSTANT import TIME_STEPS,BATCH_SIZE,INPUT_SIZE,OUTPUT_SIZE,CELL_SIZE,LR
import tensorflow as tf
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
tf.reset_default_graph()

class LSTM(object):
    
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
    
    def build_model(self):
        
        with tf.variable_scope('in_hidden'):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            bs_in = self._bias_variable([self.cell_size,])
            
            with tf.name_scope('Wx_plus_in_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
    
        ##lstm cell
        with tf.variable_scope('LSTM_cell'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope('initial_state'):
                cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                    lstm_cell, l_in_y, initial_state=cell_init_state, time_major=False)
    
        ##output layer
        with tf.variable_scope('out_hidden'):
            l_out_x = tf.reshape(cell_outputs, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            with tf.name_scope('Wx_plus_out_b'):
                pred = tf.matmul(l_out_x, Ws_out) + bs_out
        

            
        model={'pred':pred,'final_state':cell_final_state}   
        return model
    
    def train_lstm(self, pred):
 
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
    generator = DataGenerator('AAPL',BATCH_SIZE, TIME_STEPS, split)
    

    train_times = int((generator.length-TIME_STEPS*BATCH_SIZE)/TIME_STEPS*split)
    test_times = int(round(((generator.length-TIME_STEPS*BATCH_SIZE)/TIME_STEPS)*(1-split)))

    drawtrain=[]
    drawtest=[]
    

    
    for i in range(2000):
        index = 19
        seq, res = generator.get_batch()
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
        
#        drawtrain.append(pred[:TIME_STEPS])
        
        for j in range(BATCH_SIZE):
            drawtrain.append(pred[index])
            index+=20

            
    
            
    print("pause")
    
#    for i in range(test_times-1):
#        test_seq, test_res = generator.get_batch()
#        feed_dict = {lstm.xs: test_seq, lstm.ys: test_res}
#        
#        cost, pred = sess.run([lstm.optimizer['cost'],lstm.prediction], feed_dict = feed_dict)
#        print('cost: ', round(cost, 4))
#        
#        if i < test_times-2:
#            drawtest.append(pred[:TIME_STEPS])
#        if i == test_times-2:
#            drawtest = np.array(drawtest).reshape([-1])
#            last_pred = np.array(pred.reshape([-1]))
#            drawtest = np.concatenate((drawtest,last_pred))
#            



    drawtrain = np.array(drawtrain).reshape([-1])
#    drawtest = np.array(drawtest)
#    draw = np.concatenate((drawtrain,drawtest)).reshape([-1])
    
              
    pd_drawpred = pd.DataFrame(draw)
    
    pd_or = pd.DataFrame(generator.norm_close)
    

    
    train_pic = go.Scatter(x=pd_drawpred.index, y=pd_drawpred[0], name='train_pre')
    origin_pic = go.Scatter(x=pd_or.index,y = pd_or[0],name='real data')
    
    r = [train_pic,origin_pic]
    fig = go.Figure(data=r)
    py.plot(fig)
#    
#    
    
