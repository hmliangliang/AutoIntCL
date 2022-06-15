#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 11:16
# @Author  : Liangliang
# @File    : AutoIntCL.py
# @Software: PyCharm

import os
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import tensorflow.keras as keras
import time
import s3fs
import pandas as pd
import random
import datetime
import math
import multiprocessing
#https://developer.aliyun.com/article/700586
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#设置随机种子点
random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
os.environ['TF_DETERMINISTIC_OPS'] = '1' #设置GPU随机种子点

e = 0.00000001

def multiprocessingWrite(file_number,data,output_path,count):
    #print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    with fs.open(output_path + 'pred_{}.csv'.format(int(file_number)), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个大数据文件的第{}个子文件已经写入完成,写入数据的行数{} {}".format(count,file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )
class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output

    def write(self, data, args,count):
        #注意在此业务中data是一个二维list
        n_data = len(data) #数据的数量
        n = math.ceil(n_data/args.file_max_num) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        pool = multiprocessing.Pool(processes=args.process_num)
        start = time.time()
        for i in range(0,n):
            pool.apply_async(multiprocessingWrite, args=(i, data[i*args.file_max_num:min((i+1)*args.file_max_num,n_data)],self.output_path,count,))
        pool.close()
        pool.join()
        cost = time.time() - start
        print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))



class MLP(keras.Model):
    def __init__(self,output_feat1=200,output_feat2=100,output_feat=64):
        super(MLP, self).__init__()
        self.cov1 = keras.layers.Dense(output_feat1)
        self.cov2 = keras.layers.Dense(output_feat2)
        self.cov3 = keras.layers.Dense(output_feat)
        self.BatchNormalization = tf.keras.layers.BatchNormalization()
    def call(self, feat,training=None, mask=None):#采用resnet结构
        h1 = self.cov1(feat)
        h = tf.nn.elu(h1)
        h = tf.concat([h,h1],axis=1)
        h1 = self.cov2(h)
        h = self.BatchNormalization(h, training=training)
        h = tf.nn.elu(h1)
        h = tf.concat([h, h1], axis=1)
        h = self.cov3(h)
        return h

#AutoInt代码参考：https://zhuanlan.zhihu.com/p/53462648
#首先假设输入inputs的shape为(batch_size,field_size,embedding_size)
class AutoIntNet(keras.Model):
    def __init__(self,att_embedding_size, head_num,output_feat=64):
        super(AutoIntNet, self).__init__()
        self.Q = keras.layers.Dense(att_embedding_size*head_num,use_bias=False)
        self.K = keras.layers.Dense(att_embedding_size*head_num,use_bias=False)
        self.V = keras.layers.Dense(att_embedding_size*head_num,use_bias=False)
        self.w_res = keras.layers.Dense(att_embedding_size*head_num,use_bias=False)
        self.cov = keras.layers.Dense(output_feat)
        self.cov3 = keras.layers.Dense(1)  # 预测样本类标签的概率
        self.BatchNormalization = tf.keras.layers.BatchNormalization()
        self.use_res=True

    def call(self, feat,training=None, mask=None):#采用resnet结构
        querys = self.Q(feat) #[N,att_embedding_size*head_num]
        keys = self.K(feat) #[N,att_embedding_size*head_num]
        values = self.V(feat) #[N,att_embedding_size*head_num]
        inner_product = tf.matmul(querys,tf.transpose(keys))  # [N,N]
        normalized_att_scores = tf.nn.softmax(inner_product) # [N,N]
        result = tf.matmul(normalized_att_scores, values)  # [N,att_embedding_size*head_num]
        if self.use_res:
            result = result + self.w_res(feat) # [N,att_embedding_size*head_num]
        result = tf.nn.relu(result)  # [N,att_embedding_size*head_num]
        result = self.cov(result)
        result = self.BatchNormalization(result, training=training)
        h = tf.nn.relu(result)
        result = self.cov3(h)
        result = tf.nn.sigmoid(result)
        #result = tf.nn.softmax(result,axis=1)
        return [h,result]


def Loss(embed_autoint,embed_mlp,predict,label,args):
    #label大小:n*1
    n = label.shape[0]
    '''
    loss_pos = 0 #正样本之间的相似度->max
    loss_neg = 0 #正负样本之间的相似度->min
    loss_bpr = 0 #bpr损失(预测值与真实值之间的差异)->min
    loss_crossentropy = 0 #交叉熵->min
    for i in range(n):
        #正样本之间的loss
        loss_pos = loss_pos + tf.reduce_sum(embed_autoint[i,:]*embed_mlp[i,:])/(tf.norm(embed_autoint[i,:],2)*tf.norm(embed_mlp[i,:],2)+e)
        #正负样本之间的损失
        先进行负采样,每个样本的负采样的数目为neg_num
        sample = np.random.randint(0,n,args.neg_num)
        for j in sample:
            #r为添加的正样本比例,以提高对比学习的判别能力
            data_neg = embed_autoint[j,:] + args.r*embed_autoint[j,:]
            loss_neg = loss_neg +1/args.neg_num*tf.reduce_sum(embed_autoint[i,:]*data_neg)/(tf.norm(embed_autoint[i,:],2)*tf.norm(data_neg,2)+e))'''

    #第一步计算正样本之间的loss
    loss_pos = tf.reduce_sum(tf.reduce_sum(embed_autoint*embed_mlp,axis=1)/(tf.norm(embed_autoint,2,axis=1)*tf.norm(embed_mlp,2,axis=1)+e))
    #第二步负采样形成负样本矩阵
    neg_samples = tf.squeeze(tf.gather(embed_autoint,tf.reshape(tf.convert_to_tensor(np.random.randint(0,n,(n,args.neg_num)),dtype=tf.int32),(-1,1)),axis=0))
    #将原数据每一行重复args.neg_num次，每一行与负样本矩阵的行对应
    embed_autoint = tf.repeat(embed_autoint,args.neg_num,axis=0)
    #负样本矩阵中加入正样本信息
    neg_samples = neg_samples + args.r *embed_autoint
    #计算正负样本之间的相似性
    loss_neg = tf.reduce_sum(tf.reduce_sum(embed_autoint*neg_samples,axis=1)/(tf.norm(embed_autoint,2,axis=1)*tf.norm(neg_samples,2,axis=1)+e))

    loss_crossentropy = -0.5*tf.reduce_sum(tf.math.log(tf.math.maximum(predict,0.02))*label)-0.5*tf.reduce_sum(tf.math.log(tf.math.maximum(1-predict,0.02))*(1-label))

    loss_bpr = tf.reduce_sum(tf.nn.sigmoid(tf.math.abs(predict-label)))
    loss = 1/n*(args.bpr*loss_bpr + args.alpha*loss_crossentropy + args.beta*loss_neg - args.gamma*loss_pos)
    return loss


def noiseAgumentation(args,data):
    n,m = data.shape
    noise = tf.random.uniform((n,m))
    noise = tf.sign(data)*noise
    norm = tf.reshape(tf.norm(noise,2,axis=1),[n,-1])
    data = data + args.epsion*noise/norm
    return data

def train(args):
    tf.keras.backend.set_learning_phase(True)
    # 读取数据
    '''读取s3fs数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    if args.env == "train":  # 第一次训练,创建模型 att_embedding_size, head_num,output_feat=64
        net_auto = AutoIntNet(args.hidden_num1,args.head_num, args.output_dim)
        net_mlp = MLP(args.hidden_num1,args.hidden_num2,args.output_dim)
    else:  # 利用上一次训练好的模型，进行增量式训练
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net_auto"
        os.system(cmd)
        net_auto = keras.models.load_model("./JK_trained_model_net_auto", custom_objects={'tf': tf}, compile=False)
        print("AutoInt Model weights load!")

        cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net_mlp"
        os.system(cmd)
        net_mlp = keras.models.load_model("./JK_trained_model_net_mlp", custom_objects={'tf': tf}, compile=False)
        print("MLP Model weights load!")
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    print("开始读取数据! {}".format(datetime.datetime.now()))
    before_net_auto = net_auto
    before_net_mlp = net_mlp
    before_loss = 2 ** 32
    stop_num = 0
    for epoch in range(args.epoch):
        count = 0
        for file in input_files:
            count = count + 1
            print("epoch:{}当前正在处理第{}个文件,文件路径:{}......".format(epoch,count, "s3://" + file))
            data = pd.read_csv("s3://" + file, sep=',', header=None)# 读取数据
            data.fillna(0)
            label = tf.reshape(tf.convert_to_tensor(data.iloc[:,data.shape[1]-1],dtype=tf.float32),(data.shape[0],-1))
            data = tf.convert_to_tensor(data.iloc[:,0:data.shape[1]-1],dtype=tf.float32)
            data = tf.data.Dataset.from_tensor_slices((data,label)).shuffle(100).batch(args.batch_size,drop_remainder=True)
            count_batch = 0
            for batch_data,batch_label in data:
                count_batch = count_batch + 1
                with tf.GradientTape(persistent=True) as tape:
                    embed_autoint, predict = net_auto(noiseAgumentation(args,batch_data),training=args.flags)
                    embed_mlp = net_mlp(noiseAgumentation(args,batch_data),training=args.flags)
                    loss = Loss(embed_autoint,embed_mlp,predict,batch_label, args)  #数据的第3列为类标签
                if count_batch%10 == 0:
                    acc = accuracy_score(tf.math.maximum(tf.math.sign(predict-0.5),0),batch_label)
                    print("当前第{}个epoch第{}个文件第{}个batch的batch data训练结果loss:{} 训练集的accuracy:{} {}".format(epoch,count,count_batch,loss, acc,datetime.datetime.now()))
                gradients = tape.gradient(loss, net_auto.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net_auto.trainable_variables))
                gradients2 = tape.gradient(loss, net_mlp.trainable_variables)
                optimizer.apply_gradients(zip(gradients2, net_mlp.trainable_variables))
        if loss < before_loss:
            before_loss = loss
            before_net_auto = net_auto
            before_net_mlp = net_mlp
            stop_num = 0
        else:
            stop_num = stop_num + 1
            if stop_num >= args.stop_num:
                print("Stop early!")
                print("The epoch:{} The Loss:{} The best loss:{}  {}".format(epoch, loss, before_loss,datetime.datetime.now()))
                break
        print("The epoch:{} The Loss:{} The best loss:{}  {}".format(epoch, loss, before_loss,datetime.datetime.now()))
    # 保存神经网络模型
    net_auto = before_net_auto
    net_mlp = before_net_mlp
    # 保存teacher net
    # net.summary()
    net_auto.save("./JK_trained_model_net_auto", save_format="tf")
    print("net_auto已保存!")
    cmd = "s3cmd put -r ./JK_trained_model_net_auto " + args.model_output
    os.system(cmd)
    # 保存student net
    net_mlp.save("./JK_trained_model_net_mlp", save_format="tf")
    cmd = "s3cmd put -r ./JK_trained_model_net_mlp " + args.model_output
    os.system(cmd)
    print("模型保存完毕! {}".format(datetime.datetime.now()))


def test(args):
    tf.keras.backend.set_learning_phase(False)
    '''读取s3fs数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    # 装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net_auto"
    os.system(cmd)
    net = keras.models.load_model("./JK_trained_model_net_auto", custom_objects={'tf': tf}, compile=False)
    print("AutoIntCL Model weights load!")
    scaler = MinMaxScaler()
    print("开始训练归一化模型! {}".format(datetime.datetime.now()))
    #读取50个文件数据大约800万行数据来计算归一化各特征的max与min 数据在导入python前已经进行了shuffle操作
    for file in input_files[0:50]:
        data = pd.read_csv("s3://" + file, sep=',', header=None).astype('str')# 读取数据,第一列为roleid,第二列为clubid
        scaler.partial_fit(data.iloc[:,2::].astype("double").values)
    print("归一化模型训练完成! {}".format(datetime.datetime.now()))
    count = 0
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.read_csv("s3://" + file, sep=',', header=None).astype('str')# 读取数据,第一列为roleid,第二列为clubid
        id = data.iloc[:,0:2] #获取id编号(roleid与clubid)
        data = scaler.transform(data.iloc[:,2::].astype("double").values)
        data = tf.convert_to_tensor(data,dtype=tf.float32)
        #data = (data-tf.reduce_min(data,0))/tf.where(tf.equal(tf.reduce_max(data,0)-tf.reduce_min(data,0),0),1,tf.reduce_max(data,0)-tf.reduce_min(data,0))
        #result_test = np.zeros((data.shape[0],3)).astype("str") #三列roleid,clubid,predict_label
        '''count:当前的文件序号
        id:记录了玩家roleid与俱乐部的cluid,roleid与clubid都是字符串型'''
        _,label = net(data,training=False)
        label= label.numpy().astype("str")  # 获取预测的概率
        label = np.concatenate([id.values, label], axis=1)
        writer = S3Filewrite(args)
        writer.write(label.tolist(), args, count)
    print("已完成第{}个文件数据的推断! {}", count, datetime.datetime.now())

