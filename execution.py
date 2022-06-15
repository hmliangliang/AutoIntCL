#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 11:11
# @Author  : Liangliang
# @File    : execution.py
# @Software: PyCharm
import argparse
import AutoIntCL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--flags", help="指示BatchNormalization的training参数(True or False)", type=bool, default=True)
    parser.add_argument("--lr", help="学习率", type=float, default=0.0001)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=20)
    parser.add_argument("--thread_num", help="多线程编程的线程数目", type=int, default=1000)
    parser.add_argument("--process_num", help="多进程编程的线程数目", type=int, default=1)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=150)
    parser.add_argument("--neg_num", help="负采样的数目", type=int, default=5)
    parser.add_argument("--head_num", help="多头注意力机制的数目", type=int, default=8)
    parser.add_argument("--epsion", help="噪声数据的半径", type=float, default=1.5)
    parser.add_argument("--alpha", help="loss函数中的alpha参数", type=float, default=1.5)
    parser.add_argument("--beta", help="loss函数中的beta参数", type=float, default=1)
    parser.add_argument("--gamma", help="loss函数中的gamma参数", type=float, default=2)
    parser.add_argument("--r", help="负样本中混合正样本的比例", type=float, default=0.5)
    parser.add_argument("--bpr", help="bpr的loss系数", type=float, default=10)
    parser.add_argument("--batch_size", help="每个batch中样本的数目", type=int, default=1024)
    parser.add_argument("--hidden_num1", help="隐含层神经元的数目", type=int, default=150)
    parser.add_argument("--hidden_num2", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="隐含层神经元的数目", type=int, default=64)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=15000000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置", type=str, default='s3://JK/models20220602/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        args.flags = True
        AutoIntCL.train(args)
    elif args.env == "test":
        args.flags = False
        AutoIntCL.test(args)
    else:
        print("输入的环境参数错误,env只能为train或test!")