from ast import Store
from math import ceil
from random import sample
from sys import argv
import pandas as pd
import glob
import csv
# from faker import Faker
import os
import argparse
import numpy as np


def info_reader(csv_name):
    csv_read = pd.read_csv(csv_name, sep=",", encoding="utf-8")
    csv_read.info()


def split_writer(origin_name, out_name, split_num):
    origin_csv = pd.read_csv(origin_name, sep=",", encoding="utf-8")
    i = 0
    num = 0
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    split_size = ceil(origin_csv.shape[0] / split_num)
    for i in range(0, split_num):
        out_file = out_name+"/split"+str(i+1)+".csv"
        print("Splitting: "+out_file)
        if i != split_num:
            out_lines = origin_csv.iloc[num: num+split_size]
        else:
            out_lines = origin_csv.iloc[num:]
        out_lines.to_csv(out_file, index=False)
        num += split_size

def point_split(origin_name,out_name,split_point):
    origin_csv = pd.read_csv(origin_name, sep=",", encoding="utf-8")
    i = 0
    num = 0
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    out_file = out_name+"/split"+str(split_point)+".csv"
    print("Preparing "+str(split_point)+" lines from original dataset...")
    out_lines = origin_csv.iloc[0:split_point]
    out_lines_rest = origin_csv.iloc[split_point+1:]
    out_lines.to_csv(out_file, index=False)
    out_lines_rest.to_csv(out_name+"/rest.csv", index=False)
    


def combine_writer(folder_name):
    all_files = glob.glob(folder_name + "/*.csv")
    all_dataframe = []
    for file in all_files:
        dataframe = pd.read_csv(file)
        all_dataframe.append(dataframe)
    dataframe_concat = pd.concat(all_dataframe, axis=0, ignore_index=True, sort=False)
    print("Combining: " + folder_name+"/ph_combined.csv")
    dataframe_concat.to_csv(folder_name+"/ph_combined.csv", index=False, encoding="utf-8")


def fake_data_maker(line):
    # 实例化faker库,zh-CN允许生成中文
    fake = Faker('zh-CN')

    # open()打开一个csv文件,以a+追加方式打开,encoding='utf-8'允许插入中文
    with open('data.csv', 'a+', encoding='utf-8') as csvfile:
        # 设置字段
        fieldnames = ['id', 'name', 'phone', 'address', 'python']
        # DictWriter以字典形式写入csv文件
        print("Genarating fake data…")
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 调用writerheader()先写入头信息
        writer.writeheader()
        for i in range(1, line):
            ctx = {
                'id': i,
                'name': fake.name(),
                'phone': fake.phone_number(),
                'address': fake.address(),
                'python': fake.sha256(raw_output=False)
            }
            print(ctx)

            # 调用writerows方法写入多行,方法使用:writerows([])
            writer.writerows([ctx])


def random_sampler(dataset, sample_frac, out_name):
    print("Sampling "+str(sample_frac*100)+"% data…")
    dataset_df = pd.read_csv(dataset, sep=",", encoding="utf-8")
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    dataset_df.sample(frac=sample_frac, replace="False", axis=0).to_csv(out_name+"/sampled.csv", index=False, encoding="utf-8")

def sequential_spliter(dataset, split_frac, out_name):
    print("Spliting "+str(split_frac*100)+"% data…")
    origin_csv = pd.read_csv(dataset, sep=",", encoding="utf-8")
    i = 0
    num = 0
    split_point = int(split_frac*len(origin_csv.index))
    print("Total:",len(origin_csv.index),"Split at:",split_point)
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    out_file = out_name+"/original.csv"
    # print("Preparing "+str(split_point)+" lines from original dataset...")
    out_lines = origin_csv.iloc[0:split_point]
    print("Splitting rest from ",split_point+1)
    out_lines_rest = origin_csv.iloc[split_point:]
    out_lines.to_csv(out_file, index=False)
    out_lines_rest.to_csv(out_name+"/incremental.csv", index=False)

def range_spliter(dataset,frac1,frac2,out_name):
    print("Spliting ",frac1*100,"% to",frac2*100,"% data…")
    origin_csv = pd.read_csv(dataset, sep=",", encoding="utf-8")
    i = 0
    num = 0
    point1 = int(frac1*len(origin_csv.index))
    point2 = int(frac2*len(origin_csv.index))
    print("Total:",len(origin_csv.index),"Split at:",point1," ",point2)
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    out_file1 = out_name+"/original.csv"
    # print("Preparing "+str(split_point)+" lines from original dataset...")
    out_lines1 = origin_csv.iloc[0:point1]
    print("Splitting rest from",point1+1,"to",point2)
    out_lines2 = origin_csv.iloc[point1:point2]
    out_lines1.to_csv(out_file1, index=False)
    out_lines2.to_csv(out_name+"/incremental.csv", index=False)

def random_sampler_entry(dataset, sample_frac, out_name):
    print( "Sampling "+str(sample_frac)+"\% entries…")
    dataset_df = pd.read_csv(dataset, sep=",", encoding="utf-8")
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    
    sampled_df = dataset_df.sample(frac=sample_frac, replace="False", axis=0)
    remaining_df = dataset_df.drop(labels= sampled_df.index)
    
    out_file = out_name+"/original.csv"
    sampled_df.to_csv(out_file, index=False, encoding="utf-8")
    remaining_df.to_csv(out_name+"/incremental.csv", index=False)

def random_shuffle(dataset,out_name):
    dataset_df = pd.read_csv(dataset, sep=",", encoding="utf-8")
    sampler = np.random.permutation(dataset_df.shape[0])
    random_df = dataset_df.take(sampler)
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    random_df.to_csv(out_name+"/random.csv", index=False)






def main(argv):
    csvparse=argparse.ArgumentParser(description="Csv Operating Tools")
    csvparse.add_argument('operation', choices=['sample', 'split', 'combine', 'genfake',"point","sequential","range","sample_entry","shuffle"])
    csvparse.add_argument('-i','--ifile',default="./data.csv",help="Input file path")
    csvparse.add_argument('-o','--ofile',default=".",help="Output file path and prefix")
    csvparse.add_argument('-f','--fragment',default="0.25",type=float,help="Sampling fragment, default 0.25")
    csvparse.add_argument('--splitnum', default=4,type=int, help="Split num, default 4")
    csvparse.add_argument('-l','--line', default=1000, type=int, help="Fake data lines, default 1000")
    csvparse.add_argument('--point', default=100000,type=int, help="Split point, default 100000")
    csvparse.add_argument('--upper',default=0,type=float,help="Using for upper range")
    # csvparse.add_argument('--si',default=10000, type=int, help="Randomly sample by entry count")
    args = csvparse.parse_args()
    print("=======CSV Opreating Tools=======")
    print("Mode:"+args.operation)
    if args.operation == "sample":
        random_sampler(args.ifile, args.fragment, args.ofile)
    elif args.operation == "split":
        split_writer(args.ifile, args.ofile, args.splitnum)
    elif args.operation == "combine":
        combine_writer(args.ifile)
    elif args.operation == "genfake":
        fake_data_maker(line)
    elif args.operation == "point":
        point_split(args.ifile, args.ofile, args.point)
    elif args.operation == "sequential":
        sequential_spliter(args.ifile, args.fragment, args.ofile)
    elif args.operation == "range":
        range_spliter(args.ifile, args.fragment, args.upper, args.ofile)
    elif args.operation == "sample_entry":
        random_sampler_entry(args.ifile, args.fragment, args.ofile)
    elif args.operation == "shuffle":
        random_shuffle(args.ifile, args.ofile)
    

if __name__ == "__main__":
    main(argv)