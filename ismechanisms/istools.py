import pandas as pd
import numpy as np
import os
from mbi import Dataset, Domain
import time

def random_delete(data, percent):
    data_pd = pd.read_csv(data)
    num_drop = int(len(data_pd) * percent)
    rows_drop = data_pd.sample(n=num_drop)
    return data_pd.drop(rows_drop.index).reset_index(drop=True)

def increment_fetcher(data, syn, incre_num, out_name):
    syn_pd = pd.read_csv(syn)
    data_pd = pd.read_csv(data)

    split_point = len(data_pd) - len(syn_pd)
    if split_point <= 0:
        print("Error: Bad fetch point")
        return -1
    split_end = split_point+incre_num
    if not os.path.exists(out_name):
        os.makedirs(out_name)
    out_file = out_name+"/incremental.csv"

    print("Preparing "+str(split_point)+" lines from original dataset...")
    out_lines = data_pd.iloc[split_point+1:split_end]
    out_lines_rest = data_pd.iloc[split_end+1:]
    
    out_lines.to_csv(out_file, index=False)
    out_lines_rest.to_csv(out_name+"/rest.csv", index=False)
    return 0

def log_init(attr_name, csv_name = None, folder = None):
    if folder == None:
        folder = "./Results/"
    if csv_name == None: #新建文件
        now_time = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
        csv_name = "result"+now_time+".csv"
    csv_file = open(folder+csv_name, "w", encoding="utf-8") #无论是否存在，init都会直接洗掉同名文件
    out_str = ""
    for name in attr_name:
        out_str = out_str+","+name
    out_str = out_str[1:] + "\n"# Remove rebundant quote
    csv_file.write(out_str)
    csv_file.close()
    return [csv_name,folder]

def log_append(attr_value, csv_name = None, folder = None):
    if folder == None:
        folder = "./Results/"
    if (csv_name == None) or (not os.path.exists(folder+csv_name)): #探测存在性
       print("Error: You need use csv_init to init a csv log file first")
       return -1
    csv_file = open(folder+csv_name, "a", encoding="utf-8")
    out_str = ""
    for value in attr_value: 
        out_str = out_str+","+str(value)
    out_str = out_str[1:] + "\n"# Remove rebundant quote
    csv_file.write(out_str)
    csv_file.close()
    return [csv_name,folder]

def info_logger(info_str, txt_name = None, folder = None):
    if folder == None:
        folder = "./Information/"
    if txt_name == None: #新建文件
        now_time = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
        txt_name = "info"+now_time+".txt"
    txt_file = open(folder+txt_name, "a", encoding="utf-8")
    txt_file.write(info_str+"\n")
    txt_file.close()
    return [txt_name, folder]

    

    

