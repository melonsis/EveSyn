import pandas as pd
import numpy as np
import os
from mbi import Dataset, Domain
import time

def log_init(attr_name, csv_name = None, folder = None):
    if folder == None:
        folder = "./Results/"
    if csv_name == None: # Init file
        now_time = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
        csv_name = "result"+now_time+".csv"
    csv_file = open(folder+csv_name, "w", encoding="utf-8")
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
    if (csv_name == None) or (not os.path.exists(folder+csv_name)): #Detecting file
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
    if txt_name == None: # Init file
        now_time = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
        txt_name = "info"+now_time+".txt"
    txt_file = open(folder+txt_name, "a", encoding="utf-8")
    txt_file.write(info_str+"\n")
    txt_file.close()
    return [txt_name, folder]

    

    

