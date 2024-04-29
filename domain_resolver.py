import itertools
import pandas as pd
import json

def DomainResolver(dataset):
    data_pd = pd.read_csv(dataset,header=0)
    domains = {}
    print("Collecting domains...")
    for column in data_pd.columns:
        data_list = list(data_pd.loc[:,column])
        domain_max = max(data_list)
        domain_min = min(data_list)
        domain_size = domain_max - domain_min + 1
        domains[column] = domain_size
    return domains

def DomainComplier(dataset):
    data_pd = pd.read_csv(dataset,header=0)
    domain_dict_all = {}
    for column in data_pd.columns:
        data_list = list(data_pd.loc[:,column])
        if not isinstance(data_list[0], str):
            print("Skipping "+column)
            continue
        else:
            print("Start to convert "+column)
            domain_str_list = []
            domain_data_int = []
            domain_dict = {}
            i = 0
            for line in data_list:
                if line not in domain_str_list:
                    domain_str_list.append(line)
                    domain_dict[line] = i
                    i+=1
                domain_data_int.append(domain_dict[line])
            data_pd.loc[:,column] = domain_data_int
            domain_dict_all[column] = domain_dict
    data_pd.to_csv("./data/adult_modified.csv",index=None)
    print(domain_dict_all)
    return domain_dict_all

def main():
    dataname = "adult"
    dataset = "./data/"+dataname+"/"+dataname+".csv"
    domainJson = "./data/"+dataname+"/"+dataname+".json"
    # domains = DomainComplier(dataset)
    #json_file = open("./data/adult-domain-codebook.json","a")
    #json.dump(domains,json_file)
    print("Working on:" + dataset)
    domains = DomainResolver(dataset)
    print("Writing to file...")
    json_file = open(domainJson,"a") 
    json.dump(domains,json_file)



main()