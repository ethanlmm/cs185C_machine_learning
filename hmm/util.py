import os
import re
import json


def read(path):
    with open(path,'r') as f:

        return f.read().splitlines()


def list_path(path, type=None):
    if type is None: return [os.path.join(path, name) for name in os.listdir(path)]
    paths = []
    for name in os.listdir(path):
        if (re.match(type, name)):
            paths.append(os.path.join(path, name))
    return paths

def list_rec(path,regex=None):
    listOfPath=[]
    for full_path in [os.path.join(path, name) for name in os.listdir(path)]:
        if os.path.isdir(full_path):
            sub_folder=list_rec(full_path, regex)
            if sub_folder!=[]:
                listOfPath.append(sub_folder)
        elif(regex == None or re.match(regex,full_path)):
            listOfPath.append(os.path.abspath(full_path))
    return listOfPath

def save_dataset(path,name,obj):
    with open(os.path.abspath(path+"/"+name), 'w') as f:
        json.dump(obj, f)

def load_dataset(path,name):
    with open(os.path.abspath(path+"/"+name),'r') as f:
        return json.load(f)