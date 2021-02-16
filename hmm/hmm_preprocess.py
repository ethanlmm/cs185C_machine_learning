import os
from util import *
from hmm import hmm
import json
root=os.path.dirname(__file__)
data_path=root+'/data'
model_path=root+'/model'
dataset_path=root+'/dataset'
txt='.*\.txt'
zeroaccess='.*zeroaccess'
zbot='.*zbot'
zeroaccess_zbot='(.*zbot|.*zeroaccess)'

target_path_collection=list_rec(data_path,zeroaccess_zbot)






opcodes={}
zbot_dataset=[]
for path in target_path_collection[0]:
    data=read(path)
    zbot_dataset.append(data)
    for opcode in data:
        if(opcode in opcodes):
            opcodes[opcode]+=1
        else:opcodes[opcode]=1

zeroaccess_dataset = []
for path in target_path_collection[1]:
    data = read(path)
    zeroaccess_dataset.append(data)
    for opcode in data:
        if (opcode in opcodes):
            opcodes[opcode] += 1
        else:
            opcodes[opcode] = 1


care=filter(lambda x:x[1]>10000,opcodes.items())
care_name,y=zip(*care)
alphabet=list(care_name)

zbot_dataset=zbot_dataset[0:1000]
zeroaccess_dataset=zeroaccess_dataset[0:1000]


for i in range(len(zeroaccess_dataset)):
    zeroaccess_dataset[i]=list(map(lambda x:x if x in alphabet else ' ',zeroaccess_dataset[i]))

for i in range(len(zbot_dataset)):
    zbot_dataset[i]=list(map(lambda x:x if x in alphabet else ' ',zbot_dataset[i]))

alphabet.append(' ')
print("data loaded")



dataset={}
dataset["alphabet"]=alphabet
dataset["zeroaccess_dataset"]=zeroaccess_dataset
dataset["zbot_dataset"]=zbot_dataset
save_dataset(dataset_path,"zbot-zeroaccess-1000-10000.txt",dataset)

