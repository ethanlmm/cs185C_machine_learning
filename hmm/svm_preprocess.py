import os
from util import *
from hmm import hmm
root=os.path.dirname(__file__)
data_path=root+'/data'
model_path=root+'/model'
dataset_path=root+'/dataset'







dataset=load_dataset(dataset_path,'zbot-zeroaccess-1000-10000.txt')

alphabet=dataset['alphabet']
zeroaccess_dataset=dataset['zeroaccess_dataset']
zbot_dataset=dataset["zbot_dataset"]

states=['A','B']

zeroaccess_hmm=hmm('zeroaccess',states,alphabet)
zeroaccess_hmm.load_model(model_path,'zeroaccess_best.txt')

zbot_hmm=hmm('zbot',states,alphabet)
zbot_hmm.load_model(model_path,'zbot_best.txt')


dataset={}
zeroaccess_index=0
zbot_index=1

labled_zeroaccess_train=[]
labled_zeroaccess_test=[]
labled_zbot_train=[]
labled_zbot_test=[]

print("data loaded")
for x in zeroaccess_dataset[0:800]:
    score1 = zeroaccess_hmm.score(x)
    score2 = zbot_hmm.score(x)
    labled_zeroaccess_train.append([zeroaccess_index,score1,score2])

for x in zeroaccess_dataset[800:]:
    score1 = zeroaccess_hmm.score(x)
    score2 = zbot_hmm.score(x)
    labled_zeroaccess_test.append([zeroaccess_index,score1,score2])

for x in zbot_dataset[0:800]:
    score1=zeroaccess_hmm.score(x)
    score2= zbot_hmm.score(x)
    labled_zbot_train.append([zbot_index,score1,score2])

for x in zbot_dataset[800:]:
    score1=zeroaccess_hmm.score(x)
    score2= zbot_hmm.score(x)
    labled_zbot_test.append([zbot_index,score1,score2])

dataset["labled_zeroaccess_train"]=labled_zeroaccess_train
dataset["labled_zeroaccess_test"]=labled_zeroaccess_test
dataset["labled_zbot_train"]=labled_zbot_train
dataset["labled_zbot_test"]=labled_zbot_test

save_dataset(dataset_path,'zbot-zeroaccess-labled.txt',dataset)



