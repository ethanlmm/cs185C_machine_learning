import os
from util import *
from hmm import hmm
root=os.path.dirname(__file__)
data_path=root+'/data'
model_path=root+'/model'
txt='.*\.txt'
zeroaccess='.*zeroaccess'
zbot='.*zbot'
zeroaccess_zbot='(.*zbot|.*zeroaccess)'
dataset_path=root+'/dataset'


dataset=load_dataset(dataset_path,'zbot-zeroaccess-1000-10000.txt')
alphabet=dataset['alphabet']
zeroaccess_dataset=dataset['zeroaccess_dataset']
zeroaccess_dataset=zeroaccess_dataset[0:800]
zbot_dataset=dataset["zbot_dataset"]
zbot_dataset=zbot_dataset[0:800]

states=['A','B']
print('data loaded')

zbot_hmm=hmm('zbot',states,alphabet)
zbot_hmm.fit(100,zbot_dataset)
zbot_hmm.save_model(model_path)

zeroaccess_hmm=hmm('zeroaccess',states,alphabet)
zeroaccess_hmm.fit(100,zeroaccess_dataset)
zeroaccess_hmm.save_model(model_path)