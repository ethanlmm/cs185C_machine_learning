import os
from util import *

import numpy as np
from sklearn import svm
from joblib import dump, load
import matplotlib.pyplot as plt


root=os.path.dirname(__file__)
dataset_path=root+'/dataset'


dataset=load_dataset(dataset_path,'zbot-zeroaccess-labled.txt')
labled_zeroaccess_train=dataset["labled_zeroaccess_train"]
labled_zeroaccess_test=dataset["labled_zeroaccess_test"]
labled_zbot_train=dataset["labled_zbot_train"]
labled_zbot_test=dataset["labled_zbot_test"]

avg_zeroaccess_to_zeroaccess_train=sum([x[1] for x in labled_zeroaccess_train])/len(labled_zeroaccess_train)
avg_zeroaccess_to_zbot_train=sum([x[2] for x in labled_zeroaccess_train])/len(labled_zeroaccess_train)

avg_zbot_to_zeroaccess_train=sum([x[1] for x in labled_zbot_train])/len(labled_zbot_train)
avg_zbot_to_zbot_train=sum([x[2] for x in labled_zbot_train])/len(labled_zbot_train)
print(avg_zeroaccess_to_zeroaccess_train)
print(avg_zeroaccess_to_zbot_train)
print(avg_zbot_to_zeroaccess_train)
print(avg_zbot_to_zbot_train)

labled_train=[]
labled_train.extend(labled_zeroaccess_train)
labled_train.extend(labled_zbot_train)
labled_train=np.array(labled_train)

labled_zeroaccess_test=np.array(labled_zeroaccess_test)
labled_zbot_test=np.array(labled_zbot_test)


clf = svm.SVC(kernel='linear', C=1)
clf.fit(labled_train[:,1:],labled_train[:,0])
dump(clf, 'zbot-zeroaccess.joblib')
predict_label_zeroaccess=clf.predict(labled_zeroaccess_test[:,1:])
predict_label_zbot=clf.predict(labled_zbot_test[:,1:])

print(predict_label_zeroaccess)
print((len(predict_label_zeroaccess)-sum(predict_label_zeroaccess))/len(predict_label_zeroaccess))
print(predict_label_zbot)
print(sum(predict_label_zbot)/200)
plt.scatter(labled_train[:,1], labled_train[:,2],c=labled_train[:,0])
plt.show()
