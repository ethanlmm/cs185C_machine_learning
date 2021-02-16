im

root=os.path.dirname(__file__)
dataset_path=root+'/dataset'


dataset=load_dataset(dataset_path,'zbot-zeroaccess-labled.txt')
labled_zeroaccess_train=dataset["labled_zeroaccess_train"]
labled_zeroaccess_test=dataset["labled_zeroaccess_test"]
labled_zbot_train=dataset["labled_zbot_train"]
labled_zbot_test=dataset["labled_zbot_test"]



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
