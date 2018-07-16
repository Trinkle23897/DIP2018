import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def one_hot(x):
	return x
	label = np.zeros((x.shape[0], x.max()+1), dtype=np.uint8)
	for i in range(x.shape[0]):
		label[i, x[i]] = 1
	return label

d=np.load('feature_500.npy').item()
train_feature = d[u'feature']
train_label = one_hot(d[u'label'])

d = np.load('feature_all.npy').item()
test_feature = d[u'feature']
test_label = one_hot(d[u'label'])

xg_train = xgb.DMatrix(train_feature, label=train_label)
xg_test = xgb.DMatrix(test_feature, label=test_label)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 50
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
print('predicting, classification error=%f' % (sum(int(pred[i]) != test_label[i] for i in range(len(test_label))) / float(len(test_label))))
os
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict(xg_test).reshape(test_label.shape[0], param['num_class'])
ylabel = np.argmax(yprob, axis=1)
print('predicting, classification error=%f' % (sum(int(ylabel[i]) != test_label[i] for i in range(len(test_label))) / float(len(test_label))))
