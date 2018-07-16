from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pylab
import numpy as np
d = np.load('feature.npy').item()
X = d['feature']
labels = d['label']

data_pca_tsne = TSNE(n_components=2).fit_transform(X)
cls_num = -45
# pylab.figure()
pylab.scatter(data_pca_tsne[cls_num*5:, 0], data_pca_tsne[cls_num*5:, 1], 10, np.zeros_like(labels[cls_num*5:]))
pylab.scatter(data_pca_tsne[:cls_num*5, 0], data_pca_tsne[:cls_num*5, 1], 10, labels[:cls_num*5])
pylab.savefig('tsne.pdf')

