import xml.etree.ElementTree as ET
import os
import  shutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def get_xyw(path):
    tree=ET.parse(path)
    root=tree.getroot()

    dates=root.findall('date')

    x_l=[]
    y_l=[]
    w_l=[]
    for date in dates:
        x=date.find('x').text
        y=date.find('y').text
        w=date.find('speed').text
        presure=date.find('presure').text
        x_l.append(int(x)/10)
        y_l.append(int(y)/10)
        w_l.append(int(w))

    x_data=np.array(x_l)
    y_data=np.array(y_l)
    w_data=np.array(w_l)

    return x_data,y_data,w_data
def get_five(path):
    x_data,y_data,w_data=get_xyw(path)

    wi=np.sqrt(w_data)
    X=np.sum(wi*x_data)/np.sum(wi)
    Y=np.sum(wi*y_data)/np.sum(wi)

    VX=np.sum(wi*(x_data-X)**2)/np.sum(wi)
    VY=np.sum(wi*(y_data-Y)**2)/np.sum(wi)
    VXY=np.sum(wi*(x_data-X)*(y_data-Y))/np.sum(wi)
    return X,Y,VX,VY,VXY

#get data from folder
filepath='tmp/sea'
datas=[]

tmp_file=[]
for filename in os.listdir(filepath):
    if filename.endswith('.xml'):
        path=os.path.join(filepath,filename)
        X,Y,VX,VY,VXY=get_five(path)
        datas.append([X,Y,VX,VY,VXY])
        tmp_file.append(filename)

# dataset get
datas=np.array(datas)
print('your datas.shape ',datas.shape)
# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(datas)

means=scaler.mean_
stds=scaler.scale_
print(f'means : {means} and std: {stds}')

# try to find the best K  which is in thr elbow area in the ploted picture
inertia = []
k_range=range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#  select a K and use k-means
k = 4
kmeans = KMeans(n_clusters=k, random_state=2)
labels = kmeans.fit_predict(X_scaled)

#concat an outputs

outputs = np.concatenate([datas, labels.reshape(-1, 1)], axis=1)

# use pca and tsne to visualize and down -demention
'''pca'''
pca = PCA(n_components=2, random_state=2)
X = pca.fit_transform(X_scaled)

'''tsne'''
# tsne = TSNE(n_components=2, random_state=2, perplexity=30, max_iter=5000)
# X = tsne.fit_transform(X_scaled)

# #  plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap= 'jet' ,s=6)
plt.title('K-means Clustering')
plt.colorbar()
plt.show()

##put into folder
# for classes, filename in zip(labels, tmp_file):
#     source_path = os.path.join(filepath, filename)
#     new_path = os.path.join('tmp/sea_class', f'{classes}', filename)
#     shutil.copy(source_path, new_path)
