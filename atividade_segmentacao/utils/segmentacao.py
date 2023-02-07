import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from skimage.filters import threshold_niblack, threshold_otsu


class Segmentacao:
    def __init__(self) -> None:
        pass


    def kmeans(self, img, n_clusters=2):
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(img.reshape(-1,1))
        result = km.labels_.reshape(img.shape)
        return result


    def niblack(self, img):
        result = img > threshold_niblack(img)
        return result
    

    def otsu(self, img):
        result = img > threshold_otsu(img)
        return result
    

    def agglomerative(self, img):
        ac = AgglomerativeClustering(n_clusters=2).fit(img.reshape(-1,1))
        result = ac.labels_.reshape(img.shape)
        return result
