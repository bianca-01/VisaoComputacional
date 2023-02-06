import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters.rank import mean
from skimage.transform import resize
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, cohen_kappa_score
from utils.segmentacao import Segmentacao


class Image:
    def __init__(self, path):
        self.load(path)
        self.img_media = None
        self.img_segmentada = {}
        self.metricas = {}


    def load(self, path):   
        self.name = path.split('/')[2]

        self.img = imread(path)
        if len(self.img.shape) == 3:
            self.img = self.img[:,:,0]
        

    def load_mask(self, path):
        if self.name != path.split('/')[2]:
            raise Exception('A imagem e a máscara não são da mesma imagem')
       
        self.mask = imread(path)
        if len(self.mask.shape) == 3:
            self.mask = self.mask[:,:,0] 
        
    def show(self, img=None):
        if img is None:
            img = self.img
        plt.imshow(img, cmap='gray')
        plt.show()


    def filtro_media(self, k=10):
        filtro = np.ones((k,k)) 
        self.img_media = mean(self.img, selem=filtro)
    

    def segmentar(self, algoritmo, n_clusters=None):
        if algoritmo == 'kmeans':
            self.img_segmentada['kmeans'] = Segmentacao().kmeans(self.img_media, n_clusters)
        elif algoritmo == 'niblack':
            self.img_segmentada['niblack'] = Segmentacao().niblack(self.img_media)
        elif algoritmo == 'otsu':
            self.img_segmentada['otsu'] = Segmentacao().otsu(self.img_media)
        else:
            raise Exception('Algoritmo não reconhecido')
        

    def clusters(self):
        cluster1 = self.img_segmentada['kmeans'] == 0
        cluster2 = self.img_segmentada['kmeans'] == 1
        
        self.background = np.zeros_like(self.img_media)
        self.background[cluster1] = self.img_media[cluster1]

        self.foreground = np.zeros_like(self.img_media)
        self.foreground[cluster2] = self.img_media[cluster2]


    def save(self, path, img):
        plt.imsave(path, img, cmap='gray')


    def redimensionar(self, porcentagem=0.8):
        h, w = self.img.shape
        nh, nw = int(h*porcentagem), int(w*porcentagem)
        self.img_res = resize(self.img, (nh,nw), anti_aliasing=True)
        

    def calcular_metricas(self):
        self.metricas['accuracy'] = accuracy_score(self.mask.ravel(), self.img_segmentada['kmeans'].ravel())
        self.metricas['kappa'] = cohen_kappa_score(self.mask.ravel(), self.img_segmentada['kmeans'].ravel())
        self.metricas['jaccard'] = jaccard_score(self.mask.ravel(), self.img_segmentada['kmeans'].ravel(), average='micro')
        
        print('Accuracy:', self.metricas['accuracy'])
        print('Kappa:', self.metricas['kappa'])
        print('Jaccard:', self.metricas['jaccard'])