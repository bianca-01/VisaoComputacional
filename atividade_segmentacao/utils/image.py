import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters.rank import mean
from skimage.transform import resize
from sklearn.metrics import accuracy_score,  jaccard_score, f1_score
from utils.segmentacao import Segmentacao


class Image:
    def __init__(self, path):
        self.name = None
        self.img = None
        self.mask = None
        self.mask_res = None
        self.img_media = None
        self.img_res = None
        self.background = None
        self.foreground = None
        self.algoritmos = None
        self.img_segmentada = {}
        self.metricas = {}
        self.load(path)


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


    def filtro_media(self, k=5):
        filtro = np.ones((k,k)) 
        self.img_media = mean(self.img, selem=filtro)
    

    def segmentar(self, algoritmo, n_clusters=None):
        if algoritmo == 'kmeans':
            self.img_segmentada['kmeans'] = Segmentacao().kmeans(self.img_media, n_clusters)
        
        elif algoritmo == 'niblack':
            self.img_segmentada['niblack'] = Segmentacao().niblack(self.img_media)
        
        elif algoritmo == 'otsu':
            self.img_segmentada['otsu'] = Segmentacao().otsu(self.img_media)
        
        elif algoritmo == 'agglomerative':
            self.img_segmentada['agglomerative'] = Segmentacao().agglomerative(self.img_res)
        
        else:
            raise Exception('Algoritmo não reconhecido')
        

    def clusters(self):
        cluster1 = self.img_segmentada['kmeans'] == 0
        cluster2 = self.img_segmentada['kmeans'] == 1
        
        self.background = np.zeros_like(self.img)
        self.background[cluster1] = self.img[cluster1]

        self.foreground = np.zeros_like(self.img)
        self.foreground[cluster2] = self.img[cluster2]


    def save(self, path, img):
        plt.imsave(path, img, cmap='gray')


    def redimensionar(self, porcentagem=0.6):
        h, w = self.img.shape
        p = 1-porcentagem
        nh, nw = int(h*p), int(w*p)
        self.img_res = resize(self.img, (nh,nw), anti_aliasing=True)

    
    def redimensionar_mask(self, porcentagem=0.6):
        h, w = self.mask.shape
        p = 1-porcentagem
        nh, nw = int(h*p), int(w*p)
        self.mask_res = resize(self.mask, (nh,nw), anti_aliasing=True)
        

    def calcular_metricas(self):
        self.algoritmos = sorted(list(self.img_segmentada.keys()))
        for algoritmo in self.algoritmos:
            m = {}

            if algoritmo != 'agglomerative':
                m['accuracy'] = accuracy_score(self.mask.ravel(), self.img_segmentada[algoritmo].ravel())
                m['jaccard'] = jaccard_score(self.mask.ravel(), self.img_segmentada[algoritmo].ravel(), average='weighted')
                m['f1'] = f1_score(self.mask.ravel(), self.img_segmentada[algoritmo].ravel(), average='weighted')

            else:
                m['accuracy'] = accuracy_score(np.argmax(self.mask_res, axis=1), np.argmax(self.img_segmentada[algoritmo], axis=1))
                m['jaccard'] = jaccard_score(np.argmax(self.mask_res, axis=1), np.argmax(self.img_segmentada[algoritmo], axis=1), average='weighted')
                m['f1'] = f1_score(np.argmax(self.mask_res, axis=1), np.argmax(self.img_segmentada[algoritmo], axis=1), average='weighted')

            self.metricas[algoritmo] = m
