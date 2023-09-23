from skimage.io import imread, imshow
from scipy.spatial import KDTree
from vecImage import VecIMage

class Kdtree:
    def __init__(self, images):
        self.tree = KDTree()
        self.ind = []
        temp = []
        for carpeta in images:
            for imagen in carpeta:
                temp.append(VecIMage(imread(f'{images}/{carpeta}/{imagen}')).process(120),
                            str(f'{images}/{carpeta}/{imagen}'))
        self.images = temp

    def insert(self):
        self.tree = KDTree(self.images[:, 0], 128)

    def searchknn(self, image, k):
        dist, self.ind = self.tree.query(image, k=k)

    def recoverImgs(self, files):
        images = list()
        for i in self.ind:
            images.append(images[i][1])
        return images #rutas de las k imagenes mas cercanas
