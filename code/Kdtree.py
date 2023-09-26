from scipy.spatial import KDTree
from collections import Counter

class Kdtree:
    def __init__(self, images, etiquetas):  # [vector, etiqueta]
        self.tree = KDTree(images, 128)
        self.ind = []
        self.etiquetas = etiquetas
        self.images = images

    def searchknn(self, image, k):
        dist, self.ind = self.tree.query(image, k=k)
        result = [self.etiquetas[x] for x in self.ind]
        print(result)
        self.plurality_voting(result)

    
    def plurality_voting(self, predictions):
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common()
        porcentaje = round(int(most_common[0][1]) / sum(vote_counts.values()), 4)

        print(f"Clasificación final: {most_common[0][0]}")
        print(f"Porcentaje: {porcentaje*100} %") 