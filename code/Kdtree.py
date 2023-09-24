from scipy.spatial import KDTree
from collections import Counter

class Kdtree:
    def __init__(self, images):  # [vector, etiqueta]
        self.tree = KDTree()
        self.ind = []
        self.images = images

    def insert(self):
        self.tree = KDTree(self.images[:, 0], 128)

    def searchknn(self, image, k):
        dist, self.ind = self.tree.query(image, k=k)
        result = [self.images[x][1] for x in self.ind]
        self.plurality_voting(result)

    
    def plurality_voting(predictions):
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common(1)
        porcentaje = round(int(most_common[0][1]) / sum(vote_counts.values()), 4)

        print("Clasificación final:", most_common[0][0])
        print(f"Porcentaje: {porcentaje} %")