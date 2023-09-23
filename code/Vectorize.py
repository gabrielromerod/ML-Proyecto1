import pywt
import pywt.data


class VecIMage:
    def __init__(self, image):
        self.image = image

    def process(self, cortes):
        LL = self.image
        for i in range(cortes):
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        return LL.flatten()