import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = '../images/combine.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def gabor_filter(ksize, sigma, theta, lambd, gamma, psi):
    """
    Erstelle Gabor-Filter mit folgenden Parametern:

    :param ksize: Groesse des Filterkernels
    :param sigma: Standardabweichung der Gauss-Verteilung
    :param lambd: Wellenlaenge der Sinuswelle
    :param gamma: Skalierung
    :param psi: Phasenverschiebung
    :return: Gabor-Filter
    """
    return cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi)

# Erstelle eine Filterbank mit verschiedenen Orientierungen und Wellenlaengen
ksize = 21          # Groesse Filterkernel
sigma = 5.0         # Standardabweichung
theta = np.pi/4     # 45 Grad
lambd = 10.0        # Wellenlaenge
gamma = 0.5         # Skalierung
psi = 0             # Phasenverschiebung


# Erzeuge filter
gabor_kernel = gabor_filter(ksize, sigma, theta, lambd, gamma, psi)

filtered_image = cv2.filter2D(image, -1, gabor_kernel)

plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)                # Originalbild
plt.imshow(image, cmap='gray')
plt.title('Originalbild')
plt.axis('off')

plt.subplot(1, 3, 2)                # Gabor-Filter
plt.imshow(gabor_kernel, cmap='gray')
plt.title('Gabor-Filter')
plt.axis('off')

plt.subplot(1, 3, 3)                # Gefiltertes Bild
plt.imshow(filtered_image, cmap='gray')
plt.title('Bild mit Gabor-Filter')
plt.axis('off')

plt.show()
