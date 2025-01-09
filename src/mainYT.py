import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = './images/IMAG0224.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Filterbank parameter
width, height = image.shape                     # Breite und Hoehe des Bildes
# ksize = int(np.floor(width*height*0.001))             # 1% der bilddimension#
ksize = 400
gamma = 0.5                                       # Skalierung
psi = 0                                         # Phasenverschiebung
orientations = 1*np.pi/2
lambd = 1*np.pi/4                          # Manuelle Wellenlängen
k_sigma = 2.0                                 # Faktor für Standardabweichung

kernel = cv2.getGaborKernel((ksize,ksize), k_sigma, orientations, lambd, gamma, psi)

# Speichere Gabor-Filter als bilder
# plt.imshow(kernel, cmap='gray')
# plt.title(f"Gabor Filter")
# plt.axis('off')
# plt.show()
# plt.savefig(f"./out/test_kernel.png", dpi=300, bbox_inches='tight', transparent=False)

# Anwenden der Filterbank auf das Bild und Visualisierung der Ergebnisse
filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)

# Visualisierung der gefilterten Bilder
plt.figure(figsize=(8,8))
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.tight_layout()

plt.show()

