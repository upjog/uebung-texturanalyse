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

def create_gabor_filterbank(ksize, sigma, gamma, psi, orientations, wavelengths):
    filterbank = {}
    for theta in orientations:
        for wavelength in wavelengths:
            key = f"theta_{round(theta, 3)}_wavelength_{round(wavelength, 3)}"
            gabor_kernel = gabor_filter(ksize, sigma, theta, wavelength, gamma, psi)
            filterbank[key] = gabor_kernel
    return filterbank

# Filterbank parameter
ksize = 21          # Groesse Filterkernel
sigma = 5.0         # Standardabweichung
gamma = 0.5         # Skalierung
psi = 0             # Phasenverschiebung
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 Orientierungen
wavelengths = [1, 5, 10, 20]   # 3 verschiedene Wellenlängen

# Erzeuge Gabor-Filterbank
filterbank = create_gabor_filterbank(ksize, sigma, gamma, psi, orientations, wavelengths)

# Speichere Gabor-Filter als bilder
for key, gabor_kernel in filterbank.items():
    plt.imshow(gabor_kernel, cmap='gray')
    plt.title(f"Gabor Filter {key}")
    plt.axis('off')
    plt.savefig(f"../out/{key}_gabor_kernel.png", dpi=300, bbox_inches='tight', transparent=False)


# Anwenden der Filterbank auf das Bild und Visualisierung der Ergebnisse
filtered_images = []
for key, gabor_kernel in filterbank.items():
    filtered_image = cv2.filter2D(image, -1, gabor_kernel)
    filtered_images.append(filtered_image)

# Visualisierung der gefilterten Bilder
plt.figure(figsize=(15,10))
for i, filtered_image in enumerate(filtered_images):
    plt.subplot(len(orientations), len(wavelengths), i+1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Filtered with {list(filterbank.keys())[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Segmentierung: Schwellenwert-basierte Segmentierung für das Baum-Hintergrund-Bild
# Zum Beispiel mit Otsu's Schwellenwertverfahren für jedes gefilterte Bild
thresholded_images = []
for filtered_image in filtered_images:
    _, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_images.append(thresholded_image)

# Visualisierung der segmentierten Bilder
plt.figure(figsize=(15, 10))
for i, thresholded_image in enumerate(thresholded_images):
    plt.subplot(len(orientations), len(wavelengths), i+1)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f"Segmented {list(filterbank.keys())[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
