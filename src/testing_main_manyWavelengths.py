import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # Importiere StandardScaler

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

def create_gabor_filterbank(orientations, wavelengths, gamma, psi, k_sigma=1.0, ksize_factor=3):
    """
    Erstelle eine Gabor-Filterbank mit variablen Wellenlängen und Orientierungen.
    :param orientations: Liste der Filterorientierungen (z.B. [0, np.pi/4, np.pi/2, 3*np.pi/4])
    :param wavelengths: Liste der Wellenlängen (z.B. [4, 6, 8, 10, 12, 14, 16])
    :param k_sigma: Multiplikationsfaktor für die Standardabweichung
    :param ksize_factor: Faktor zur Berechnung der Filtergröße basierend auf der Wellenlänge
    :return: Filterbank als Dictionary
    """
    filterbank = {}
    for theta in orientations:
        for wavelength in wavelengths:
            sigma = k_sigma * wavelength  # Standardabweichung in Bezug auf Wellenlänge
            ksize = int(ksize_factor * wavelength)  # Berechne die Filtergröße
            key = f"theta_{round(theta, 3)}_wavelength_{round(wavelength, 3)}"
            gabor_kernel = gabor_filter(ksize, sigma, theta, wavelength, gamma=1, psi=0)
            filterbank[key] = gabor_kernel
    return filterbank

# Filterbank parameter
gamma = 1                                       # Skalierung
psi = 0                                         # Phasenverschiebung
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 Orientierungen
wavelengths = [4,8,16,32,64,128,256]                           # Verschiedene Wellenlängen
k_sigma = 1.0                                   # Faktor für Standardabweichung
ksize_factor = 3                                # Faktor für Filtergrösse (abhängig von Wellenlänge)

# Erzeuge Gabor-Filterbank
filterbank = create_gabor_filterbank(orientations, wavelengths, gamma, psi,  k_sigma, ksize_factor)

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
    plt.title(f"{list(filterbank.keys())[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"../out/filterbank_applied_to_image_k_sigma{k_sigma}.png", dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Feature-Vektoren erstellen für jedes Pixel: Gabor-Filter-Energie + Geometrische Informationen
features = []
rows, cols = image.shape
for i in range(rows):
    for j in range(cols):
        feature_vector = []
        # Extrahiere die Gabor-Filter-Antworten für das Pixel (i, j)
        for filtered_image in filtered_images:
            feature_vector.append(filtered_image[i, j])
        # Füge geometrische Informationen hinzu (Reihe, Spalte)
        feature_vector.append(i)
        feature_vector.append(j)
        features.append(feature_vector)

# Umwandlung der Feature-Liste in ein NumPy-Array
features = np.array(features)

# Normalisierung der Features: StandardScaler (Z-Standardisierung)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Überprüfen der Form der Features (sollte (Anzahl der Pixel, Anzahl der Merkmale) sein)
print("Form der Feature-Daten:", features.shape)

# K-Means Clustering
kmeans = KMeans(n_clusters=8, random_state=42)  # Anzahl der Cluster anpassen
#kmeans.fit(features_normalized)
kmeans.fit(features)

# Kategorisierung der Pixel basierend auf den Clustern
segmented_image = kmeans.labels_.reshape(image.shape)

# Visualisierung der segmentierten Ergebnisse
plt.figure(figsize=(10, 8))
plt.imshow(segmented_image, cmap='jet')
plt.title('Segmentiertes Bild nach K-Means')
plt.axis('off')
plt.savefig(f'../out/segmented_image_kmeans_k_sigma_{k_sigma}.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()
