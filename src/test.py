import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)

def create_gabor_filterbank(orientations, wavelengths, gamma, psi, k_sigma_list, ksize_factor=3):
    """
    Erstelle eine Gabor-Filterbank mit variablen Wellenlängen, Orientierungen und verschiedenen Glättungsgraden (k_sigma).
    :param orientations: Liste der Filterorientierungen
    :param wavelengths: Liste der Wellenlängen
    :param k_sigma_list: Liste der Glättungsgrade
    :param ksize_factor: Faktor zur Berechnung der Filtergröße
    :return: Filterbank als Dictionary
    """
    filterbank = {}
    for theta in orientations:
        for wavelength in wavelengths:
            for k_sigma in k_sigma_list:
                sigma = k_sigma * wavelength  # Standardabweichung in Bezug auf Wellenlänge
                ksize = int(ksize_factor * wavelength)  # Berechne die Filtergröße
                key = f"theta_{round(theta, 3)}_wavelength_{round(wavelength, 3)}_sigma_{round(sigma, 3)}"
                gabor_kernel = gabor_filter(ksize, sigma, theta, wavelength, gamma=1, psi=0)
                filterbank[key] = (gabor_kernel, sigma)  # Speichern des sigma-Werts und des Filters
    return filterbank

def gaussian_kernel_2d(size, sigma):
    """ Erzeuge eine 2D-Gaußsche Kernfunktion """
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(- ((x - (size-1)/2) ** 2 + (y - (size-1)/2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_smoothing_feature_space(features, sigma_f=1.0, sigma_p=1.0, window_size=3):
    """
    Wendet Gaussian Smoothing im Feature Space an, um benachbarte Pixel zu glätten.

    :param features: Das Feature-Array der Größe (Anzahl der Pixel, Anzahl der Merkmale)
    :param sigma_f: Glättungsfaktor im Feature-Raum
    :param sigma_p: Glättungsfaktor für die räumliche Nähe
    :param window_size: Größe des Fensters für die benachbarten Pixel
    :return: Glättetes Feature-Array
    """
    rows, cols = features.shape
    smoothed_features = np.copy(features)

    # Erstelle eine 2D-Gaußsche Kernel für die räumliche Nähe
    gaussian_kernel = gaussian_kernel_2d(window_size, sigma_p)

    for i in range(rows):
        for j in range(cols):
            # Extrahiere den Feature-Wert des aktuellen Pixels
            current_pixel_feature = features[i, j]

            # Nachbarschaftsfenster bestimmen
            neighbors = []
            weights = []
            for di in range(-window_size//2, window_size//2 + 1):
                for dj in range(-window_size//2, window_size//2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Berechne Gewichtung durch den Gaussian Kernel
                        spatial_distance = np.sqrt(di**2 + dj**2)
                        feature_distance = np.abs(current_pixel_feature - features[ni, nj])  # Hier: 1D-Feature-Vektor
                        weight = np.exp(- (feature_distance**2) / (2 * sigma_f**2) - (spatial_distance**2) / (2 * sigma_p**2))

                        neighbors.append(features[ni, nj])
                        weights.append(weight)

            # Berechne gewichteten Mittelwert der Nachbarn
            weights = np.array(weights)
            neighbors = np.array(neighbors)
            smoothed_feature = np.average(neighbors, axis=0, weights=weights)
            smoothed_features[i, j] = smoothed_feature

    return smoothed_features


# Filterbank-Parameter
gamma = 1
psi = 0
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 Orientierungen
wavelengths = [4, 8, 32, 64]  # Verschiedene Wellenlängen
k_sigma_list = [0.25,0.5, 1.0, 3.0]  # Verschiedene Glättungsgrade
ksize_factor = 3  # Faktor für Filtergröße (abhängig von Wellenlänge)

# Erzeuge Gabor-Filterbank
filterbank = create_gabor_filterbank(orientations, wavelengths, gamma, psi, k_sigma_list, ksize_factor)

# Anwenden der Filterbank auf das Bild
filtered_images = []
for key, (gabor_kernel, sigma) in filterbank.items():
    filtered_image = cv2.filter2D(image, -1, gabor_kernel)
    filtered_images.append(filtered_image)

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
smoothed_features = gaussian_smoothing_feature_space(features, sigma_f=1.0, sigma_p=1.0, window_size=3)


# Normalisierung der Features: StandardScaler (Z-Standardisierung)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(smoothed_features)

# Überprüfen der Form der Features
print("Form der Feature-Daten:", features.shape)

# K-Means Clustering
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(features_normalized)

# Kategorisierung der Pixel basierend auf den Clustern
segmented_image = kmeans.labels_.reshape(image.shape)

# Visualisierung der segmentierten Ergebnisse
plt.figure(figsize=(10, 8))
plt.imshow(segmented_image, cmap='jet')
plt.title(f'Segmentiertes Bild nach K-Means mit k_sigma {k_sigma_list}')
plt.axis('off')
plt.savefig(f'../out/segmented_image_kmeans_k_sigma_{k_sigma_list}_wavel_{wavelengths}.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()
