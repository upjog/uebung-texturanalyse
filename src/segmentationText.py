# Struktur- und Objektextraktion WS2425
# Uebung 1: Texturanalyse
# Gruppe: Jan, Korvin, Ramon,
#
# Visualisierung der Wirkung des Verfahren an einfachen Bildern:    Bild mit Text (text.png)
# ==========================================================================================
# 1. Erzeuge Gabor-Filterbank
# 2. Wende Filter auf Bild an
# 3. Erzeuge Feature-Vektor aus Filterantworten und (x,y) Koordinaten
# 4. K-Means-Clustering zur Erkennung der Bereiche mit Text

import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = '../images/text.png'                      # Bildimport Linux
#image_path = './images/text.png'                       # Bildimport Windows
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

out_path = '../out/text/'   # Linux
#out_path = './out/text'     # Windows

colormap = 'jet'            # Auswahl der Colorbar
#colormap = 'gray'

height, width = image.shape                             # Breite und Hoehe des Bildes
print("Bildbreite: ", width, "Bildhöhe: ",height)

# === Definition Filterbankparameter ===
ksize = 31                  # Kernelgröße
gamma = 1.0                 # Ellipzität der Gauß-Funktion
psi = 0                     # Phasenverschiebung der Sinusfunktion
sigma = 5.0                 # Standardabweichung

# Winkelausrichtung der Sinusfunktion
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#orientations = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]

# Wellenlänge
# --------------------------------------------------
# Diese wird so definiert, wie z.B. in Jain & Bhattacharjee (1992)
# Hinweis zur Definition :
# Minimale Wellenlänge: 4/sqrt(2) = 2*sqrt(2)
#       -> ein Pixel hat Hypotenuse sqrt(2)
#       -> 2 zueinander Diagonale Pixel ergeben doppelte Distanz
#       -> Wellenlänge bei 45° kann nicht kürzer sein
# Maximale Wellenlänge: Hypotenuse der Bilddimensionen
#       -> von einer Ecke in die gegenüberleigende
# --------------------------------------------------

lambd_min = 2*np.sqrt(2)
lambd_max = np.sqrt(np.abs(width)**2 + np.abs(height)**2)/2
n = int(np.log2(lambd_max/lambd_min))               # Anzahl an Schritten min->max mit Exponent von 2
print("Number of wavelength steps: ", n)
wavelengths = 2**np.arange((n-1)) * lambd_min       # Nutze alle Wellenlängen ohne Begrenzung

# === Erstelle Filterbank und filtere Bild ===
filterbank = {}
filtered_img_bank_normalized = []

# Schleife über Wellenlängen und Ausrichtungen
for lambd in wavelengths:
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32FC1)     # erzeuge Filterkernel
        key = f"theta_{int(theta*180/np.pi)}_lambda_{round(lambd,1)}"                                       # erzeuge key für dict
        filterbank[key] = kernel

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ACHTUNG HIER MIT DATENTYPEN!!!
        filterresponse = cv2.filter2D(image,cv2.CV_32FC1, kernel)       # sinnvoll
        # filterresponse = cv2.filter2D(image,cv2.CV_8UC3, kernel)       # FALSCH, ergibt fehlerhaftes Bild
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Normalisiere mit Z-Score
        filterresponse_normalized = (filterresponse - np.mean(filterresponse)) / np.std(filterresponse)
        filtered_img_bank_normalized.append(filterresponse_normalized)

# === Visualisierungen ===
# Filterbank
plt.figure(figsize=(10,10))
for idx, filterkernel in enumerate(filterbank):
    #print(filterkernel)
    plt.subplot(len(wavelengths),len(orientations),idx+1)
    plt.imshow(filterbank.get(list(filterbank.keys())[idx]), cmap=colormap)
    plt.title(f"{list(filterbank.keys())[idx]}")
    plt.axis('off')
plt.suptitle("Filterbank")
plt.tight_layout()
plt.savefig(out_path + "Filterbank.png")

# Gefiltertes Bild (normalisiert)
plt.figure(figsize=(12,6))
for idx, filtered_img_normalized in enumerate(filtered_img_bank_normalized):
    plt.subplot(len(wavelengths),len(orientations),idx+1)
    plt.imshow(filtered_img_normalized, cmap=colormap)
    plt.title(f"{list(filterbank.keys())[idx]}")
    plt.axis('off')
    plt.colorbar()
plt.suptitle("Gefiltertes Bild")
plt.tight_layout()
plt.savefig(out_path + "FilteredImage.png")
plt.show()

# === Aufsummierung aller Filterantworten für eine Wellenlänge (Summe über theta) ===
sum_filtered_images = []

# Schleife über jede Wellenlänge
for idx_lambda, lambd in enumerate(wavelengths):
    summed_image = np.zeros_like(filtered_img_bank_normalized[0])   # Initialisiere Summenbild

    # Iteriere über alle Orientierungen und addiere die Filterantworten für diese Wellenlänge
    for idx_theta, theta in enumerate(orientations):
        idx = idx_lambda * len(orientations) + idx_theta    # Berechne den Index für Bild aktueller Filterantwort
        summed_image += filtered_img_bank_normalized[idx]   # Summiere die Antwort

    # Füge das Summenbild der Liste hinzu
    sum_filtered_images.append(summed_image)

# Visualisierung der Summenbilder für jede gewählte Wellenlänge
plt.figure(figsize=(10,14))
for idx, summed_image in enumerate(sum_filtered_images):
    plt.subplot(len(wavelengths), 1, idx + 1)
    plt.imshow(summed_image, cmap=colormap)
    plt.title(f"Wavelength: {wavelengths[idx]:.2f}")
    plt.axis('off')
    plt.colorbar()
#plt.tight_layout()
plt.suptitle('Filterantwort - Summe über alle Winkel')
plt.savefig(out_path + "FilteredImageSumOverAngles")
plt.show()

# === Erzeuge Featurevektoren aus Filterantworten (für Segmentierung) ===

# Definiere Aktivierungsfunktion
def activationFunction(t, alpha = 0.25):
    psi = (1 - np.exp(-2 * alpha * t)) / (1 + np.exp(-2 * alpha * t))
    return psi

summed_image_bsp = sum_filtered_images[4]

# Visualisiere Wirkung der Aktivierungsfunktion -> Kantenstärkung
plt.figure()
plt.subplot(2,1,1)
plt.imshow(summed_image_bsp,cmap=colormap)
plt.title("Ohne Aktivierungsfunktion")
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(activationFunction(summed_image_bsp),cmap=colormap)
plt.colorbar()
plt.title("Aktivierungsfunktion in Action")
plt.savefig(out_path + "Aktivierungsfunktion")

# Bestimme Texturmerkmal Average Absolute Deviation (AAD) [nach Jain & Bhattacharjee (1992)]
# Dieses enthält ein Maß für die mittlere "Texturenergie" in der Umgebung MxM
# M = 9                 # Alternative: fester Wert für M
M = int(5 * sigma)      # Wähle in Abhängigkeit der Standardabweichung. Hier Faktor 5 gewählt

if M & 2 == 0:          # M muss ungerade sein!
    M = M + 1

# Funktion zur Berechnung der AAD
def getAADImage(M, inputImage):

    # Ist sowohl mit Mittelwertfilter (gemäß Jain & Bhattacharjee (1992)) als auch mit Gaußfilter umgesetzt
    featureImageMean = np.zeros_like(inputImage)
    featureImageGauss = np.zeros_like(inputImage)
    activatedImage = activationFunction(inputImage)

    # Mittelwertfilter
    mean_kernel = np.ones((M,M), dtype = np.float32) / (M * M)

    # Gaussfilter
    gauss_kernel_1D = cv2.getGaussianKernel(M, sigma*5)
    gauss_kernel_2D = gauss_kernel_1D @ gauss_kernel_1D.T

    featureImageMean = cv2.filter2D(np.abs(activatedImage),-1, mean_kernel)
    featureImageGauss = cv2.filter2D(np.abs(activatedImage),-1, gauss_kernel_2D)
    return featureImageMean, featureImageGauss


featureImageMean, featureImageGauss = getAADImage(M, summed_image_bsp)
plt.figure(figsize=(10,2))
plt.subplot(1,2,1)
plt.imshow(featureImageMean,cmap=colormap)
plt.title("AAD Image Mean")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(featureImageGauss,cmap=colormap)
plt.title("AAD Image Gauss")
plt.colorbar()

plt.show()

# AAD für alle Summenbilder/FeatureKandidaten
aadImages = []
for image in sum_filtered_images:
    _, aadImage = getAADImage(M, image)
    aadImages.append(aadImage)

# Feature-Vektoren für jedes Pixel erstellen
# Hier verwendete features:
# - Summenbilder der Filterantworten
# - x und y-Koordinaten
feature_vectors = []

# Schleife über alle Pixel, jedes Pixel erhält einen feature_vector
for y in range(height):
    for x in range(width):
        feature_vector = []
        for featureImage in sum_filtered_images:        # hier nicht sum_filtered_images, sondern aad_images verwenden
            #print(featureImage[y,x])
            feature_vector.append(featureImage[y,x])

        # Geometrie-Infos (Pixelposition) als Feature einfügen
        feature_vector.append(x)
        feature_vector.append(y)

        feature_vectors.append(feature_vector)

# print((feature_vectors[19191])) # debugging

# === Clustering mit K-Means ===

feature_vectors = np.array(feature_vectors, dtype=np.float32)

# Z-Score Normalisierung der x- und y-Koordinaten (und ggf. anderen Einträgen auch)
print(feature_vectors.shape[1])
for i in range(feature_vectors.shape[1]):
    mean = np.mean(feature_vectors[:,i], axis=0)
    #print(mean)
    std = np.std(feature_vectors[:,i], axis=0)
    feature_vectors[:,i] = (feature_vectors[:,i] - mean) / std

# print((feature_vectors[19191]))    # debugging

# K-Means-Clustering
k = 2       # Wähle sinnvolle Clusteranzahl !
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)                           # termination criteria: max_iter oder epsilon
_, labels, centers = cv2.kmeans(feature_vectors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # Standardverwendung

clustered_image = labels.reshape(height, width)

# Visualisierung der Clustering-Ergebnisses
plt.figure(figsize=(10,4))
plt.imshow(clustered_image, cmap='jet')
plt.title(f"K-Means Clustering mit {k} Clustern")
plt.savefig(out_path + "Segmentierungsergebnis")
plt.show()
