import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature, color, filters, measure
from skimage.transform import resize
from sklearn.cluster import KMeans

# Bild einlesen und verkleinern
image = cv2.imread("../images/hund.png",cv2.IMREAD_GRAYSCALE)  # Bild einlesen
plt.imshow(image,'gray')
numRows, numCols = image.shape  # Bilddimensionen

# Selbe Wellenlängen und Orientierungen wie im Matlab-Beispiel bestimmen
wavelengthMin = 4 / np.sqrt(2)
wavelengthMax = np.hypot(numRows, numCols)
n = np.floor(np.log2(wavelengthMax / wavelengthMin)).astype(int)
wavelength = wavelengthMin * 2 ** np.arange(n - 1)

deltaTheta = 45
orientation = np.arange(0, 180, deltaTheta)

# Gabor-Filterbank erstellen
gabor_kernels = []
for theta in orientation:
    for w in wavelength:
        kernel = cv2.getGaborKernel((int(16*w), int(16*w)), 2*w, np.deg2rad(theta), w, 1.0, 0, ktype=cv2.CV_32F)
        kernel /= 1.0 * kernel.sum()  # Brightness normalization
        gabor_kernels.append(kernel)

# Ausgabe der Kernelgröße zur Verifizierung, dass sie gleich ist wie in Matlab
#print(gabor_kernels[0].shape)

# Anzeige der Gabor-Filter
num_filters = len(gabor_kernels)
cols = 6                                    # Anzahl der Spalten für das Bild
rows = (num_filters + cols - 1) // cols     # Berechnung der Anzahl der Zeilen
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
axes = axes.flatten()
for i, kernel in enumerate(gabor_kernels):
    ax = axes[i]
    ax.imshow(kernel, cmap='gray')
    ax.set_title(f"Filter {i + 1}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Anwendung der Filterbank auf das Bild
fig, axes = plt.subplots(rows, cols, figsize=(12,12))
axes = axes.flatten()
gabormag = np.zeros((image.shape[0], image.shape[1], len(gabor_kernels)))
for i, kernel in enumerate(gabor_kernels):
    ax = axes[i]
    filtered_image = cv2.filter2D(image, -1, kernel)  # Gabor-Filter auf das Bild anwenden
    gabormag[:, :, i] = filtered_image
    ax.imshow(filtered_image, cmap='gray')
    ax.set_title(f"Filtered image {i + 1}")
plt.tight_layout()
plt.show()


num_filters = len(gabor_kernels)
cols = 6                                    # Anzahl der Spalten für das Bild
rows = (num_filters + cols - 1) // cols     # Berechnung der Anzahl der Zeilen
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
axes = axes.flatten()
for i, kernel in enumerate(gabor_kernels):
    ax = axes[i]
    ax.imshow(kernel, cmap='gray')
    ax.set_title(f"Filter {i + 1}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Gaussian Blur anwenden (analog wie im Matlab-Beispiel)
print(gabormag.shape[2]-1)
for i in range(0,gabormag.shape[2]-1):
    g = gabormag[:,:,i]
    print(g)
    print("Modulo i%5", i%6)
    lambd = wavelength[i % 6]
    print(lambd)
    sigma = 0.5*lambd
    kernelsize =  int(2*np.ceil(2*sigma)+1)
    print("Kernelsize: ",kernelsize)
    g = cv2.GaussianBlur(g, [kernelsize, kernelsize], 3*sigma)

# Anzeige der Filterantworten -> PROBLEM: Diese unterscheiden sich stark vom Matlab Beispiel
num_filters = len(gabor_kernels)
cols = 6  # Anzahl der Spalten für die Montage
rows = (num_filters + cols - 1) // cols  # Berechnung der Anzahl der Zeilen
fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
axes = axes.flatten()

for i, ax in enumerate(axes[:num_filters]):
    ax.imshow(gabormag[:, :, i], cmap='gray')
    ax.set_title(f"Filter {i + 1}")
    ax.axis('off')

plt.tight_layout()
plt.show()
