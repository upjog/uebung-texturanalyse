# Struktur- und Objektextraktion WS2425
# Uebung 1: Texturanalyse
# Gruppe: Jan, Korvin, Ramon,
#
# Vergleich selbst implementierten Funktion für den Gabor-Filter mit der Funktion von opencv
# ==========================================================================================
# Die Funktionen cv2.getGaborKernel() und skimage.gabor_kernel() unterscheiden sich folgendermaßen:
# - cv2.getGaborKernel() erhält eine feste Kernelsize, skimage.gabor_kernel() erstellt diese anhand eines Faktors der Standardabweichung
#   -> und zwar ist dann ksize = n_stds * sigma * 2 + 1 (aber nur für theta = 0 oder 90° )
# - cv2.getGaborKernel() gibt nur den Realteil aus, skimage.gabor_kernel() die komplexe Funktion
# - cv2.getGaborKernel() benötigt die Wellenlänge, skimage.gabor_kernel() die Frequenz
# - cv2.getGaborKernel() benutzt gamma für die Ellipsizität der Gaußkurve, skimage.gabor_kernel() nutzt direkt sigma_x und sigma_y
# - VORSICHT: skimage.gabor_kernel() ändert seine größe für unterschiedliche theta

import numpy as np
import matplotlib.pyplot as plt
import cv2                                  # Für Funktion getGaborKernel
from skimage.filters import gabor_kernel    # Für Funktion gabor_kernel

# Eigene Funktion
def getMyGaborKernelReal(ksize = 31, lambd = 11, theta = 0.0, sigma = 3.0, gamma = 1.0, psi = 0):
    # ksize muss ungerade sein -> erhöhe wenn notwendig Kernelsize um eins
    if ksize % 2 == 0:
        ksize = ksize + 1

    xSpace = np.linspace(-(ksize-1) // 2, (ksize-1) // 2, ksize)
    ySpace = np.linspace(-(ksize-1) // 2, (ksize-1) // 2, ksize)
    x, y = np.meshgrid(xSpace, ySpace)
    xTheta = x * np.cos(theta) + y * np.sin(theta)
    yTheta = -x * np.sin(theta) + y * np.cos(theta)
    gaborKernel = np.exp(-(xTheta**2 + gamma**2 * yTheta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xTheta / lambd + psi)
    return gaborKernel

# Parameter des Gabor-Filters
ksize = 31              # Größe des Gabor-Filters
lambd = 2*np.sqrt(2)              # Wellenlänge
f = 1/lambd             # Frequenz
theta = np.pi/1         # Orientierung des Filters (in Radiant)
sigma = 2.0             # Breite der Gausschen Hüllkurve
sigma = lambd * 1.0
gamma = 1.0             # Form des Filters (1.0 kreisförmig)
psi = 0                 # Phase

# bestimme n_stds so, dass ksize für scikitKernel gleich groß ist wie für die anderen Funktionen
n_stds = int((ksize-1)/2 / sigma)
#n_stds = 3
#ksize = int(n_stds * sigma * 2 + 1)
print(n_stds)

# Kernel im Bildraum
myKernel = getMyGaborKernelReal(ksize, lambd, theta, sigma, gamma, psi)
cvKernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi)
scikitKernel = gabor_kernel(frequency = 1/lambd, theta = theta, sigma_x = sigma, sigma_y = sigma, n_stds = n_stds, offset = psi )
scikitKernelRealPart = scikitKernel.real
#print("myKernel und cvKernel gleich: ", np.array_equal(myKernel, cvKernel))
#print("Diff myKernel und cvKernel", myKernel - cvKernel)
#print("myKernel und scikitKernelRealPart gleich: ", np.array_equal(myKernel, scikitKernelRealPart))
print("MyKernel: ",myKernel.shape)
print("CvKernel: ",cvKernel.shape)
print("ScikitKernel: ",scikitKernelRealPart.shape)

# Kernel im Frequenzraum
myKernel_f = np.fft.fft2(myKernel)
myKernel_f_shifted = np.fft.fftshift(myKernel_f)          # Verschiebung der Nullfrequenz zur Mitte
myKernel_mag_spectrum = np.abs(myKernel_f_shifted)

cvKernel_f = np.fft.fft2(cvKernel)
cvKernel_f_shifted = np.fft.fftshift(cvKernel_f)
cvKernel_mag_spectrum = np.abs(cvKernel_f_shifted)

scikitKernel_f = np.fft.fft2(scikitKernelRealPart)
scikitKernel_f_shifted = np.fft.fftshift(scikitKernel_f)
scikitKernel_mag_spectrum = np.abs(scikitKernel_f_shifted)

# Visualisierung
plt.figure(figsize=(12, 6))

# Original Gabor-Filter im Bildraum
plt.subplot(3, 2, 1)
plt.imshow(myKernel)
plt.title('Gabor-Filter im Bildraum')
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(cvKernel)
plt.title('Gabor-Filter im Bildraum - opencv')
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(scikitKernelRealPart)
plt.title('Gabor-Filter im Bildraum - scikit-image')
plt.colorbar()
plt.axis('off')

# Frequenzspektrum des Gabor-Filters
plt.subplot(3, 2, 2)
plt.imshow(myKernel_mag_spectrum)
plt.title('Frequenzspektrum des Gabor-Filters')
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cvKernel_mag_spectrum)
plt.title('Gabor-Filter im Bildraum - opencv')
plt.colorbar()
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(scikitKernel_mag_spectrum)
plt.title('Gabor-Filter im Bildraum - scikit-image')
plt.colorbar()
plt.axis('off')

plt.show()
