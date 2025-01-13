# Struktur- und Objektextraktion WS2425
# Uebung 1: Texturanalyse
# Gruppe: Jan, Korvin, Ramon,
#
# Einfache Visualisierung der Frequenzspektren von Bildern
# ========================================================
# Durch fft werden eingegebene Bilder in den Frequenzraum transformiert.
#

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Ordnerpfad zu den Bildern
image_folder = '../images'

# Alle Bilddateien im Ordner finden
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
# image_files = ['../images/easySynthetic1.png']
# filename = 'easySynthetic1Freq.png'

for image_file in image_files:

    image_path = os.path.join(image_folder, image_file)     # Bildpfad erstellen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    # Bild einlesen (im Graustufenformat)

    if image is None:
        print(f"Fehler beim Laden des Bildes: {image_file}")
        continue

    print(f"Verarbeite Bild: {image_file}, Größe: {image.shape}")

    # Fourier-Transformation
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # Verschiebe die Nullfrequenz in die Mitte
    magnitude_spectrum = np.abs(fshift)

    # Visualisierung des Frequenzspektrums
    plt.imshow(np.log(magnitude_spectrum + 1),cmap='gray')
    plt.title(f'Amplitudenspektrum (log) - {image_file}')
    plt.colorbar()
    plt.xlabel('Frequenz X')            # Achsenbeschriftung
    plt.ylabel('Frequenz Y')

    # Anzahl der Ticks festlegen
    print(image.shape)
    nTicks = 5      # Bei ungerader Zahl ist ein Tick (fast) in der Mitte
    xticks = np.linspace(0, image.shape[1] - 1, nTicks)
    yticks = np.linspace(0, image.shape[0] - 1, nTicks)

    # Frequenzbereich für Labels berechnen (-N/2 bis N/2)
    xticklabels = np.linspace(-image.shape[1] // 2, image.shape[1] // 2 - 1, nTicks)
    yticklabels = np.linspace(-image.shape[0] // 2, image.shape[0] // 2 - 1, nTicks)

    # Ticks und Labels setzen
    plt.xticks(xticks, labels=xticklabels.astype(int))
    plt.yticks(yticks, labels=yticklabels.astype(int))

    saveTo = '../out/frequency/freq_'+image_file
    plt.savefig(saveTo)
    plt.show()
