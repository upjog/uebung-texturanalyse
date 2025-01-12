import numpy as np
import matplotlib.pyplot as plt
import cv2

struktur_frequenzen = {'combine.png': [1, 5, 6, 8, 9, 30],
                       'hund.png': [2,4,6,8,10,12,14,16,18,20],                    # 19
                       'IMAG0224.jpg': [6,12,18,24,32],
                       'IMG_3156.JPG': [30],
                       'poolraum_bsp.jpg': [17,295], #12,25,70,140
                       'text.png':[1,2,3,7],
                       'woodring.png':[9,38]}

index = 1

image_name = list(struktur_frequenzen.keys())[index]
image_path = '../images/'+image_name
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

test = image_name[:-4]

# Filterbank parameter
width, height = image.shape                     # Breite und Hoehe des Bildes
ksize = int(min(width,height)*0.1)             # 10% der kleineren Bilddimension
# ksize = 400
gamma = 1                                    # Rundheit: 0.5 = Kreis, 0/1 = Linie
psi = 0                                        # Phasenverschiebung
# 4 Orientierungen: 0°, 45°, 90°, 135°
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Min: 4/sqrt(2) = 2*sqrt(2)
#       -> ein Pixel hat Hypotenuse sqrt(2), 2 zueinander Diagonale Pixel ergeben doppelte Distanz
#       -> kürzeste mögliche Wellenlänge bei 45°
lambd_min = 2*np.sqrt(2)
# Max: längeste mögl. Wellenlänge ist die Hypotenuse der Bilddimensionen
#       -> von einer Ecke in die gegenüberleigende
lambd_max = np.sqrt(np.abs(width)**2 + np.abs(height)**2)
# n: Wie viele Schritte können wir von min bis max machen, wenn wir den Exponenten von 2 erhöhen
n = int(np.log2(lambd_max/lambd_min))
wavelengths2 = 2**np.arange((n-1)) * lambd_min       # Wellenlängen
# wavelengths = wavelengths[:2]*2               # begrenzen, da größeres lambda kein anschauliches ergebnis erzeugt
wavelengths = struktur_frequenzen[image_name]
# wavelengths.extend([lam for lam in wavelengths2 if lam > wavelengths[-1]])
k_sigma = 12.0                                 # Faktor für Standardabweichung

filterbank = {}
filtered_img_bank = []
############# hier loop
# loop über wellenlängen, dann loop über orientierungen
for lambd in wavelengths:
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize,ksize), k_sigma, theta, lambd, gamma, psi)
        plt.figure()
        plt.imshow(kernel, cmap='grey')
        key = f"theta_{int(theta*180/np.pi)}_lambda_{round(lambd,1)}"
        filterbank[key] = kernel
        filterresponse = cv2.filter2D(image,cv2.CV_8UC3, kernel)
        filtered_img_bank.append(filterresponse)

plt.figure(figsize=(15,10))
for idx, filtered_img in enumerate(filtered_img_bank):
    plt.subplot(len(wavelengths),len(orientations),idx+1)
    plt.imshow(filtered_img,cmap='gray')
    plt.title(f"{list(filterbank.keys())[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"../out/{image_name[:-4]}_response_collection_lambdaMax_{round(lambd,1)}.png",dpi=300, bbox_inches='tight',transparent=False)
plt.show()


# plt.figure()
# plt.imshow(kernel)
# plt.title(f"Gabor Filter {key}")
# plt.axis('off')
# plt.colorbar()
# # plt.savefig(f"./out/{key}_gabor_kernel.png", dpi=300, bbox_inches='tight', transparent=False)
# plt.show()
