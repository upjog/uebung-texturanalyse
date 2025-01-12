import numpy as np
import matplotlib.pyplot as plt
import cv2

def rep_or_ins(arr1, arr2, maxDiff):
    """
    Ersetze or füge Elemente aus array1 in array2 ein,
    abhängig von der Differenz zu dem nächsten Element

    arr1: array mit Werten, welche eingefügt werden sollen
    arr2: array mit Werten, welche ersetzt oder ergänzt werden
    maxDiff: inklusive Differenz, bis zu welcher Werte ersetzt statt ergänzt werden   
    """
    arr_return = arr2.copy()

    for num in arr1:
        idx = np.abs(arr_return - num).argmin()
        diff = arr_return[idx] - num

        if np.abs(diff) <= maxDiff: # num ist nahe an einer Potenz von 2 und ersetzt diese
            arr_return[idx] = num
        elif diff > 0: # num ist kleiner als der nächste Wert -> wird davor eingefügt
            np.insert(arr_return,idx, num)
        else: # num ist größer als der nächste Wert -> wird dahinter eingefügt
            np.insert(arr_return,idx+1, num)

    return arr_return

struktur_frequenzen = {'combine.png': [4,6,8,10,30],
                       'hund.png': [10,15,20,25,30],
                       'IMAG0224.jpg': [4,8,12,16,30],
                       'IMG_3156.JPG': [15,20,25,30],
                       'poolraum_bsp.jpg': [15,20,25,300], #12,25,70,140
                       'text.png':[3,7,10,14],
                       'woodring.png':[9,18,27,38]} #7,14,21,28
# 0 - 6
index = 1

image_name = list(struktur_frequenzen.keys())[index]
image_path = './images/'+image_name
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

variable_kernel = True
# variable_kernel = False

# Filterbank parameter
width, height = image.shape                     # Breite und Hoehe des Bildes
ksize = int(min(width,height)*0.1)              # Kernelgröße
gamma = 0.5                                    # Rundheit: 1 = Kreis, 0 = Linie
psi = 0                                        # Phasenverschiebung
# 4 Orientierungen: 0°, 45°, 90°, 135°
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

wavelengths = struktur_frequenzen[image_name]
k_sigma = 12.0                                 # Faktor für Standardabweichung

######### Alternativ, automatische Festlegung von lambda 
# Min: 4/sqrt(2) = 2*sqrt(2) 
#       -> ein Pixel hat Hypotenuse sqrt(2), 2 zueinander Diagonale Pixel ergeben doppelte Distanz
#       -> kürzeste mögliche Wellenlänge bei 45° 
lambd_min = 2*np.sqrt(2)
# Max: längeste mögl. Wellenlänge ist die Hypotenuse der Bilddimensionen
#       -> von einer Ecke in die gegenüberleigende
lambd_max = np.sqrt(np.abs(width)**2 + np.abs(height)**2)
# n: Wie viele Schritte können wir von min bis max machen, wenn wir den Exponenten von 2 erhöhen
n = int(np.log2(lambd_max/lambd_min)) 
wavelengths2 = 2**np.arange((n-1)) * lambd_min       # Wellenlängen als Potenzen von 2

############# Nicht ganz so sinnvoll, da die größeren Wellenlängen keine interpretierbaren Ergebnisse zeigen 
# # Kombination von beiden Lsiten. Wenn die manuell festgelegte Wellenlänge nahe an einer Potenz von 
# # 2 liegt, ersetzt diese den entsprechenden Eintrag in der automatisch generierten Liste.
# # Sonst wird die manuelle Wellenlänge ergänzt
wavelengths = rep_or_ins(wavelengths,wavelengths2,3)

if variable_kernel:
    ksize = 3*np.array(wavelengths).astype(int)
    # entferne alle Kernelgrößen die größer als die kleinere Bilddimension sind
    # ksize = ksize[ksize < min(width,height)] # damit würde die innere Schleife kaputt gehen

filterbank = {}
filtered_img_bank = []
# loop über orientierungen, dann loop über wellenlängen
for i, lambd in enumerate(wavelengths):
    if isinstance(ksize,np.ndarray):
        kernel_size = (ksize[i],ksize[i])
    else:
        kernel_size = (ksize,ksize)
    for theta in orientations:
        kernel = cv2.getGaborKernel(kernel_size, k_sigma, theta, lambd, gamma, psi)
        key = f"theta_{int(theta*180/np.pi)}_lambda_{round(lambd,1)}"
        filterbank[key] = kernel
        filterresponse = cv2.filter2D(image,cv2.CV_32FC1, kernel)
        filtered_img_bank.append(filterresponse)

plt.figure(figsize=(15,10))
for idx, filtered_img in enumerate(filtered_img_bank):
    plt.subplot(len(wavelengths),len(orientations),idx+1)
    plt.imshow(filtered_img,cmap='gray')
    plt.title(f"{list(filterbank.keys())[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"./out/response_collection_dynKernel_{variable_kernel}_{image_name[:-4]}_lambdaMax_{round(lambd,1)}.png",dpi=300, bbox_inches='tight',transparent=False)


# plt.figure()
# plt.imshow(kernel)
# plt.title(f"Gabor Filter {key}")
# plt.axis('off')
# plt.colorbar()
# # plt.savefig(f"./out/{key}_gabor_kernel.png", dpi=300, bbox_inches='tight', transparent=False)
# plt.show()
