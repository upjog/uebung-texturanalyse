# Uebung 1 zu Objektextraktion: Texturanalyse mit dem Gabor-Filter

## Google Drive Präsentation
[Gabor-Filter](https://docs.google.com/presentation/d/14SjZZfaMBFQu1au9Zh6sT888lLA81IJ94J1iDLPqCm4/edit?usp=sharing)

## Struktur des Repos:
- Ausführbahre Skripte befinden sich im Ordner `src`
  - `getFrequencyOfImages.py` veranschaulicht den Frequenzraum von Bildern
  - `easyExamples.py` veranschaulicht die dynamische Wahl von Kernelgrößen und Standardabweichungen basierend auf der Wellenlänge
  - `interactive_gabor.py` veranschaulicht den Einfluss der Parameter auf die Filtermaske
  - `ownImplementation.py` enthält eine eigene Implementierung der Gabor-Funktion und einen Vergleich mit den Funktionen aus den `cv2` und `scikit-image` Paketen
  - `segmentation*.py` enthalten Workflows zur Segmentierung mithilfe von Gabor-Filtern für verschiedene Bilder
- Eingabebilder befinden sich im Ordner `images`
- Ausgabedateien und -bilder befinden sich im Ordner `out` in Unterordnern für das jeweilige Eingabebild
  - Der Ordner `old` enthält frühere/falsche (Zwischen-)Ergebnisse, welche nur für die Historie noch gespeichert werden.

## Hilfreiche Links:
- [Youtube-Video zum Gabor-Filter](https://www.youtube.com/watch?v=QEz4bG9P3Qs)
- [Basic Beispiel zum erstellen von Gabor-Filterkerneln mit openCV](https://www.geeksforgeeks.org/opencv-getgaborkernel-method/)
- [Matlab Beispiel zum Gabor-Filter](https://de.mathworks.com/help/images/texture-segmentation-using-gabor-filters.html)
- [Scikit-image Beispiel zum Gabor-Filter](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html)
- [Matlab-Implementierung einer Textursegmentierung mit Gabor-Filtern](https://github.com/mortezamg63/Texture-Segmentation-using-Gabor-Filters)
- [Paper zur unüberwachter Textursegmentierung mit Gabor-Filter](https://www.ee.columbia.edu/~sfchang/course/dip-S06/handout/jain-texture.pdf)
- [Tutorial zur Mathematik des Gabor-Filters](https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf)
- [Kurzer Artikel mit Infos](https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97)
- [Paper zur Texturanalyse mit Gabor und SVM](https://www.researchgate.net/publication/283684622_Texture-based_Classification_of_Workpiece_Surface_Images_using_the_Support_Vector_Machine)
- [In-depth-Paper](https://www.sciencedirect.com/science/article/pii/S0031320399001818#FIG5)

## Sonstige Links:
- [Gabor-Filter für Edge Detection](https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/)

## Relevante Punkte bei der Definierung des Filters:
- Optimale Kernelgröße
- Optimale Wellenlängen / welche verwenden? Immer alle oder nur kleiner Bereich?
- Normierung (MinMax vs Z-Normalisierung)
- Workflow Segmentierung (K-Means vs andere Verfahren)
