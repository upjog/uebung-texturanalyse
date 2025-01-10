import numpy as np
import matplotlib.pyplot as plt

# Daten für die erste Normalverteilung (blau)
mu1 = 0  # Mittelwert
sigma1 = 1  # Standardabweichung
x = np.linspace(-5, 5, 500)
y1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)

# Daten für die zweite Normalverteilung (grün) mit größerem Sigma
mu2 = 0  # Mittelwert bleibt gleich
sigma2 = 2  # Größere Standardabweichung
y2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

# Plotten der Verteilungen
plt.figure(figsize=(8, 5), facecolor='none')
plt.plot(x, y1, label=f'mit $\mu={mu1}, \sigma={sigma1}$', color='blue')
plt.plot(x, y2, label=f'mit $\mu={mu2}, \sigma={sigma2}$', color='green')

# Zusätzliche Formatierungen
plt.xlabel('x')
plt.ylabel('Dichte')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Nullachse
plt.legend()
plt.grid(alpha=0.3)

# Speichern als transparentes PNG
plt.savefig('./src/plotting/normalverteilungen_transparent.png', transparent=True)
plt.show()



