import numpy as np
import matplotlib.pyplot as plt

# Funktion zum Plotten einer modulierten Sinuswelle
def plot_modulated_sine(k, m, A, b, phi, x_range, num_points):
    """
    Plottet eine modulierte Sinuswelle der Form:
    y(x) = sin(k * (x + phi)) * (1 + A * sin(m * (x + phi))) + b

    Parameter:
    - k: Frequenz der Grundschwingung
    - m: Frequenz der Modulation
    - A: Amplitude der Modulation
    - b: Offset der Modulation (in y-Richtung)
    - phi: Phasenverschiebung
    - x_range: Bereich der x-Werte (min, max)
    - num_points: Anzahl der Punkte zur Diskretisierung
    """
    # x-Werte generieren
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Modulierte Sinuswelle berechnen
    y = np.sin(k * (x + phi)) * (1 + A * np.sin(m * (x + phi))) + b
    
    return x, y

# Hauptplot erstellen
def plot_combined_sines():
    k, m, A, b = -2.2, -3.3, 1.5, 0.6
    # Erste modulierte Sinuswelle
    x1, y1 = plot_modulated_sine(k, m, A, b, phi=0, x_range=(-10, 10), num_points=1000)
    
    # Zweite modulierte Sinuswelle mit phi=0.5 (Phasenverschiebung)
    x2, y2 = plot_modulated_sine(k, m, A, b, phi=0.5, x_range=(-10, 10), num_points=1000)

    # Plotten
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label=f"mit $\phi$ = 0", color='darkblue')
    plt.plot(x2, y2, label=f"mit $\phi$ = 0.5", color='magenta', linestyle='--')

    # Diagrammgestaltung
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(alpha=0.4)
    plt.xlim(-2, 2)
    plt.ylim(-2, 4)
    plt.legend()
    
    
    # Speichern als transparentes PNG
    plt.savefig('./src/plotting/modulierte_sinuswellen_transparent.png', transparent=True)
    plt.show()

# Beispielaufruf der Funktion
plot_combined_sines()
