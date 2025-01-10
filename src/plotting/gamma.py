import numpy as np
import matplotlib.pyplot as plt

# Funktion zum Plotten von Kreis und Ellipse
def plot_shapes(gamma, theta, title, color):
    # Wertebereich für x und y
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)

    # Formel für Kreis oder Ellipse
    Z = (X * np.cos(theta) + Y * np.sin(theta)) ** 2 + gamma * (-X * np.sin(theta) + Y * np.cos(theta)) ** 2

    # Konturplot erstellen
    plt.contour(X, Y, Z, levels=[1], colors=color)

# Plotten von Kreis und Ellipse
plt.figure(figsize=(8, 8), facecolor='none')

# Kreis (gamma = 1)
plot_shapes(gamma=1, theta=0, title="Kreis", color='orange')

# Ellipse (gamma > 1)
plot_shapes(gamma=10, theta=np.pi / 6, title="Ellipse", color='brown')

# Diagrammgestaltung
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(alpha=0.3)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)


# Speichern als transparentes PNG
plt.savefig('./src/plotting/kreis_ellipse_transparent.png', transparent=True)
plt.show()

