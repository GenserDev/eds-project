import numpy as np
import matplotlib.pyplot as plt

# Solución analítica
def y_analitica(t, C):
    return 0.5*(np.sin(t) - np.cos(t)) + C*np.exp(-t)

# Configuración de la gráfica
t = np.linspace(0, 10, 500)
plt.figure(figsize=(10, 6))

# Graficamos con diferentes constantes
plt.plot(t, y_analitica(t, 0.5), 'b-', linewidth=2, label='C = 0.5')
plt.plot(t, y_analitica(t, 1.0), 'r--', linewidth=2, label='C = 1.0')
plt.plot(t, y_analitica(t, 0.0), 'g-.', linewidth=2, label='C = 0.0')

plt.title('Solución analítica de $dy/dt = -y + \sin(t)$', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()