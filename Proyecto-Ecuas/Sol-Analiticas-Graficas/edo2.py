import numpy as np
import matplotlib.pyplot as plt

# Solución analítica completa
def y_analitica(t, A, B):
    transitorio = np.exp(-0.25*t) * (A*np.cos(np.sqrt(31)/4 * t) + B*np.sin(np.sqrt(31)/4 * t))
    estacionario = (1/205)*(-28*np.cos(3*t) + 6*np.sin(3*t))
    return transitorio + estacionario

# Configuración de la gráfica
t = np.linspace(0, 20, 1000)
plt.figure(figsize=(12, 8))

# Graficamos con diferentes condiciones iniciales
plt.plot(t, y_analitica(t, 0.5, 0), 'b-', linewidth=2, label='A=0.5, B=0')
plt.plot(t, y_analitica(t, 0, 0.5), 'r--', linewidth=2, label='A=0, B=0.5')
plt.plot(t, y_analitica(t, -0.5, -0.5), 'g-.', linewidth=2, label='A=-0.5, B=-0.5')

plt.title('Solución analítica del oscilador forzado', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()