import numpy as np
import matplotlib.pyplot as plt

# Solución analítica
def solucion(t, C1, C2):
    factor = np.exp(3*t)
    x = factor * (C1 * np.cos(np.sqrt(2)*t) + C2 * np.sin(np.sqrt(2)*t))
    y = factor * ((C1 + np.sqrt(2)*C2) * np.cos(np.sqrt(2)*t) + 
                 (C2 - np.sqrt(2)*C1) * np.sin(np.sqrt(2)*t))
    return x, y

# Configuración de la gráfica
t = np.linspace(0, 2, 500)  
plt.figure(figsize=(14, 10))

# Diferentes condiciones iniciales
configs = [
    (1, 0, 'C1=1, C2=0'),
    (0, 1, 'C1=0, C2=1'),
    (0.5, 0.5, 'C1=0.5, C2=0.5')
]

# Soluciones en función del tiempo
plt.subplot(2, 2, 1)
for C1, C2, label in configs:
    x, y = solucion(t, C1, C2)
    plt.plot(t, x, linewidth=2, label=f'x(t): {label}')
    plt.plot(t, y, '--', linewidth=2, label=f'y(t): {label}')
plt.title('Soluciones analíticas', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('Valores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plano fase
plt.subplot(2, 2, 2)
for C1, C2, label in configs:
    x, y = solucion(t, C1, C2)
    plt.plot(x, y, linewidth=1.5, label=label)
plt.title('Plano fase analítico (x vs y)', fontsize=14)
plt.xlabel('x(t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()


plt.tight_layout()
plt.show()