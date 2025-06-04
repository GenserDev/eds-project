import numpy as np
import matplotlib.pyplot as plt

# ==================================================================
# Implementación del método RK4
# ==================================================================
def rk4_system(f, t0, y0, t_final, h):
    # Inicialización
    n = int((t_final - t0) / h) + 1
    t = np.linspace(t0, t_final, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    # Iteración RK4
    for i in range(n - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# ==================================================================
# 1. Primera EDO: dy/dt = -y + sin(t)
# ==================================================================
def edo1(t, y):
    return np.array([-y[0] + np.sin(t)])

# Parámetros de solución
t0 = 0
y0 = [0.5]  
t_final = 10
h = 0.05

# Resolver con RK4
t1, y1 = rk4_system(edo1, t0, y0, t_final, h)

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t1, y1[:, 0], 'b-', linewidth=2)
plt.title('Solución RK4 de $dy/dt = -y + \sin(t)$', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==================================================================
# 2. Segunda EDO: y'' + 0.5y' + 2y = cos(3t)
# ==================================================================
def edo2(t, z):
    y, yp = z
    return np.array([yp, 
                    np.cos(3*t) - 0.5*yp - 2*y])

# Parámetros de solución
t0 = 0
y0 = [0, 0]  
t_final = 20
h = 0.05

# Resolver con RK4
t2, z2 = rk4_system(edo2, t0, y0, t_final, h)
y2 = z2[:, 0]  
yp2 = z2[:, 1]  

# Gráficas
plt.figure(figsize=(12, 8))

# Solución y(t)
plt.subplot(2, 1, 1)
plt.plot(t2, y2, 'b-', linewidth=2)
plt.title('Solución RK4 de $y\'\' + 0.5y\' + 2y = \cos(3t)$', fontsize=14)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# ==================================================================
# 3. Sistema 2x2: dx/dt = 2x + y, dy/dt = -3x + 4y
# ==================================================================
def sistema(t, v):
    x, y = v
    return np.array([2*x + y, 
                    -3*x + 4*y])

# Parámetros de solución
t0 = 0
v0 = [1, 0]  
t_final = 2  
h = 0.01

# Resolver con RK4
t3, v3 = rk4_system(sistema, t0, v0, t_final, h)
x3 = v3[:, 0]
y3 = v3[:, 1]

# Gráficas
plt.figure(figsize=(14, 10))

# Soluciones en función del tiempo
plt.subplot(2, 2, 1)
plt.plot(t3, x3, 'b-', linewidth=2, label='x(t)')
plt.plot(t3, y3, 'r-', linewidth=2, label='y(t)')
plt.title('Soluciones RK4', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('Valores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plano fase
plt.subplot(2, 2, 2)
plt.plot(x3, y3, 'g-', linewidth=1.5)
plt.plot(x3[0], y3[0], 'bo', label='Inicio (t=0)')
plt.plot(x3[-1], y3[-1], 'ro', label=f'Fin (t={t_final})')
plt.title('Plano fase (x vs y)', fontsize=14)
plt.xlabel('x(t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()