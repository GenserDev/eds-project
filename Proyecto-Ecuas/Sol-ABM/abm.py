import numpy as np
import matplotlib.pyplot as plt

# ==================================================================
# Implementación del método Predictor-Corrector ABM
# ==================================================================
def adams_bashforth_moulton(f, t0, tf, y0, h, steps=4):
    # Inicialización con RK4 para los primeros puntos
    n = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    
    # Función RK4 para inicialización
    def rk4_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Calcular primeros 'steps-1' puntos con RK4
    for i in range(1, steps):
        y[i] = rk4_step(t[i-1], y[i-1], h)
    
    # ABM de 4 pasos
    for i in range(steps-1, n-1):
        # Predictor (Adams-Bashforth de 4 pasos)
        y_pred = y[i] + h/24 * (55*f(t[i], y[i]) 
                                - 59*f(t[i-1], y[i-1]) 
                                + 37*f(t[i-2], y[i-2]) 
                                - 9*f(t[i-3], y[i-3]))
        
        # Corrector (Adams-Moulton de 4 pasos)
        y[i+1] = y[i] + h/24 * (9*f(t[i+1], y_pred) 
                                 + 19*f(t[i], y[i]) 
                                 - 5*f(t[i-1], y[i-1]) 
                                 + f(t[i-2], y[i-2]))
    
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
h = 0.1

# Resolver con ABM
t1, y1 = adams_bashforth_moulton(edo1, t0, t_final, y0, h)

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t1, y1[:, 0], 'b-', linewidth=2, label='Solución ABM')
plt.title('Solución de $dy/dt = -y + \sin(t)$', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('edo1_abm.png')
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
h = 0.1

# Resolver con ABM
t2, z2 = adams_bashforth_moulton(edo2, t0, t_final, y0, h)
y2 = z2[:, 0]  
yp2 = z2[:, 1] 

# Gráficas
plt.figure(figsize=(12, 8))

# Solución y(t)
plt.subplot(2, 1, 1)
plt.plot(t2, y2, 'b-', linewidth=2)
plt.title('Solución de $y\'\' + 0.5y\' + 2y = \cos(3t)$', fontsize=14)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.savefig('edo2_abm.png')
plt.show()

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

# Resolver con ABM
t3, v3 = adams_bashforth_moulton(sistema, t0, t_final, v0, h)
x3 = v3[:, 0]
y3 = v3[:, 1]

# Gráficas
plt.figure(figsize=(14, 10))

# Soluciones en función del tiempo
plt.subplot(2, 2, 1)
plt.plot(t3, x3, 'b-', linewidth=2, label='x(t)')
plt.plot(t3, y3, 'r-', linewidth=2, label='y(t)')
plt.title('Solucion ABM', fontsize=14)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('Valores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plano fase
plt.subplot(2, 2, 2)
plt.plot(x3, y3, 'g-', linewidth=1.5)
plt.plot(x3[0], y3[0], 'bo', markersize=8, label='Inicio (t=0)')
plt.plot(x3[-1], y3[-1], 'ro', markersize=8, label=f'Fin (t={t_final})')
plt.title('Plano fase (x vs y)', fontsize=14)
plt.xlabel('x(t)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()