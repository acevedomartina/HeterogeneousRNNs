import numpy as np

# Parámetros de simulación
T = 1000
dt = 0.1
sigman = 0.0
v_threshold = 0
v_rest = [-22, -12.3, -17, -8.5]

def dynamics(x, I, tipo):
    if tipo == "LIF":
        dx = -x + I + np.random.randn() * sigman
    elif tipo == "QIF":
        dx = 1 - np.cos(x) + I * (1 + np.cos(x)) + np.random.randn() * sigman
    return dx

def detect(x_old, x_new, tipo, vt, vrest):
    spike = False
    if tipo == "LIF":
        if x_old < vt and x_new > vt:
            spike = True
            x_new = vrest
    elif tipo == "QIF":
        dpi = (np.pi - (x_old % (2*np.pi))) % (2*np.pi)
        if (x_new - x_old) > dpi:
            spike = True
    return x_new, spike

def simulate_single_cell(I, tipo, vt=None, vrest=0):
    x = vrest
    spikes = 0
    for _ in range(T):
        dx = dynamics(x, I, tipo)
        x_new = x + dt * dx
        x_new, spike = detect(x, x_new, tipo, vt, vrest)
        if spike:
            spikes += 1
        x = x_new
    return spikes / (T / 1000)  # frecuencia en Hz

# Corrientes
I_vals = np.linspace(0, 4, 20)
I_qif = np.linspace(0, 4, 20)

# Guardar frecuencias LIF
freqs_lif = {}
for vrest_value in v_rest:
    freqs = np.array([simulate_single_cell(I, "LIF", v_threshold, vrest=vrest_value) for I in I_vals])
    freqs_lif[vrest_value] = freqs

# Guardar frecuencias QIF
freqs_qif = np.array([simulate_single_cell(I, "QIF") for I in I_qif])

# Guardar todo en un archivo npz
np.savez('fi_data.npz', I_vals=I_vals, v_rest=v_rest, freqs_lif=freqs_lif, I_qif=I_qif, freqs_qif=freqs_qif)
print("Datos guardados en 'fi_data.npz'")
