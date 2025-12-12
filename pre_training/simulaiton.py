import numpy as np
import pandas as pd
from scipy import sparse
import os



# Parámetros de simulación
T = 1000
dt = 0.1
sigman = 0.0
v_threshold = 0
v_rest = [-22, -12.3, -17, -8.5]
#v_rest = [-12.3, -17]
path = os.getcwd()

N = 200                                   # numero de neuronas
N2 = int(N/2)

# probabilidad de elementos no nulos en la matriz de pesos
p = 0.3

# inversa constante de tiempo sinaptica  (escala de tiempo 10 ms)
b = 0.5

# paso de tiempo (escala de tiempo 10 ms)
dt = 0.1
# numero de iteraciones (1000 -> 1 sec)
itmax = 50
nloop = 2000

sigman = 1


iout = np.array((0, N2, N-1))
pqif_vector = [0, 1, 0.25, 0.5, 0.75]  # 0 -> LIF, 1 -> QIF, 0.5 -> MIX

gsyn_vect = [0]
Iext = np.zeros((N, itmax))


outdir = 'cov_matrices'
column_names = ['vrest', 'gsyn', 'dpr', 'dpr_bias', 'frequency', 'seed']
if not os.path.exists(outdir):
    os.makedirs(outdir)

############ FUNCIONES ##############

def initialize_weights(N, p, gsyn):
    # definicion matriz de conectividad
    w = (sparse.random(N, N, p, data_rvs=np.random.randn)
            ).todense()    # matriz de conexiones
    # no autapses
    np.fill_diagonal(w, 0)
    # normalizacion
    w *= gsyn/np.sqrt(p*N)
    # suma de filas -> 0
    for i in range(N):
        i0 = np.where(w[i, :])[1]
        av0 = 0
        if(len(i0) > 0):
            av0 = np.sum(w[i, i0])/len(i0)
            w[i, i0] = w[i, i0]-av0
    return w


def dynamics(x_var,r_var,I_var, nqif, b):
    dx=np.zeros(N)

    #LIF
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] 
    #QIF
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif]))
    dr = -b*r_var
    return dx,dr


def detect(x,xnew,rnew,nspike,nqif, b, vt, vrest):
     #LIF
     ispike_lif = np.where((x[nqif:] < vt) & (xnew[nqif:] > vt))[0]+nqif
     if(len(ispike_lif)>0):
         rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
         xnew[ispike_lif[:]] = vrest
         nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
     #QIF 
     dpi=np.mod(np.pi - np.mod(x,2*np.pi),2*np.pi)  # distancia a pi
     ispike_qif = np.where(((xnew[:nqif] - x[:nqif]) > 0) & ((xnew[:nqif] - x[:nqif] - dpi[:nqif]) > 0))[0]
     if(len(ispike_qif)>0):
         rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
         nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
     return xnew,rnew,nspike


def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vrest):
    II = np.squeeze(np.asarray(Iext[:, it]))
    I_noise = np.random.randn(N)*sigman 

    v = w.dot(r.T).A1
    dx, dr = dynamics(x, r, II + v + I_noise, nqif, b)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr = dynamics(xnew, rnew, II + v + I_noise , nqif, b)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vrest)
    x[:] = xnew
    r[:] = rnew


    return x, r, nspike, r[iout], II, v, I_noise

########## SIMULACION ##############


for pqif in pqif_vector:  # 0 -> LIF, 1 -> QIF, 0.5 -> MIX

    if pqif == 1:
        vt_vect = [None]
    else:
        vt_vect = [0]  # LIF

    for vt in vt_vect:
        if vt == 0:
            vrest_vect = v_rest
        if vt == 0.5:
            vrest_vect = [-1, -3.1, -8.5]
        if vt is None:
            vrest_vect = [None]

        for vrest in vrest_vect:

            for seed in range(1):
                print(f"Simulacion pqif={pqif}, vt={vt}, vrest={vrest}, seed={seed}")
                np.random.seed(seed)

                nqif = int(N * pqif)
                dpr_vect = np.array([])
                frecuency = np.array([])
                
                
                x = np.random.uniform(size=N) * 2 * np.pi
                r = np.zeros(N)

                for gsyn in gsyn_vect:
                    w = initialize_weights(N, p, gsyn)
                    
                    I_entrada = np.zeros(itmax)

                    nspike = np.zeros(N)

                    
                    rprom = np.zeros(shape=(N, nloop))
                    rprom_tot = 0
    

                    for iloop in range(nloop):

                        for it in range(itmax):
                            x, r, nspike, rout, II, v, I_noise = evolution(
                                x, r, Iext, w, nqif, it, dt, iout, nspike, b,
                                vt=vt, vrest=vrest
                            )
        
                            rprom[:, iloop] += r

                    rprom /= itmax
                    rprom_tot = np.mean(rprom, axis=1)
                    


                    ccorr = np.cov(rprom, bias=False, rowvar=True)

                    # Sanitizar vrest y vt para nombre de archivo
                    vrest_str = "None" if vrest is None else str(vrest)
                    vt_str = "None" if vt is None else str(vt)

                    fname = f"prueba_cov_pqif_{pqif}_vt_{vt_str}_vrest_{vrest_str}_gsyn_{gsyn:.2f}_seed_{seed}.npy"
                    fpath = os.path.join(outdir, fname)


                    # Asegurarse de que no hay NaN/Inf
                    if np.isfinite(ccorr).all():
                        np.save(fpath, ccorr)
                    else:
                        print(f"Advertencia: ccorr contiene NaN/Inf para {fpath}. No se guarda.")
