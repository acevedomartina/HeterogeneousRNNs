import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr
import pandas as pd
import csv
import os




def crear_subcarpeta(nombre_carpeta, nombre_subcarpeta):
    
    subcarpeta_path_total = (os.path.join(nombre_carpeta, nombre_subcarpeta))
    if not os.path.exists(subcarpeta_path_total):
        os.makedirs(subcarpeta_path_total)
    
    return subcarpeta_path_total


def crear_subcarpeta(carpeta_padre, nombre_subcarpeta):
    ruta = os.path.join(carpeta_padre, nombre_subcarpeta)
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    return ruta

def crear_carpetas(num_simulacion): 
    # Carpeta principal de la simulación
    nombre_carpeta = f"simulacion_{num_simulacion}"
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)

    # Subcarpetas dentro de la simulacion
    sub_act = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_ejemplos_actividad")
    sub_pesos = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_matrices_pesos")
    sub_corrientes = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_corrientes")
    sub_inputs = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_inputs")
    sub_outputs = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_outputs")
    sub_nspikes = crear_subcarpeta(nombre_carpeta, f"simulacion_{num_simulacion}_nspikes")

    return nombre_carpeta, sub_act, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes


N = 200                                   # numero de neuronas
N2 = int(N/2)

#Conexiones sinápticas
p =  0.3                                 # probabilidad de elementos no nulos en la matriz de pesos
gsyn = 0.5                                # peso sinaptico inicial

alpha = 0.25                              # regularizador pesos

#Dinámica
dt = 0.1                                  # paso de tiempo (escala de tiempo 10 ms)
itmax = 1000                              # numero de iteraciones (1000 -> 1 sec)              
sigman = 1                                # Noise standard deviation -> ruido en la dinámica


#b_decay = 0.1
#Estímulo
itstim = 200                              # tiempo de estimulo
amp_corriente = 20                     # intensidad estímulo
amp0 = 4                            # Changed to 4 (was 8)

iout = np.linspace(0,N,num=N,endpoint=False).astype('int')
igraph=np.array((0,1,2,N2+1,N2+2,N2+3))   # indices a graficar                                 # para guardar 10 salidas



#romega1 = 1                                # cociente frecuencia alta/baja
#romega2 = 5



#ENTRENAMIENTO
ftrain = 1                                # fraccón de neuronas a entrenar
nloop  = 16                              # numero de loops, 0 pre-entramiento, ultimo: post-entrenamiento. Poner nloop=2 para no hacer aprendizaje
nloop_train = 10                         #ultimo loop de entrenamiento

cant_seed = 50




def crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta, b, vt, vrest):
 #file donde guardo los parámetros de la simulación
    data_parametros = {
        'N': [N],
        'p': [p],
        'gsyn': [gsyn],
        'nloop': [nloop],
        'nloop_train':[nloop_train],
        'cant_seed': [cant_seed],
        'dt': [dt],
        'itmax': [itmax],
        'itstim': [itstim],
        'amp_corriente': [amp_corriente],
        'amp0': [amp0],
        'ftrain': [ftrain],
        'alpha': [alpha],
        'sigman': [sigman],
        'vt': [vt],
        'b': [b],
        'vrest': [vrest],
        'results_file': [filename_resultados],
    }


    df = pd.DataFrame(data_parametros)
    filename_parametros = f'simulacion_{num_simulacion}_parametros.csv'
    csv_parametros_path = os.path.join(nombre_carpeta, filename_parametros)
    df.to_csv(csv_parametros_path, index=False)


amp_ci_target = 0.5

def generate_target(num_simulacion, nombre_carpeta,sg_index, amp0, pqif):

    target = np.zeros(shape=(N,itmax))
    #para secuencias periodicas

    
    gg=np.zeros(N)
    sg=sg_index*N            # ancho de la gaussiana. trelativo al tamanio del sistema
    omegagauss=0.1       # velocidad de desplazamiento
    for i in range(N):
        gg[i]=amp0*np.exp(-(i-N/2)**2/(2*sg**2))
    for it in range(itmax):
        target[:,it]=np.roll(gg,int(omegagauss*it))

    target_df = pd.DataFrame(target.T, columns=[f'Neurona_{i}' for i in range(N)])
    nombre_archivo_target = f'simulacion_{num_simulacion}_targets_{pqif}.csv'
    csv_target_path = os.path.join(nombre_carpeta, nombre_archivo_target)
    target_df.to_csv(csv_target_path, index=False)

    data = {'sg_index': sg_index, 'omegagauss': omegagauss, 'amp0': amp0}
    df = pd.DataFrame(data, index=[0])
    nombre_archivo = f'simulacion_{num_simulacion}_targets_parametros.csv'
    csv_target_path = os.path.join(nombre_carpeta, nombre_archivo)
    df.to_csv(csv_target_path, index=False)
            
    return target 






def guardar_matriz_csv(matriz, nombre_archivo):
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        for fila in matriz:
            fila_lista = [str(elemento) for elemento in fila.flat]
            escritor_csv.writerow(fila_lista)
            
def motifs(w,gsyn,N):
    
    w=w-np.mean(w)
    
    ww=np.matmul(w,w)
    wtw=np.matmul(w.T,w)
    wwt=np.matmul(w,w.T)
    
    sigma2=np.trace(wwt)/N
    
    tau_rec=np.trace(ww)
    tau_rec/=sigma2*N
    
    tau_div=np.sum(wwt)-np.trace(wwt)
    tau_div/=sigma2*N*(N-1)
    
    tau_con=np.sum(wtw)-np.trace(wtw)
    tau_con/=sigma2*N*(N-1)
    
    tau_chn=2*(np.sum(ww)-np.trace(ww))
    tau_chn/=sigma2*N*(N-1)
    
    return sigma2,tau_rec,tau_div,tau_con,tau_chn

def dynamics(x_var,r_var,I_var,nqif, b):
    dx=np.zeros(N)
    I_noise_lif = np.random.randn(N - nqif)*sigman 
    I_noise_qif = np.random.randn(nqif)*sigman
    #LIF
    dx[nqif:] = -x_var[nqif:] + I_var[nqif:] + I_noise_lif
    #QIF
    dx[:nqif] = 1 - np.cos(x_var[:nqif]) + I_var[:nqif]*(1 + np.cos(x_var[:nqif])) + I_noise_qif
    dr = -b*r_var
    return dx,dr


def detect(x,xnew,rnew,nspike,nqif, b, vt, vrest):
     #LIF
     ispike_lif=np.where(x[nqif:]<vt) and np.where(xnew[nqif:]>vt)
     ispike_lif=ispike_lif[0]+nqif
     if(len(ispike_lif)>0):
         rnew[ispike_lif[:]] = rnew[ispike_lif[:]] + b
         xnew[ispike_lif[:]] = vrest
         nspike[ispike_lif[:]] = nspike[ispike_lif[:]] + 1
     #QIF 
     dpi=np.mod(np.pi - np.mod(x,2*np.pi),2*np.pi)  # distancia a pi
     ispike_qif=np.where((xnew[:nqif]-x[:nqif])>0) and np.where((xnew[:nqif]-x[:nqif]-dpi[:nqif])>0)
     if(len(ispike_qif)>0):
         rnew[ispike_qif[:]] = rnew[ispike_qif[:]] + b
         nspike[ispike_qif[:]] = nspike[ispike_qif[:]] + 1
     return xnew,rnew,nspike

def evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt, vrest):
    II = np.squeeze(np.asarray(Iext[:, it]))
    v = w.dot(r.T).A1
    dx, dr = dynamics(x, r, II + v, nqif, b)
    xnew = x + dt * dx / 2
    rnew = r + dt * dr / 2
    dx, dr = dynamics(xnew, rnew, II + v, nqif, b)
    xnew = x + dt * dx
    rnew = r + dt * dr
    xnew, rnew, nspike = detect(x, xnew, rnew, nspike, nqif, b, vt, vrest)
    x, r = np.copy(xnew), np.copy(rnew)


    return x, r, nspike, r[iout], II, v


def dpr_bias(ccorr,N,nloop):
    a=np.extract(np.identity(N),ccorr)
    c=np.extract(1-np.identity(N),ccorr)
    am2=np.mean(a)**2
    astd2=np.var(a)*N/(N-1)
    cm2=np.mean(c)**2
    cstd2=np.var(c)*N*(N-1)/(N*(N-1)-2)
    
    astd_bias2=astd2*(nloop-1)/(nloop+1) -2*(am2-cm2)/(nloop-1)+ 2*cstd2/(nloop+1)
    cstd_bias2=(nloop-1)*cstd2/nloop - (am2-cm2)/nloop -4*(cm2-np.sqrt(am2*cm2))/(nloop*(N+1))
    
    dpr_bias=N/(1+(astd_bias2/am2)+(N-1)*((cstd_bias2/am2)+(cm2/am2)))
    
    return dpr_bias



def initialize_connectivity_matrix(N, p, gsyn):
    w = sparse.random(N, N, p, data_rvs=np.random.randn).todense()
    np.fill_diagonal(w, 0)  # No autapses
    w *= gsyn / np.sqrt(p * N)
    
    for i in range(N):
        i0 = np.where(w[i, :])[1]
        if len(i0) > 0:
            av0 = np.sum(w[i, i0]) / len(i0)
            w[i, i0] -= av0
    
    return w

def initialize_neurons(N):
    x = np.random.uniform(size=N) * 2 * np.pi
    r = np.zeros(N)
    nspike = np.zeros(N)

    return x, r, nspike

def initialize_training(N, w):
    # matrices de correlacion de las entradas
    nind=np.zeros(N).astype('int')
    idx=[]
    P=[]
    for i in range(N):
        ind=np.where(w[i,:])[1]
        nind[i]=len(ind)
        idx.append(ind)
        P.append(np.identity(nind[i])/alpha)   
    return P, idx

def currents(N, itmax):
    Iext=np.zeros((N,itmax))
    Ibac=amp_corriente*(2*np.random.uniform(size=N)-1)
    Iext[:, :itstim] = Ibac[:, None]  # Vectorized assignment

    return Iext


def learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer):
    error = target[:, it:it + 1] - w @ r.reshape(N, 1)
    for i in range(N):
        ri = r[idx[i]].reshape(len(idx[i]), 1)
        k1 = P[i] @ ri
        k2 = ri.T @ P[i]
        den = 1 + ri.T @ k1
        P[i] -= (k1 @ k2) / den
        dw = error[i, 0] * P[i] @ r[idx[i]]
        w[i, idx[i]] += dw

    if it % 10 == 0:
        modt_value = it + iloop * itmax
        modw_value = np.log(np.linalg.norm(w) / norm_w0)
        
        # Guardar los valores en el archivo CSV
        csv_writer.writerow([modt_value, modw_value])
        
    return w, P

amp0=8
sg_index = 0.15
ts= 5
b = 1 / ts


########## MAIN SIMULATION LOOP ##########

for pqif in [0, 0.25, 0.5, 0.75, 1]:  # Included all pqif compositions
    num_simulacion = 8  # Starting point for simulation folders (will be 9-12)
    if pqif==1:
        vt_vect = [None]
    else:
        vt_vect = [0]  # LIF


    for vt in vt_vect:
        if vt==0:
            vrest_vect = [-8.5, -12.3, -17, -22]  # all vrest values (one per simulation)
        if vt == None:
            vrest_vect = [None, None, None, None]  # QIF is included in all simulations

        
        for vrest in vrest_vect:
            num_simulacion +=1
            nombre_carpeta, sub_act, sub_pesos, sub_corrientes, sub_inputs, sub_outputs, sub_nspikes = crear_carpetas(num_simulacion)


            target = generate_target(num_simulacion, nombre_carpeta,sg_index, amp0, pqif=pqif)
            #file donde voy a guardar los resultados (CC, taus)
            filename_resultados = f'simulacion_{num_simulacion}_resultados.csv'
            csv_file_path = os.path.join(nombre_carpeta, filename_resultados)
            column_names = [ 'pqif' ,'seed','nloop', 'sigma2', 'tau_rec','tau_div','tau_con','tau_chn']


            crear_archivo_parametros(filename_resultados, num_simulacion, nombre_carpeta, b, vt, vrest=vrest)
                

            cant_seed = 50

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(column_names)
                
        
                    
                path_corrientes_seed = os.path.join(sub_corrientes, f'simulacion_{num_simulacion}_corrientes_pqif_{pqif}.csv')

                with open(path_corrientes_seed, mode='a', newline='') as file_:
                    writer_ = csv.writer(file_)
                    if file_.tell() == 0:
                        writer_.writerow(['pqif', 'seed', 'iloop', 'it', 'II_0', 'v_0', 'Inoise_0', 'II_1', 'v_1', 'Inoise_1', 'II_N2+1', 'v_N2+1','Inoise_N2+1', 'II_N2+2', 'v_N2+2', 'Inoise_N2+2'])

                    nqif = int(N * pqif)
                    print(nqif)

                    for seed in range(cant_seed):


                        np.random.seed(seed = seed)

                        x, r, nspike = initialize_neurons(N)

                                                    

                        # corriente externa
                        Iext= currents(N, itmax)
                        path_corrientes_seed = os.path.join(nombre_carpeta, f'simulacion_{num_simulacion}_Iext_pqif_{pqif}_seed_{seed}.csv')
                        guardar_matriz_csv(Iext, path_corrientes_seed)

                        w = initialize_connectivity_matrix(N, p, gsyn)

                        norm_w0 = np.linalg.norm(w)

                        P, idx = initialize_training(N, w)

                        filename_dw = os.path.join(nombre_carpeta, f'simulacion_{num_simulacion}_dw_pqif_{pqif}_seed_{seed}.csv')

                        with open(filename_dw, mode='w', newline='') as file:
                            csv_writer_dw = csv.writer(file)
                            csv_writer_dw.writerow(['modt', 'modw'])  # Escribir encabezados

                            for iloop in range(nloop):
                                
                                path_inputs= os.path.join(sub_inputs, f'simulacion_{num_simulacion}_inputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                                path_nspikes = os.path.join(sub_nspikes, f'simulacion_{num_simulacion}_nspikes_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                                path_outputs= os.path.join(sub_outputs, f'simulacion_{num_simulacion}_outputs_pqif_{pqif}_iloop_{iloop}_seed_{seed}.csv')
                                
                                for it in range(itmax):
                                    nspike = np.zeros(N)

                                    x, r, nspike, rout, II, v = evolution(x, r, Iext, w, nqif, it, dt, iout, nspike, b, vt=vt, vrest=vrest)

                                    entrada = II +v

                                    rout_row = rout.reshape(1, -1)
                                    entrada_row = entrada.reshape(1,-1)



                                    if iloop in [nloop_train + 1, nloop - 1] and it % 20 == 0:
                                        writer_.writerow([pqif, seed, iloop, it, II[0], v[0], II[1], v[1], II[N2+1], v[N2+1], II[N2+2], v[N2+2]])
                                    
                                    # aprendizaje
                                    if  iloop>0  and iloop <= nloop_train and int(it>itstim):
                                        w, P = learning(it, iloop, w, r, P, idx, target, norm_w0, csv_writer_dw)



                                    nspike_row = np.array(nspike).reshape(1, -1)
                                    


                                    # Guardar `rout` como una fila en el CSV
                                    with open(path_nspikes, 'a') as f:
                                        np.savetxt(f, nspike_row, delimiter=',')


                                    # Guardar `rout` como una fila en el CSV
                                    with open(path_inputs, 'a') as f:

                                        np.savetxt(f, entrada_row, delimiter=',')

                                    # Guardar `rout` como una fila en el CSV
                                    with open(path_outputs, 'a') as f:

                                        np.savetxt(f, rout_row, delimiter=',')
                                
                                sigma2,tau_rec,tau_div,tau_con,tau_chn=motifs(w,gsyn,N)
                                


                                if iloop == 0 or iloop == (nloop_train + 1):
                                    path_w_seed = os.path.join(sub_pesos, f'simulacion_{num_simulacion}_pesos_pqif_{pqif}_matriz_iloop_{iloop}_semilla_{seed}')
                                    guardar_matriz_csv(w, path_w_seed)



                                writer.writerow([
                                    pqif,
                                    seed,
                                    iloop,
                                    sigma2,
                                    tau_rec,
                                    tau_div,
                                    tau_con,
                                    tau_chn, 
                                        
                                
                                ])
                                
