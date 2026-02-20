import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os



matplotlib.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 20,  
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})


N = 200

itmax = 1000
itstim = 200
nloop = 16
ult_loop = 15
correlations_df = pd.DataFrame(columns=['idx','pqif', 'b','CC_outputs', 'CC_inputs', 'tau_rec','tau_div', 'tau_con', 'tau_chn'])

for pqif in [0, 0.5, 1]:

    
    for idx in [1,2]:
        
        if pqif == 1 and idx==2: 
            continue

        parametros = f'simulation_{idx}/simulation_{idx}_parametros.csv'
        parametros_df = pd.read_csv(parametros)
        b_values = parametros_df['b'].values
        

        target_csv = pd.read_csv(f'simulation_{idx}/simulation_{idx}_targets_{pqif}.csv')
        targets = target_csv.values

        parametros_target = f'simulation_{idx}/simulation_{idx}_targets_parameters.csv'
        parametros_target_df = pd.read_csv(parametros_target)



        cc_outputs= np.zeros(5)
        cc_inputs= np.zeros(5)

   
        for iloop in range(11,nloop):

            file_inputs = f'simulation_{idx}/simulation_{idx}_inputs/simulation_{idx}_inputs_pqif_{pqif}_iloop_{iloop}_seed_0.csv'

            df_inputs = pd.read_csv(file_inputs, header=None)
            df_inputs = df_inputs.values
            ci_inputs = 0


            file_outputs = f'simulation_{idx}/simulation_{idx}_outputs/simulation_{idx}_outputs_pqif_{pqif}_iloop_{iloop}_seed_0.csv'
            df_outputs = pd.read_csv(file_outputs, header=None)
            df_outputs = df_outputs.values
            ci_outputs = 0

            for i in range(N):

                rout_i = df_outputs[:,i]
                rin_i = df_inputs[:,i]
  
        
                if np.var(targets[itstim:, i]) > 0 and np.var(rout_i[itstim:]) > 0:
                    #print(targets[itstim:, i].shape, rout_i[itstim:].shape)
                    ci_outputs += pearsonr(targets[itstim:, i], rout_i[itstim:])[0]/N
                    ci_inputs += pearsonr(targets[itstim:, i], rin_i[itstim:])[0]/N

            cc_outputs[iloop-11] = ci_outputs
            cc_inputs[iloop-11] = ci_inputs



        nombre_archivo = f'simulation_{idx}/simulation_{idx}_results.csv'
        data = pd.read_csv(nombre_archivo)
        data_loop = data[(data['nloop'] == 15)]
        df_filtered = data_loop[(data_loop['pqif'] == pqif)]

      

        temp_df = pd.DataFrame({'idx':[idx], 'pqif':[pqif],'b':[b_values[0]], 'CC_outputs': [np.mean(cc_outputs)], 'CC_inputs': [np.mean(cc_inputs)],'tau_rec': df_filtered['tau_rec'].values[0], 'tau_div': df_filtered['tau_div'].values[0],'tau_con': df_filtered['tau_con'].values[0],'tau_chn': df_filtered['tau_chn'].values[0]})


        if correlations_df.empty:
            correlations_df = temp_df.copy()
        else:
            correlations_df = pd.concat([correlations_df, temp_df], ignore_index=True)





new_data = correlations_df


filename = f'CC_tau.csv'                                           

if os.path.exists(filename):

    existing_data = pd.read_csv(filename)
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
else:
    combined_data = new_data

combined_data.to_csv(filename, index=False)