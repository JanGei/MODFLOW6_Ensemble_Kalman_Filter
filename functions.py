import numpy as np
import shutil
from generator import generator
# import os

def Directories(from_directory, absolute_path, inter_path, i):
    
    ens_dir      = absolute_path + inter_path + "/ensemble/m" + str(i)
    ens_hds_dir  = absolute_path + inter_path + "/ensemble/m" + str(i) \
                                    + "/flow_output/flow.hds"
    shutil.copytree(from_directory, ens_dir)
    
    return ens_dir, ens_hds_dir

def k_gen(X, Y, lx, ang, sigma2,  mu, PP_cell):

    k = generator(X, Y, lx, ang, sigma2, mu)
    Zone_K  = np.ones((len(PP_cell),1))
    
    for j in range(len(PP_cell)):
        Zone_K[j,0] = k[PP_cell[j][0], PP_cell[j][1]]
        
    return k, Zone_K


def NRMSE(h_true, h_sim, sigma, case, ens_n):
    if case == 1:
        nt = h_true.shape[0]  
        nc = h_true.shape[1]
        er = np.sqrt(1/(nt*nc) * np.sum(np.sum((h_true - h_sim)**2 / sigma)))
    # mask this shizzle for better overview
    elif case == 2:
        nt = h_true.shape[0]  
        nc = h_true.shape[1]*h_true.shape[2]
        
        
        er = np.sqrt(1/(nt*nc) * np.sum(np.sum((h_true - h_sim)**2 / sigma)))
    return er

def error(h_true, h_true_obs, h_obs, h_sim, sigma, Ens_var, ens_n):
    
    ole = NRMSE(h_true_obs, h_obs, sigma, 1, ens_n)
    
    Ens_var = np.ma.masked_equal(Ens_var, 0)
    
    te1 = NRMSE(np.ma.masked_where(Ens_var.mask, h_true),
                np.ma.masked_where(Ens_var.mask, h_sim),
                Ens_var,
                2,
                ens_n)
    
    te2 = NRMSE(np.ma.masked_where(Ens_var.mask, h_true),
                np.ma.masked_where(Ens_var.mask, h_sim),
                np.ma.masked_where(Ens_var.mask, 0.5*(h_true+h_sim)),
                2, 
                ens_n)
    
    return ole, te1, te2

    
# def plotOLE(n_obs, obs_true, obs_sim, t_enkf, sigma, dir_ole):
    
#     for i in range(n_obs):   
#         nt = 365 
#         nc = 1
#         er = np.sqrt(1/(nt*nc) * np.sum(np.sum((obs_true[:,i] - obs_sim[:,i])**2 / sigma)))
#         plt.figure()
#         plt.title("Truth vs. Simulated at obs. " + str(int(i+1)) + ' with OLE = ' + str("%.2f" % er))
#         plt.plot(obs_true[:,i], label = 'Truth')
#         plt.plot(obs_sim[:,i], label = 'Simulated')
#         plt.legend()
#         plt.vlines(t_enkf, 
#                    np.min((np.min(obs_true[:,i]),np.min(obs_true[:,i]))), 
#                    np.max((np.max(obs_true[:,i]),np.max(obs_sim[:,i]))), 
#                    colors='k', 
#                    linestyles='solid')
#         plt.savefig(dir_ole  + str(int(i+1)) + '.png')

# def saveRes(ole, te1, te2, true_heads, K_true, Ens_h_mean_mat, Ens_h_var_mat,
#             Ens_lnK_mean_mat, Ens_lnK_var_mat, name, n_obs, Y_sim,
#             tstp, t_enkf, sigma, obsYcell, obsXcell):
    
#     # Print errors to result folder
#     dir_res = 'Results/'+str(name)
#     dir_ole = dir_res + '/OLE/'
#     os.mkdir(dir_res)
#     os.mkdir(dir_ole)
#     with open(dir_res +'/Errors.txt', 'w') as f:
#         f.write('OLE' + str(ole) + '\n TE1' + str(te1) +'\n TE2' + str(te2))
#         ###CONTINUE HERE
#     with open(dir_res +'/Geometry.txt', 'w') as f:
#         f.write('OLE' + str(ole) + '\n TE1' + str(te1) +'\n TE2' + str(te2))
     
#     obs_sim = np.mean(Y_sim, axis = 2)
#     obs_true = np.ones(np.shape(obs_sim))

#     for i in range(tstp):
#         for j in range(n_obs):
#             obs_true[i,j] = true_heads[i,int(obsYcell[j]),int(obsXcell[j])]
#     plotOLE(n_obs, obs_true, obs_sim, t_enkf, sigma, dir_ole)
        