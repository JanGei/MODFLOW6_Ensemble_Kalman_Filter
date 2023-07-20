import time
import os
import flopy
import shutil
import multiprocessing
import numpy as np

from joblib import Parallel, delayed
from gstools import Matern
from scipy.io import savemat

from functions import Directories, k_gen, error
from Objectify import Ensemble, Member, VirtualReality

if __name__ == '__main__':
    
    # =========================================================================
    #### Defining Paths & Loading Original (To be copied) Model
    # =========================================================================
    start_time      = time.perf_counter()

    absolute_path   = os.path.dirname(__file__)
    inter_path      = "/Current/BaseModel_models/MODFLOW 6"
    ens_path        = "/Current/BaseModel_models/MODFLOW 6/ensemble"
    model1L_path    = absolute_path + inter_path +"/sim1L"
    model2L_path    = absolute_path + inter_path + "/sim1t"
    
    # This Model is not to be changed during the computation
    sim_orig        = flopy.mf6.modflow.MFSimulation.load(
                            # mname, 
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = model1L_path, 
                            verbosity_level     = 0
                            )
    
    model_orig      = sim_orig.get_model()
    
    model_orig.npf.save_specific_discharge = True
    
    # check whether model is in transient mode
    # sim_orig.tdis.nper
    # model_orig.sto.transient
    # Transient conditions will apply until the STEADY-STATE
    # keyword is specified in a subsequent BEGIN PERIOD block.
    
    # Here, preliminary changes to the model can be appllied before it is copied
    
    sim_orig.write_simulation()
    
    finish_time = time.perf_counter()
    print("Original Model loaded in {} seconds - using sequential processing"
          .format(finish_time-start_time)
          )
    print("---")
    # =========================================================================
    #### Copying original Model and defining Ensemble
    # =========================================================================
    start_time      = time.perf_counter()
    
    if os.path.isdir(absolute_path + ens_path) == True :
        shutil.rmtree(absolute_path + ens_path)
    
    ncores          = multiprocessing.cpu_count()
    nreal           = 10
    
    ens_dir         = [None] * nreal
    ens_hds_dir     = [None] * nreal
    
    result = Parallel(n_jobs=ncores)(delayed(Directories)(
        model1L_path, absolute_path, inter_path, i) 
        for i in range(nreal)
        )
    
    for i in range(nreal):
        ens_dir[i]      = result[i][0]
        ens_hds_dir[i]  = result[i][1]
        
    finish_time = time.perf_counter()
    print("Ensemble directories created in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")
    
    # =========================================================================
    #### Loading Transient Data
    # =========================================================================
    start_time      = time.perf_counter()
    
    # Importing transient data
    EvT_data    = np.genfromtxt(absolute_path +'/csv data/ET.csv', delimiter=',')
    Rch_data    = np.genfromtxt(absolute_path +'/csv data/RCHunterjesingen.csv', delimiter=',')
    Qpu_data    = np.genfromtxt(absolute_path +'/csv data/Pumpingrates.csv', delimiter=',')

    # Importing locations and cells of observations and Pilot Points
    Obs_PL      = np.genfromtxt(absolute_path +'/csv data/Obs_PL.csv', delimiter=',')
    Pp_PL       = np.genfromtxt(absolute_path +'/csv data/Pp_PL.csv', delimiter=',')

    # Automate this based on well positions
    wellx, welly = zip(*[(54, 28), (71, 190)])
    wellloc = [wellx, welly]
    PPloc  = [(int(Pp_PL[i,2]), int(Pp_PL[i,3])) for i in range(len(Pp_PL))]
    PPcor  = [(int(Pp_PL[i,1]) , int((100 - (Pp_PL[i,0] - 10) /20) *20 +10)) 
              for i in range(len(Pp_PL))]
    obsloc = [(int(Obs_PL[i,2]), int(Obs_PL[i,3])) for i in range(len(Obs_PL))]
    obscor = [(int((100 - (Obs_PL[i,0] - 10) /20) *20 +10), int(Obs_PL[i,1])) 
              for i in range(len(Obs_PL))]

    n_PP   = len(PPloc)
    
    finish_time = time.perf_counter()
    print("Transient Data loaded in {} seconds - using sequential processing"
          .format(finish_time-start_time)
          )
    print("---")
    
    # =========================================================================
    #### Generating K_ensemble
    # =========================================================================
    start_time = time.perf_counter()
    
    # TODO: User input für k-feld specs
    mean    = 5
    var     = 0.5
    lx      = np.array([500, 250])
    ang     = 0

    # Covariance model of the random fields
    cov_mod = Matern(dim = 2, var = var, len_scale = lx,  angles=ang)

    # Extracting Grid
    X       = model_orig.modelgrid.get_xcellcenters_for_layer(0)
    Y       = model_orig.modelgrid.get_ycellcenters_for_layer(0)
    rows    = model_orig.modelgrid._StructuredGrid__nrow
    cols    = model_orig.modelgrid._StructuredGrid__ncol
    delr    = model_orig.modelgrid._StructuredGrid__delr
    delc    = model_orig.modelgrid._StructuredGrid__delc

    # Generating initial K-fields (1 more for VR)
    result = Parallel(n_jobs=ncores)(delayed(k_gen)(
        X, Y,
        lx, ang, 
        var,  mean,
        PPloc) 
        for i in range(nreal+1)
        )

    # Assigning results to list of K-fields and K @ Pilot points
    k_ens  = [None] * nreal
    K_PP   = np.ones((n_PP,nreal)) 
    
    for i in range(nreal):
        k_ens[i]        = result[i][0]
        K_PP[:,i]       = np.squeeze(result[i][1])
    # Setting up virtual, complex truth
    K_true              = result[-1][0]
    K_PP_true           = result[-1][1]

    finish_time = time.perf_counter()
    print("Random field generation finished in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")
    
    # =========================================================================
    #### Ensemble Initialization
    # =========================================================================
    start_time      = time.perf_counter()
    
    tstp            = 0
    t_enkf          = 300
    t_tot           = 365
        
    Ysim            = np.zeros((len(obsloc),nreal))
    Xnew            = np.zeros(((rows*cols+n_PP),nreal))
    Yobs            = np.ones((len(obsloc),nreal))
    
    Virtual_Truth = VirtualReality(model1L_path, obsloc)
    Virtual_Truth.set_kfield(cov_mod,PPcor, K_PP_true, X, Y)

    Ensemble = Ensemble(Xnew, Ysim, obsloc, PPcor, tstp, [], [], ncores)
    for i in range(nreal):
        Ensemble.add_member(Member(ens_dir[i]))
        Ensemble.members[i].set_kfield(k_ens[i])
    Ensemble.update_PP(K_PP)

    Ensemble.PP_Kriging(cov_mod, K_PP, X, Y)
    
    finish_time = time.perf_counter()
    print("Ensemble set up in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")
    
    # =========================================================================
    #### EnKF preparation & Initial Conditions for members
    # =========================================================================
    start_time = time.perf_counter()
            
    Virtual_Truth.predict(EvT_data[0,1], Rch_data[0,1], Qpu_data[0,1:3])
    Ensemble.initial_conditions(EvT_data[0,1], Rch_data[0,1], Qpu_data[0,1:3])
    Ensemble.update_hmean()

    finish_time = time.perf_counter()
    print("Initial conditions finished in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("------")
    
    # =============================================================================
    #### EnKF Data Assimilation
    # =============================================================================

    # Dampening factor for states and parameters
    damp_h          = 0.35
    damp_K          = 0.05
    damp            = np.ones((rows*cols+n_PP))
    damp[0:n_PP]    = damp_K
    damp[n_PP:]     = damp_h

    # Measurement Error variance [m2]
    eps         = 0.01

    # Defining a stressperiod directory <-- This is hard-coded
    ts          = np.ones(t_tot) 

    # Lookup-table for heads for all ensemble members and VR
    true_heads  = np.zeros((t_tot,rows,cols))
    Obs_data    = np.ones((t_tot, Ensemble.nobs))

    # Storing matrices for EnKF algorithm
    Ens_K_mean_mat    = np.ones((t_tot,rows,cols))
    Ens_h_var_mat     = np.ones((t_tot,rows,cols))
    Ens_h_mean_mat    = np.ones((t_tot,rows,cols))
    Ens_h_obs_mat     = np.ones((t_tot,len(obsloc)))
    Error_mat         = np.ones((t_tot, 3))
    
    # Mask for error
    mask_chd        = Ensemble.meanh > 1e+29
    mask_chd[:,-1]  = True
    mask_chd_3d     = np.broadcast_to(mask_chd, true_heads.shape)
    
    for i in range(t_tot):
        # Counter for stressperiod and timestep
        Ensemble.update_tstp()
        print("Starting Time Step {}".format(Ensemble.tstp))    
        
        # Setting correct transient forcing
        EvT = EvT_data[i,1]
        Rch = Rch_data[i,1]
        Qpu = Qpu_data[i,1:3]
        
        # ================== BEGIN PREDICTION STEP ============================
        
        start_time = time.perf_counter()

        Virtual_Truth.predict(EvT, Rch, Qpu) 
        Ensemble.predict(EvT, Rch, Qpu) 
                        
        finish_time = time.perf_counter()
        print("Prediction step finished in {} seconds - using multiprocessing"
              .format(finish_time-start_time)
              )
        # ================== END PREDICTION STEP ==============================
        
        
        # ================== BEGIN ANALYSIS STEP ==============================
        if i < t_enkf:
            start_time = time.perf_counter()
            
            X_prime, Y_prime, Cyy = Ensemble.analysis(eps)
            Y_obs = Virtual_Truth.pert_obs(Ensemble.nreal, Ensemble.nobs, eps)
            finish_time = time.perf_counter()
            print("Analysis step finished in {} seconds - using sequential processing"
                  .format(finish_time-start_time)
                  )
            
        # ================== END ANALYSIS STEP ================================
        
        # ================== BEGIN UPDATE STEP ================================
            start_time = time.perf_counter()
        
            # Update the ensemble 
            Ensemble.Kalman_update(damp, X_prime, Y_prime, Cyy, Y_obs)
        
            Ensemble.PP_Kriging(cov_mod, K_PP, X, Y)
        
            finish_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            for member in Ensemble.members:
                member.set_hfield(member.model.output.head().get_data())
        
        
            finish_time = time.perf_counter()
        print("Update step finished in {} seconds - using multiprocessing"
                 .format(finish_time-start_time)
                  )   
        print("---")
        
        # ================== END UPDATE STEP ==================================
        
        # ================== Post-Processing ==================================
        # Store Values
        if Virtual_Truth.model.dis.nlay.array == 1:
            true_heads[i,:,:] = Virtual_Truth.model.output.head().get_data()
        else:
            true_heads[i,:,:] = np.mean(Virtual_Truth.model.output.head().get_data(), axis = 0)
            
        for k in range(Ensemble.nobs):
            Obs_data[i,k]       = true_heads[i,obsloc[k][0],obsloc[k][1]]
            Ens_h_obs_mat[i,k]  = Ensemble.meanh[0,obsloc[k][0],obsloc[k][1]]
        
        Ens_K_mean_mat[i,:,:]   = Ensemble.meank
        Ens_h_mean_mat[i,:,:]   = Ensemble.meanh
        Ens_h_var_mat[i,:,:]    = Ensemble.get_varh()
        # Obtain model errors
        Error_mat[i,:] = error(
            true_heads[0:i+1,:,:],
            Obs_data[0:i+1,:],
            Ens_h_obs_mat[0:i+1,:],
            Ens_h_mean_mat[0:i+1,:,:],
            eps,
            Ens_h_var_mat[0:i+1,:,:],
            nreal
            )

        if i%10 == 0:
            qx_true, qy_true, head_true = Virtual_Truth.get_spdis()
            Ensemble.compare(Virtual_Truth.model.npf.k.array,
                         qx_true, qy_true, head_true)
            

    savemat("Ens_K_mean.mat", mdict={'data':Ens_K_mean_mat})
    savemat("Ens_h_mean.mat", mdict={'data':Ens_h_mean_mat})
    savemat("Ens_h_var.mat", mdict={'data':Ens_h_var_mat})
    savemat("Error.mat", mdict={'data':Error_mat})
