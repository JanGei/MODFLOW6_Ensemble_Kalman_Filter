import numpy as np
import flopy 
from joblib import Parallel, delayed
from gstools import krige
import matplotlib.pyplot as plt

class Ensemble:
    def __init__(self, X, Ysim, obsloc, PPcor, tstp, meanh, meank, ncores):
        self.ncores = ncores
        self.X      = X
        self.nreal  = X.shape[1]
        self.Ysim   = Ysim
        self.nobs   = Ysim.shape[0] #CHECK whetehr this is true
        self.obsloc = obsloc
        self.PPcor  = PPcor
        self.nPP    = len(PPcor)
        self.tstp   = tstp
        self.meanh  = meanh
        self.meank  = meank
        self.members = []
        
    def update_tstp(self):
        self.tstp += 1 
        
    # Add Member to Ensemble
    def add_member(self, member):
        self.members.append(member)
        
    def remove_member(self, member, j):
        self.members.remove(member)
        np.delete(self.X,       j, axis = 1)
        np.delete(self.Ysim,    j, axis = 2)
        np.delete(self.Yobs,    j, axis = 1)
        self.nreal -= 1
    
    def initial_conditions(self, EvT, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                EvT, Rch, Qpu) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                self.members[j].set_hfield(np.squeeze(result[j]))
                        
                j = j + 1
        
    # Propagate Entire Ensemble
    def predict(self, EvT, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                EvT, Rch, Qpu) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                self.X[self.nPP:,j]  = np.ndarray.flatten(result[j])
                
                for k in range(self.nobs):
                    self.Ysim[k,j]  = self.members[j].model.output.head().get_data()[0,self.obsloc[k][0],self.obsloc[k][1]]

                        
                j +=  1
    
    def update_hmean(self):
        newmean = np.zeros(self.members[0].model.output.head().get_data().shape)
        for member in self.members:
            newmean += member.model.ic.strt.array
        self.meanh = newmean / self.nreal
        
    def update_kmean(self):
        newmean = np.zeros(self.members[0].model.npf.k.array.shape)
        for member in self.members:
            newmean += member.model.npf.k.array
        self.meank = newmean / self.nreal
        
    def get_varh(self):
        return  np.reshape(np.var(self.X[self.nPP:], axis = 1), self.members[0].model.npf.k.array.shape)    
    
    def update_PP(self, PPk):
        for j in range(self.nreal):
            self.X[0:self.nPP, j] = PPk[:,j]
               
    def analysis(self, eps):
        
        # Compute mean of postX and Y_sim
        Xmean   = np.tile(np.array(np.mean(self.X, axis = 1)).T, (self.nreal, 1)).T
        Ymean   = np.tile(np.array(np.mean(self.Ysim,  axis  = 1)).T, (self.nreal, 1)).T
        
        # Fluctuations around mean
        X_prime = self.X - Xmean
        Y_prime = self.Ysim  - Ymean
        
        # Variance inflation
        # priorX  = X_prime * 1.01 + Xmean
        
        # Measurement uncertainty matrix
        R       = np.identity(self.nobs) * eps 
        
        # Covariance matrix
        Cyy     = 1/(self.nreal-1)*np.matmul((Y_prime),(Y_prime).T) + R 
                        
        return X_prime, Y_prime, Cyy
    
    def Kalman_update(self, damp, X_prime, Y_prime, Cyy, Y_obs):
        
        self.X += 1/(self.nreal-1) * (damp *
                    np.matmul(
                        X_prime, np.matmul(
                            Y_prime.T, np.matmul(
                                np.linalg.inv(Cyy), (Y_obs - self.Ysim)
                                )
                            )
                        ).T
                    ).T
        
        for j in range(len(self.members)):
            self.members[j].set_hfield(np.reshape(self.X[self.nPP:,j],self.members[j].model.ic.strt.array.shape))
        
        self.update_hmean()
        
    def PP_Kriging(self, cov_mod, PP_K, X, Y):
        
        Parallel(n_jobs=self.ncores)(delayed(self.members[j].updateK)(
                cov_mod, self.PPcor, PP_K[:,j], X, Y) for j in range(len(self.members))
                )
        
        self.update_kmean()
        
    def get_obs(self):
        obs = [self.meanh[0, self.obsloc[i][0], self.obsloc[i][1]] for i in range(len(self.obsloc))]
        return obs
    
    def compare(self, k_field_true, qx_true, qy_true, head_true):
        
        result  = Parallel(n_jobs=self.ncores)(delayed(self.members[j].get_spdis)
                                              () for j in range(len(self.members))
                )
        
        qx_l = [result[i][0] for i in range(len(result))]
        qy_l = [result[i][1] for i in range(len(result))]
        
        qx = np.sum(qx_l, axis=0)
        qy = np.sum(qy_l, axis=0)
        
        h_diff = head_true-self.meanh
        h_diff[self.meanh == 1e+30] = 1e+30
        vmin = np.min([self.meanh[self.meanh > -0.1].min(), head_true[head_true > -0.1].min()])
        vmax = np.max([self.meanh[self.meanh < 500].max(), head_true[head_true < 500].max()])
        
        print("Min Head is" + str(vmin))
        print("Max Head is" + str(vmax))
        
        k_field                         = self.meank
        k_field                         = np.log10(k_field)
        k_field_true                    = np.log10(k_field_true)
        k_diff                          = (k_field_true-k_field)/k_field_true
        k_field[self.meanh== 1e+30]     = 1e+30
        k_field_true[self.meanh== 1e+30]= 1e+30
        k_diff[self.meanh== 1e+30]      = 1e+30           
        vmink                           = np.min([k_field.min(), k_field_true.min()])
        vmaxk                           = np.max([k_field[k_field < 86400].max(), k_field_true[k_field_true < 86400].max()])
        
        fig1, axes1 = plt.subplots(3, 1, figsize=(25, 25), sharex=True)
        ax11, ax12, ax13 = axes1

        ax11.set_title("Ensemble-mean, true, and difference K-field in period " + str(int(self.tstp)), fontsize = 30)
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax11)
        ax11.set_aspect("equal")
        mapable = pmv.plot_array(k_field.flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
        im_ratio = k_field.shape[1]/k_field.shape[2]
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax11)
        cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax11.yaxis.label.set_size(25)
        ax11.xaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax11.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        pmv.plot_vector(qx, qy, width=0.0008, color="black")
        
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax12)
        ax12.set_aspect("equal")
        mapable = pmv.plot_array(k_field_true.flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax12)
        cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax12.yaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax12.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        pmv.plot_vector(qx_true, qy_true, width=0.0008, color="black")
        
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax13)
        ax13.set_aspect("equal")
        mapable = pmv.plot_array(k_diff.flatten(), cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax13)
        cbar.ax.set_ylabel('Relative Difference [-]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax13.yaxis.label.set_size(25)
        ax13.xaxis.label.set_size(25)
        plt.xlabel('Easting [m]', fontsize = 25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax13.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        
        plt.savefig("K_field in t" + str(self.tstp), format="svg")
        
        fig2, axes2 = plt.subplots(3, 1, figsize=(25, 25), sharex=True)
        ax21, ax22, ax23 = axes2

        ax21.set_title("Ensemble-mean, true, and difference h-field in period " + str(int(self.tstp)), fontsize = 30)
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax21)
        ax21.set_aspect("equal")
        pmv.contour_array(
                    self.meanh, masked_values = 1e+30, levels=np.arange(vmin, vmax, 0.1), linewidths=2.0, vmin=vmin, vmax=vmax
                )
        mapable = pmv.plot_array(self.meanh, cmap="RdBu", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax21)
        cbar.ax.set_ylabel('Hydraulic Head [m]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax21.yaxis.label.set_size(25)
        ax21.xaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax21.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax22)
        ax22.set_aspect("equal")
        pmv.contour_array(
                    head_true, masked_values = 1e+30, levels=np.arange(vmin, vmax, 0.1), linewidths=2.0, vmin=vmin, vmax=vmax
                )
        mapable = pmv.plot_array(head_true, cmap="RdBu", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax22)
        cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax22.yaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax22.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax23)
        ax23.set_aspect("equal")
        mapable = pmv.plot_array(h_diff, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax23)
        cbar.ax.set_ylabel('Relative Difference [-]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax23.yaxis.label.set_size(25)
        ax23.xaxis.label.set_size(25)
        plt.xlabel('Easting [m]', fontsize = 25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax23.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        
        plt.savefig("Heads in t" + str(self.tstp), format="svg")
        
    # def plot(self):
        
    #     result  = Parallel(n_jobs=self.ncores)(delayed(self.members[j].get_spdis)
    #                                           () for j in range(len(self.members))
    #             )
        
    #     qx_l = [result[i][0] for i in range(len(result))]
    #     qy_l = [result[i][1] for i in range(len(result))]
        
    #     qx = np.sum(qx_l, axis=0)
    #     qy = np.sum(qy_l, axis=0)
        
    #     vmin = self.meanh[self.meanh > -1e+30].min()
    #     vmax = self.meanh[self.meanh < 1e+30].max()
        
    #     k_field                         = self.meank
    #     k_field                         = np.log10(k_field)
    #     k_field[self.meanh== 1e+30]     = 1e+30
    #     vmink = k_field.min()
    #     vmaxk = k_field[k_field < 86400].max()
        
    #     plt.figure(figsize=(25, 25), dpi=250)  
    #     for ilay in range(self.members[0].model.modelgrid.nlay):
    #         ax = plt.subplot(self.members[0].model.modelgrid.nlay, 1, ilay + 1)
    #         pmv = flopy.plot.PlotMapView(self.members[0].model, layer=ilay, ax=ax)
    #         ax.set_aspect("equal")
    #         pmv.plot_array(k_field.flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
    #         pmv.plot_grid(colors="k", alpha=0.1)
    #         pmv.plot_bc('riv')
    #         ax.set_title("Layer {}".format(ilay + 1))
    #         pmv.plot_vector(qx, qy, width=0.0008, color="black")


    #     plt.figure(figsize=(25, 25), dpi=250)  
    #     for ilay in range(self.members[0].model.modelgrid.nlay):
    #         ax = plt.subplot(self.members[0].model.modelgrid.nlay, 1, ilay + 1)
    #         pmv = flopy.plot.PlotMapView(self.members[0].model, layer=ilay, ax=ax)
    #         ax.set_aspect("equal")
    #         pmv.plot_array(self.meanh, cmap="jet", vmin=vmin, vmax=vmax)
    #         pmv.plot_grid(colors="k", alpha=0.1)
    #         pmv.contour_array(
    #             self.meanh, masked_values = 1e+30, levels=np.arange(vmin, vmax, 0.1), linewidths=2.0, vmin=vmin, vmax=vmax
    #         )
    #         ax.set_title("Layer {}".format(ilay + 1))
    #         pmv.plot_vector(qx, qy, width=0.0008, color="black")
       
    
class Member:
        
    def __init__(self, direc):
        self.direc      = direc
        self.hdirec     = direc + "/flow_output/flow.hds"
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
                                verbosity_level     = 0
                                )
        self.model      = self.sim.get_model()
    
    def get_hfield(self):
        return self.model.output.head().get_data()
    
    def get_kfield(self):
        return self.model.npf.k.array
        
    def set_kfield(self, Kf):
        self.model.npf.k.set_data(np.exp(Kf))
            
    def set_hfield(self, Hf):
        self.model.ic.strt.set_data(Hf)
        
    def predict(self, EvT, Rch, Qpu):
        
        self.model.rch.recharge.set_data(Rch)
        self.model.evt.rate.set_data(EvT)
        
        wel_data = self.model.wel.stress_period_data.get_data()
        wel_data[0]['q'][0] = Qpu[0]
        wel_data[0]['q'][1] = Qpu[1]
        self.model.wel.stress_period_data.set_data(wel_data)
    
        success, buff = self.sim.run_simulation()
        
        if not success:
            print(f"Model in {self.direc} has failed")
            Hf = None
               
        else:
            Hf = self.model.output.head().get_data()
        return Hf
    
    def updateK(self, cov_mod, PPcor, PP_K, X, Y):
        
        krig = krige.Ordinary(cov_mod, cond_pos=PPcor, cond_val = PP_K)
        field = krig([X,Y])
        self.set_kfield(np.reshape(field[0],self.model.npf.k.array.shape))

    def get_spdis(self):
        head                        = self.model.output.head().get_data()
        qx                          = np.zeros(np.shape(head))
        qy                          = np.zeros(np.shape(head))
        bud                         = self.model.output.budget()
        spdis                       = bud.get_data(text="DATA-SPDIS")[0]
        counter                     = 0
        
        for i in range(self.model.modelgrid.nlay):
            for j in range(self.model.modelgrid.nrow):
                for k in range(self.model.modelgrid.ncol):
                    if head[i,j,k] < 1e+30:
                        qx[i,j,k]   = spdis["qx"][counter]
                        qy[i,j,k]   = spdis["qy"][counter]
                        counter     += 1
        return qx, qy
    
    
class VirtualReality(Member):
    # Inherit Properties of the Member class
    def __init__(self, direc,  obsloc):
        super().__init__(direc)
        self.obsloc = obsloc
        
    def set_kfield(self, cov_mod, PPcor, PP_K, X, Y):

        krig = krige.Ordinary(cov_mod, cond_pos=PPcor, cond_val = PP_K)
        field = krig([X,Y])
        k_new = np.exp(np.reshape(field[0],self.model.npf.k.array.shape))
        if self.model.dis.nlay.array == 1:
            if k_new.ndim == 2:
                k_new = k_new.reshape(self.model.npf.k.array.shape)
            k_new[0,54,71]    = 86400
            k_new[0,28,190]   = 86400
        else:
            k_new = np.tile(k_new, np.array([self.model.dis.nlay.array, 1, 1]))
            for i in range(self.model.dis.nlay.array):
                k_new[i,:,:]      = k_new[i,:,:] * np.random.normal(loc=1-i*0.2, scale=i*0.01, size=self.model.npf.k.array.shape)
                k_new[i,54,71]    = 86400
                k_new[i,28,190]   = 86400

        self.model.npf.k.set_data(k_new)
        
    def predict(self, EvT, Rch, Qpu):
        
        self.model.rch.recharge.set_data(Rch)
        self.model.evt.rate.set_data(EvT)
        
        wel_data = self.model.wel.stress_period_data.get_data()
        wel_data[0]['q'][0] = Qpu[0]
        wel_data[0]['q'][1] = Qpu[1]
        self.model.wel.stress_period_data.set_data(wel_data)
    
        success, buff = self.sim.run_simulation()
        
        if not success:
            print("Virtual Truth has failed")
            Hf = None
               
        else:
            Hf = self.model.output.head().get_data()
            
        self.set_hfield(Hf)
        
    def pert_obs(self, nreal, nobs, eps, pert = True):
        # Entering observation data for all ensemble members (allows individual pert)
        if self.model.dis.nlay.array == 1:
            obs = [self.model.output.head().get_data()[:,self.obsloc[i][0], self.obsloc[i][1]] for i in range(len(self.obsloc))]
        else:
            obs = [np.mean(self.model.output.head().get_data()[:,self.obsloc[i][0], self.obsloc[i][1]]) for i in range(len(self.obsloc))]
        
        if pert == True:
            Y_obs_pert = np.tile(obs, (1,nreal)) + np.random.normal(loc=0, scale=eps, size=(nobs, nreal))
        
            return Y_obs_pert
        else:
            return obs
        
    def get_spdis(self):
        head                        = self.model.output.head().get_data()
        qx                          = np.zeros(np.shape(head))
        qy                          = np.zeros(np.shape(head))
        bud                         = self.model.output.budget()
        spdis                       = bud.get_data(text="DATA-SPDIS")[0]
        counter                     = 0
        
        for i in range(self.model.modelgrid.nlay):
            for j in range(self.model.modelgrid.nrow):
                for k in range(self.model.modelgrid.ncol):
                    if head[i,j,k] < 1e+30:
                        qx[i,j,k]   = spdis["qx"][counter]
                        qy[i,j,k]   = spdis["qy"][counter]
                        counter     += 1
        return qx, qy, head
    
    def plot(self):
        
        head                    = self.model.output.head().get_data()
        k_field                 = self.model.npf.k.array
        k_field                 = np.log10(k_field/86400)
        k_field[head == 1e+30]  = 1e+30
        
        qx, qy, head = self.get_spdis()
        
        vmin = head[head > -1e+30].min()
        vmax = head[head < 1e+30].max()
        vmink = k_field.min()
        vmaxk = k_field[k_field < 86400].max()
        
        plt.figure(figsize=(25, 25), dpi=250)
        for ilay in range(self.model.modelgrid.nlay):
            ax = plt.subplot(self.model.modelgrid.nlay, 1, ilay + 1)
            pmv = flopy.plot.PlotMapView(self.model, layer=ilay, ax=ax)
            mapable = pmv.plot_array(k_field[ilay,:,:].flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
            im_ratio = k_field.shape[1]/k_field.shape[2]
            cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax)
            cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
            cbar.ax.tick_params(labelsize=20)
            ax.yaxis.label.set_size(25)
            ax.xaxis.label.set_size(25)
            plt.xlabel('Easting [m]', fontsize = 25)
            plt.ylabel('Northing [m]', fontsize = 25)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_aspect("equal")
            pmv.plot_grid(colors="k", alpha=0.1)
            pmv.plot_bc('riv')
            # ax.set_title("Layer {}".format(ilay + 1))
            pmv.plot_vector(qx, qy, width=0.0008, color="black")


        plt.figure(figsize=(25, 25), dpi=250)  
        for ilay in range(self.model.modelgrid.nlay):
            ax = plt.subplot(self.model.modelgrid.nlay, 1, ilay + 1)
            pmv = flopy.plot.PlotMapView(self.model, layer=ilay, ax=ax)
            ax.set_aspect("equal")
            mapable = pmv.plot_array(head[ilay,:,:].flatten(), cmap="jet", vmin=vmin, vmax=vmax)
            im_ratio = head.shape[1]/head.shape[2]
            cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax)
            cbar.ax.set_ylabel('Hydraulic Head [m]', fontsize = 25)
            cbar.ax.tick_params(labelsize=20)
            ax.yaxis.label.set_size(25)
            ax.xaxis.label.set_size(25)
            plt.xlabel('Easting [m]', fontsize = 25)
            plt.ylabel('Northing [m]', fontsize = 25)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_aspect("equal")
            pmv.plot_grid(colors="k", alpha=0.1)
            pmv.contour_array(
                head, masked_values = 1e+30, levels=np.arange(vmin, vmax, 0.1), linewidths=2.0, vmin=vmin, vmax=vmax
            )
            # ax.set_title("Layer {}".format(ilay + 1))
            pmv.plot_vector(qx, qy, width=0.0008, color="black")