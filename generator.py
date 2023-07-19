#### 2-D random field generator
import numpy as np

def generator(X, Y, lx, ang, sigma2, mu):
    
    ntot    = np.shape(X)[0] * np.shape(X)[1]
    ang2    = ang/180*np.pi
    
    # ============== BEGIN COVARIANCE BLOCK =========================================
    # Transformation to angular coordinates
    X2 = np.cos(ang2)*X + np.sin(ang2)*Y;
    Y2 =-np.sin(ang2)*X + np.cos(ang2)*Y;

    # Accounting for corellation lengths
    H = np.sqrt((X2/lx[0])**2+(Y2/lx[1])**2);

    # Matérn 3/2
    RYY = sigma2 * np.multiply((1+np.sqrt(3)*H),np.exp(-np.sqrt(3)*H))

    # ============== END COVARIANCE BLOCK =====================================
    
    # ============== BEGIN POWER-SPECTRUM BLOCK ===============================
    # Fourier Transform (Origin Shifted to Node (0,0))
    # Yields Power Spectrum of the field
    SYY=np.fft.fftn(np.fft.fftshift(RYY))/ntot;
    # Remove Imaginary Artifacts
    SYY=np.abs(SYY)
    SYY[0,0] =0;
    # ============== END POWER-SPECTRUM BLOCK =================================
       
    # ============== BEGIN FIELD GENERATION BLOCK =============================
    # Generate the field
    # nxhelp is nx with the first two entries switched
    nxhelp = np.array([np.shape(X)[0], np.shape(X)[1]])

    # if X.ndim > 1:
    #     nxhelp[0:2] = [np.shape(X)[0], np.shape(X)[1]]
    # else:
    #     nxhelp = np.array([1,nx[0]]).T;

    # Generate the real field according to Dietrich and Newsam (1993)
    ran = np.multiply(np.sqrt(SYY), np.squeeze(
            np.array([np.random.randn(nxhelp[0], nxhelp[1]) + 
                      1j*np.random.randn(nxhelp[0], nxhelp[1])] 
                     ,dtype = 'complex_'))
            )
    # Backtransformation into the physical coordinates
    ran = np.real(np.fft.ifftn(ran*ntot))+mu;
    # ============== END FIELD GENERATION BLOCK ===============================
    
    return ran