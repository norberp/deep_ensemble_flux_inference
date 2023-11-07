import numpy as np
import pdb

def tsubspaceEnKA(X,Y,Yp,svdt=0.9):
    """
    tsubspaceEnKA: Implmentation of the Ensemble Kalman Analysis in the ensemble subspace.
    This scheme is more robust in the regime where you have a larger number of observations 
    and/or states and/or parameters than ensemble members.
    Inputs:
        X: Prior ensemble matrix (n x Ne array)
        Y: Perturbed observation ensemble matrix (m x Ne array)
        Yp: Predicted observation ensemble matrix (m x Ne array)
        svdt: Level of truncation of singular values for pseudoinversion, recommended=0.9 (90%)
    Outputs:
        post: Posterior ensemble matrix (n x Ne array)
    Dimensions:
        Ne is the number of ensemble members, n is the number of state variables and/or
        parameters, and m is the number of observations.
    
    The implementation follows that described in Algorithm 6 in the book of Evensen et al. (2022),
    while adopting the truncated SVD procedure described in Emerick (2016) which also adopts the
    ensemlbe supspace method to the ES-MDA.
    
    Note that the observation error covariance R is defined implicitly here through the perturbed observations.
    This matrix (Y) should be perturbed in such a way that it is consistent with R in the case of single data
    assimilation (no iterations) or alpha*R in the case of multiple data assimilation (iterations). Moreover,
    although we should strictly be perturbing the predicted observations this does not make any difference in 
    practice (see van Leeuwen, 2020) and simplifies the implmentation of the ensemble subspace approach.
    
    References:
        Evensen et al. 2022: https://doi.org/10.1007/978-3-030-96709-3
        Emerick 2016: https://doi.org/10.1016/j.petrol.2016.01.029
        van Leeuwen 2020: https://doi.org/10.1002/qj.3819
    
    Code by K. Aalstad
    """
    Ne=np.shape(X)[1] # Number of ensemble members
    No=np.shape(Y)[0]
    INe=np.eye(Ne)
    Pi=(INe-np.ones([Ne,Ne])/Ne)/np.sqrt(Ne-1) # Anomaly operator (subtracts ensemble mean)
    Ya=Y@Pi # Observation anomalies 
    Ypa=Yp@Pi # Predicted observation anomalies
    S=Ypa
    [U, E, _] = np.linalg.svd(S,full_matrices=False) 
    Evr = np.cumsum(E)/np.sum(E)
    N=min(Ne,No)
    these=np.arange(N)
    try:
        Nr=min(these[Evr>svdt]) # Truncate small singular values
    except:
        pdb.set_trace()
    these=np.arange(Nr+0) #this should be Nr+1
    E=E[these]
    U=U[:,these]
    Ei=np.diag(1/E)
    P=Ei@(U.T)@Ya@(Ya.T)@U@(Ei.T)
    [Q, L, _] = np.linalg.svd(P)
    LpI=L+1
    LpIi=np.diag(1/LpI)
    UEQ=U@(Ei.T)@Q
    Cpinv=UEQ@LpIi@UEQ.T # Pseudo-inversion of C=(C_YY+alpha*R) in the ensemble subspace
    Inn=Y-Yp # Innovation
    W=(S.T)@Cpinv@Inn
    T=(INe+W/np.sqrt(Ne-1))
    Xu=X@T # Update
    return Xu
    

def forwardMLP(X,npl,w,b):
    """
    Inputs:
        X = Feature (predictor) matrix (Ns x Nf) 
        npl = Vector containing number of neurons per layer ((Nl+1) x 1) 
        w = Weight vector (concatenation of weight matrices for each layer) 
        b = Bias vector (concatenation of bias vector for each layer)
    Short hands: Ns = Number of sampled feature vectors, Nf = Number of
    features (inputs), Nl = Number of hidden layers.
    
    Note that in this function we follow machine learning notation using X for inputs. This should 
    not be confused with X in data assimilation where X denotes what we actually update, 
    which in the case of neural nets are the weights and biases.
    
    Code by K. Aalstad and N. Pirk
    """
    
    Nf=np.shape(X)[1]
    X=X.T # Transpose so features as rows
    
    thesew=0
    theseb=0
    Zj=0
    
    # Forward pass through layers, from input to output
    for j in range(len(npl)):
        Nj=npl[j]
        if j==0:
            Nprev=Nf
            Zold=X
            wind=0
            bind=0
        else:
            Nprev=npl[j-1]
            Zold=Zj
            try:
                wind=np.max(thesew)+1
                bind=np.max(theseb)+1
            except:
                pdb.set_trace()
        thesew=np.arange(Nprev*Nj)
        thesew=thesew+wind
        Wj=w[thesew]
        if Nprev>1:
            Wj=np.reshape(Wj,(Nj,Nprev))
        theseb=np.arange(Nj)
        theseb=theseb+bind
        bj=b[theseb]
        prodj=Wj@Zold
        sumj=prodj.T+bj
        Aj=sumj.T
        if (j+1)<len(npl):
            Zj=np.maximum(Aj,0)
            #Zj=np.tanh(Aj) #other options for activation function
            #Zj=Aj/(1+np.exp(-Aj)) #Sigmoid linear unit (SiLU)
        else:
            Zj=Aj # Linear output layer
        
        
    ypred=Zj # Output (assuming linear activation on output)
    
    return ypred
        
        