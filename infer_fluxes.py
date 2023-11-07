import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import EnNN as en
import json
from urllib import request

print('Downloading dataset from Zenodo')
remote_url = 'https://zenodo.org/records/7913027/files/iskoras_measurements.csv'
local_file = 'iskoras_measurements.csv'
request.urlretrieve(remote_url, local_file)


#read data into pandas dataframe
df=pd.read_csv('iskoras_measurements.csv', index_col=0, parse_dates=True)
df=df.rename(columns={
    'T$_\mathrm{soil}$':'soil_temperature',
    'VWC':'soil_volumetric_water_content',
    'T$_\mathrm{air}$':'air_temperature',
    'SW$_\mathrm{in}$':'shortwave_incoming',
    'LW$_\mathrm{in}$':'longwave_incoming',
    'VPD':'vpd',
    'Albedo':'albedo',
    'T$_\mathrm{surf}$':'surface_temperature',
    'NDVI':'NDVI',
    'FSCA':'FSCA',
    'F$_\mathrm{total}^\mathrm{CO2}$':'co2_flux_filtered',
    'F$_\mathrm{total}^\mathrm{CH4}$':'ch4_flux_filtered',
    'w$_\mathrm{palsa}$':'palsa',
    'w$_\mathrm{ponds}$':'ponds',
    'w$_\mathrm{fen}$':'fen'
    })


#convert to mmol
df['ch4_flux_filtered']=df['ch4_flux_filtered']/1000


###select features
predictors=['soil_temperature','soil_volumetric_water_content','air_temperature', 'shortwave_incoming','longwave_incoming', 'vpd','albedo', 'surface_temperature','NDVI','FSCA']


# Hyperparameters
np.random.seed(123)
Ne=10 #number of ensemble members
Na=128 #number of Kalman iterations for ES-MDA
n_runs=2 #number of local ensemble runs for the deep ensemble approximation
frac=1.0 #fraction of data to use, should some data be left out for independent validation
mbf=0.1 #fraction of training data to use as a minibatch

n_outputs=3 #classes to predict
npl=np.array([96,48,12,48,96,  n_outputs]) #nodes per layer for the autoencoder network architecture

#dictionary to store the settings for this experiment:
exp_name='iskoras_disaggregation'
exp_dict={}
exp_dict['exp_name']=exp_name
exp_dict['level1_file']='iskoras_measurements.csv'
exp_dict['level1_index']=df.index.strftime(date_format='%Y-%m-%d %H:%M').tolist()
exp_dict['predictors']=predictors
exp_dict['flux_versions']=[] #added in the loop below

exp_dict['Ne']=Ne
exp_dict['Na']=Na
exp_dict['frac']=frac
exp_dict['n_runs']=n_runs

exp_dict['n_outputs']=n_outputs
exp_dict['class_names']=['palsa','ponds','fen']


#start training the networks for CO2 and CH4 seperately:
for flux_version in ['co2_flux_filtered','ch4_flux_filtered']:
    print(flux_version)
    exp_dict['flux_versions'].append(flux_version)
    
    if flux_version=='ch4_flux_filtered':
        sigy=0.0025 #uncertainty in flux units
    elif flux_version=='co2_flux_filtered':
        sigy=0.1 #uncertainty in flux units
    
    x_all = df[predictors].values
    
    df2=df[ predictors + [flux_version] + ['palsa','ponds'] ].dropna()
    
    plt.figure()
    df2.plot(subplots=True, style=',', figsize=(10,10))
    plt.savefig('data_in_'+flux_version+'.png', bbox_inches='tight',dpi=300)
    
    x = df2[predictors].values
    c = df2[flux_version].values
    print(" - - - len(c)= "+str(len(c)))
    
    N=len(c)
    Nt=int(np.round(frac*N))
    print(" - - - Nt= "+str(Nt))
    
    training=np.random.choice(N,Nt,replace=False)
    training=np.sort(training)
    
    testing=np.array( list( set(np.arange(N)).difference(training) ) )
    
    exp_dict[flux_version+'_trainingMask']=training.tolist()
    exp_dict[flux_version+'_testingMask']=testing.tolist()
    
    #start loop over runs here
    for rid in range(n_runs):
        print("  # # #  New run: "+str(rid))
        
        # Feature (input, predictor) matrix
        X=x
        Xs=X[training].std(0) #scaling based on training subset
        Xm=X[training].mean(0)
        X_standardised=(X-Xm)/Xs # Scale features
    
        x_all_standardised=(x_all-Xm)/Xs
    
        # Output vector (coud be a matrix too)
        y=c
        ys=y.std(0)
        ym=y.mean(0)
        y_standardised=(y-ym)/ys # Scale outputs
        sigy_standardised=sigy/ys
    
        # Select training data
        ytrain=y_standardised[training]
        Xtrain=X_standardised[training,:]
        print(" - - - len(ytrain)= "+str(len(ytrain)))
    
        # Generate priors
        Nb=np.sum(npl) # Number of bias terms
        Nf=np.shape(X_standardised)[1]
        Nw=0
        Njold=Nf
        for j in range(len(npl)):
            Nj=npl[j]
            Nw=Nw+Njold*Nj
            Njold=Nj
        b=1*np.random.randn(Nb,Ne)
        w=1*np.random.randn(Nw,Ne)   
    
        theta_prior=np.concatenate((b,w)) 
        print(" - - - len(theta_prior)= "+str(len(theta_prior)))
    
        # Size of minibatches
        Nm=int(np.round(mbf*Nt))
        print(" - - - Nm= "+str(Nm))
    
        print(" * * * Training...")
        for k in range(Na):
            print(" - - - k= "+str(k))
        
            # Select the minibatch for this iteration
            if Nm<Nt:
                mbatch=np.random.choice(Nt,Nm,replace=False)
                mbatch=np.sort(mbatch)
            else:
                mbatch=np.arange(Nt)
            
            Ypred=np.zeros((Nm,Ne))
        
            for j in range(Ne):
                ypj=en.forwardMLP(Xtrain[mbatch,:],npl,w[:,j],b[:,j])
                ypj=ypj.T
                Ypred[:,j]=ypj[:,0]*df2['palsa'][training[mbatch]] + ypj[:,1]*df2['ponds'][training[mbatch]] + ypj[:,2]*(1-df2['palsa'][training[mbatch]]-df2['ponds'][training[mbatch]])
                
            if (k+1)<Na:
                wold=w
                theta=np.concatenate((b,w))
                perts=np.sqrt(Na)*sigy_standardised*np.random.randn(Nm,Ne)
                Y=ytrain[mbatch]+perts.T
                Y=Y.T
                theta=en.tsubspaceEnKA(theta,Y,Ypred)
                b=theta[:Nb,:]
                w=theta[Nb:,:]
                
                
        fig1=plt.figure(figsize=(9,4))
        ax11=fig1.add_subplot(121)
        ax12=fig1.add_subplot(122)
        for tt in range(5):
            gg=sns.distplot(a=theta_prior[tt,:], hist=0,  ax=ax11, label=r'$\theta_{'+str(tt)+'}$')
            gg=sns.distplot(a=theta[tt,:], hist=0,  ax=ax12)
            
        ax11.set_xlabel(r'$\theta$ [-]')
        ax12.set_xlabel(r'$\theta$ [-]')
        ax11.set_ylabel('Kernel estimated density')
        ax11.set_title('Prior parameter distributions')
        ax12.set_title('Posterior parameter distributions')
        fig1.savefig('theta_subset_'+flux_version+'_run'+str(rid)+'.png', bbox_inches='tight', dpi=300)
        
        
        print(" * * * Post pred...")
        Ypost=np.zeros((len(x_all_standardised),n_outputs,Ne)) 
        for j in range(Ne):
            ypj=en.forwardMLP(x_all_standardised,npl,w[:,j],b[:,j])
            ypj=ypj.T
            Ypost[:,:,j]=ypj
            
        Ypost=Ypost*ys+ym
        
        #append this run's predictions as new ensemble members
        if rid==0:
            Ypost_all=np.copy(Ypost)
        else:
            Ypost_all=np.concatenate((Ypost_all,Ypost), axis=2) 
            
    ## end loop over runs here
    
    Ypost_mean=Ypost_all.mean(2)
    Ypost_std=Ypost_all.std(2)
    Ypost_p5 =np.nanpercentile(Ypost_all,5, axis=2)
    Ypost_p25=np.nanpercentile(Ypost_all,25, axis=2)
    Ypost_p50=np.nanpercentile(Ypost_all,50, axis=2)
    Ypost_p75=np.nanpercentile(Ypost_all,75, axis=2)
    Ypost_p95=np.nanpercentile(Ypost_all,95, axis=2)
    
    exp_dict[flux_version+'_Ypost_mean']=Ypost_mean.tolist()
    exp_dict[flux_version+'_Ypost_std'] =Ypost_std.tolist()
    exp_dict[flux_version+'_Ypost_p5']  =Ypost_p5.tolist()
    exp_dict[flux_version+'_Ypost_p25'] =Ypost_p25.tolist()
    exp_dict[flux_version+'_Ypost_p50'] =Ypost_p50.tolist()
    exp_dict[flux_version+'_Ypost_p75'] =Ypost_p75.tolist()
    exp_dict[flux_version+'_Ypost_p95'] =Ypost_p95.tolist()
    
    
    #calculate cumsum uncertainty range
    Ypost_all_cumsum=np.nancumsum(Ypost_all, axis=0)
    
    Ypost_cumsum_mean=Ypost_all_cumsum.mean(2)
    Ypost_cumsum_p5 =np.nanpercentile(Ypost_all_cumsum,5, axis=2)
    Ypost_cumsum_p25=np.nanpercentile(Ypost_all_cumsum,25, axis=2)
    Ypost_cumsum_p50=np.nanpercentile(Ypost_all_cumsum,50, axis=2)
    Ypost_cumsum_p75=np.nanpercentile(Ypost_all_cumsum,75, axis=2)
    Ypost_cumsum_p95=np.nanpercentile(Ypost_all_cumsum,95, axis=2)
    
    exp_dict[flux_version+'_Ypost_cumsum_mean']=Ypost_cumsum_mean.tolist()
    exp_dict[flux_version+'_Ypost_cumsum_p5']  =Ypost_cumsum_p5.tolist()
    exp_dict[flux_version+'_Ypost_cumsum_p25']  =Ypost_cumsum_p25.tolist()
    exp_dict[flux_version+'_Ypost_cumsum_p50'] =Ypost_cumsum_p50.tolist()
    exp_dict[flux_version+'_Ypost_cumsum_p75']  =Ypost_cumsum_p75.tolist()
    exp_dict[flux_version+'_Ypost_cumsum_p95'] =Ypost_cumsum_p95.tolist()
    
    
    #evaluation with test data
    if frac<1.0:
        mask = df[ predictors + [flux_version] + ['palsa','ponds'] ].isna().any(axis=1)
        Ypost_mean_validFlux =Ypost_mean[~mask]
        
        exp_dict[flux_version+'_testingMask_alldata']=(~mask).tolist()
        
        Ypost_mean_total = Ypost_mean_validFlux[:,0]*df2['palsa'] + Ypost_mean_validFlux[:,1]*df2['ponds'] + Ypost_mean_validFlux[:,2]*(1-df2['palsa']-df2['ponds'])
        
        r2_score_testdata = metrics.r2_score(Ypost_mean_total[testing], y[testing])
        rmse_testdata = metrics.mean_squared_error(Ypost_mean_total[testing], y[testing])
        nrmse_testdata=rmse_testdata/(np.max(y[testing])-np.min(y[testing]))
        
        r2_score_training = metrics.r2_score(Ypost_mean_total[training], y[training])
        rmse_training = metrics.mean_squared_error(Ypost_mean_total[training], y[training])
        nrmse_training=rmse_training/(np.max(y[training])-np.min(y[training]))
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(Ypost_mean_total[training], y[training], 'o', mfc=None, ms=2.0, label=r'Training data (n=%d, NRMSE=%2.1f%%, R$^2$=%0.2f)'%(len(y[training]), nrmse_training*100., r2_score_training) )
        ax.plot(Ypost_mean_total[testing], y[testing], 'o', mfc=None, ms=2.0, label=r'Test data (n=%d, NRMSE=%2.1f%%, R$^2$=%0.2f)'%(len(y[testing]), nrmse_testdata*100., r2_score_testdata) )
        
        lims=[np.array([ax.get_ylim(),ax.get_xlim()]).min(axis=0)[0],np.array([ax.get_ylim(),ax.get_xlim()]).max(axis=0)[1]]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims,lims,'--k')
        
        if 'co2' in flux_version:
            ax.set_xlabel(r'Mean predicted total CO$_2$ flux [$\mu$mol m$^{-2}$ s$^{-1}$]')
            ax.set_ylabel(r'Observed total CO$_2$ flux [$\mu$mol m$^{-2}$ s$^{-1}$]')
        elif 'ch4' in flux_version:
            ax.set_xlabel(r'Mean predicted total CH$_4$ flux [$\mu$mol m$^{-2}$ s$^{-1}$]')
            ax.set_ylabel(r'Observed total CH$_4$ flux [$\mu$mol m$^{-2}$ s$^{-1}$]')
                
        ax.legend(loc='best')
        fig.savefig('evaluation_'+flux_version+'_'+exp_name+'.png', bbox_inches='tight', dpi=300)
        
    
    
print("write results to file....")
json.dump(exp_dict, open(exp_dict['exp_name']+'.txt','w'))



