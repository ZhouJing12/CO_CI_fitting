import os
import matplotlib.pyplot as plt
import numpy as np
from supplyment import *

Msun=1.988e33
Mh=1.67e-24
pc_cm=3.08e18

def compute_Xco(nH2,Tb10,Kvir):
    return 2.65*nH2**0.5/Tb10/Kvir
def compute_rcico(Tkin,nH2,Kvir,Tb10):
    Xco=compute_Xco(nH2,Tb10,Kvir)
    Q10,Q21=compute_population(Tkin,nH2,ci_mol)[1:]
    H_fun=8*np.pi*k*ci_mol.rad_data[:2,3]**2/h/c**3/ci_mol.rad_data[:2,2]*pc_cm**2*(Mh*12)/Msun*1e5
    rcico=Xco*np.array([Q10,Q21])*ci_abundance*6/H_fun
    return rcico

############# revise observation data #############
sourcename='MACS1931-26_CGM'
redshift=0.35

co_abundance=1e-4
ci_abundance=3e-5

co_mol=mol_data('./','co.dat')
ci_mol=mol_data('./','catom.dat')

data=np.loadtxt(sourcename+'_obs_lum.csv',dtype=str,delimiter=',')
idx=np.where(data[0]=='CO_Jup')[0][0]
nempidx=data[1:,idx]!=''
co_Jup=data[1:,idx][nempidx].astype(int)
idx=np.where(data[0]=='CO_lum')[0][0]
co_lum=data[1:,idx][nempidx].astype(float)
idx_e=np.where(data[0,idx:]=='error')[0][0]
co_err=data[1:,idx+idx_e][nempidx].astype(float)

idx=np.where(data[0]=='CI_Jup')[0][0]
nempidx=data[1:,idx]!=''
ci_Jup=data[1:,idx][nempidx].astype(int)
idx=np.where(data[0]=='CI_lum')[0][0]
ci_lum=data[1:,idx][nempidx].astype(float)
idx_e=np.where(data[0,idx:]=='error')[0][0]
ci_err=data[1:,idx+idx_e][nempidx].astype(float)

co_ratio=co_lum/co_lum[0]
co_ratio_err=co_err/co_lum[0]
ci_ratio=ci_lum/co_lum[0]
ci_ratio_err=ci_err/co_lum[0]

Tback=2.73*(1+redshift)

############# revise observation data #############


################### draw likelyhood maps ################
Tkins=np.logspace(1,3,21)
nH2s=np.logspace(2,8,61)
Kvir_bond=(10,20)
Kvirs=np.logspace(np.log10(Kvir_bond[0]),np.log10(Kvir_bond[1]),21)

def log_likelyhood(params):
    Tkin,nH2,Kvir=params
    Tb,f_occupation=run_myradex_of(Tkin=Tkin,nH2=nH2,abundance_Kvir=co_abundance/Kvir,molecule=co_mol,Tbg=Tback)
    ratio_model=Tb[co_Jup-1]/Tb[co_Jup[0]-1]
    ll_co=-0.5*np.sum(((co_ratio-ratio_model)/co_ratio_err)**2)
    return ll_co

#def log_likelyhood(params):
#    Tkin,nH2,Kvir=params
#    Tb,f_occupation=run_myradex_of(Tkin=Tkin,nH2=nH2,abundance_Kvir=co_abundance/Kvir,molecule=co_mol,Tbg=Tback)
#    ratio_model=Tb[co_Jup-1]/Tb[co_Jup[0]-1]
#    ll_co=-0.5*np.sum(((co_ratio-ratio_model)/co_ratio_err)**2)
#    rcico=np.array(compute_rcico(Tkin,nH2,Kvir,Tb[0]))
#    #rci21=ciTr[1]/ciTr[0]
#    #Qul=compute_population(Tkins[i],nH2s[j],ci_mol)
#    #rci21=Qul[2]/Qul[1]*ci_mol.rad_data[1,2]/ci_mol.rad_data[0,2]*(ci_mol.rad_data[0,3]/ci_mol.rad_data[1,3])**2
#    ll_ci=-0.5*np.sum(((rcico[ci_Jup-1]-ci_ratio)/ci_ratio_err)**2)
#    return ll_co+ll_ci

nT,nn,nK=len(Tkins),len(nH2s),len(Kvirs)
lls=np.zeros([nT,nn,nK])

for i in range(nT):
    for j in range(nn):
        for l in range(nK):
            lls[i,j,l]=log_likelyhood([Tkins[i],nH2s[j],Kvirs[l]])

def show_map(data,contour):
    _,ax=plt.subplots()
    plt.imshow(data,origin='lower',cmap='jet',aspect='auto')
    xt=np.arange(0,21,5)
    yt=np.arange(0,61,10)
    plt.xlabel(r'log $T_{\rm kin}$ (K)')
    plt.xticks(xt,np.log10(Tkins[xt]))
    plt.ylabel(r'log $n_{\rm H_2}$ ($\rm cm^{-3}$)')
    plt.yticks(yt,np.log10(nH2s[yt]))
    cbar=plt.colorbar()
    #cbar.set_label('log dv/dr (km/s/pc)')
    cbar.set_label(r'<$T^{\rm CI}_{\rm 2-1}>/<T^{\rm CO}_{\rm 1-0}$>')
    plt.clim(ci_ratio[0]-ci_ratio_err[0]*3,ci_ratio[0]+ci_ratio_err[0]*3)                                                    # revise plot colorscale
    #plt.contour(contour,colors='w',origin='lower',linestyles='solid')
    contournew=np.log10(abs(contour))
    levels=np.unique(np.percentile(contournew,np.arange(0,20,2)))
    cs=plt.contour(contournew,levels,colors='w',origin='lower',linestyles='solid')
    plt.text(0.7,0.85,r'$%g<K_{\rm vir}<%g$'%Kvir_bond+'\n'+r'$T^{\rm CI}_{\rm 2-1}/T^{\rm CO}_{\rm 1-0}$=%.2f'%(ci_ratio[0]),transform=ax.transAxes)
    levels=np.array([0.2,0.3,0.4,0.6,1.0])
    cs=plt.contour(data,levels,colors='purple',origin='lower',linestyles='solid')
    plt.clabel(cs,inline=True,colors='purple')
    #plt.savefig('ratio_1.35_Kvir_0.5_2.pdf')

max_ll=np.max(lls,axis=2)
max_ll_i=np.argmax(lls,axis=2)
rci2co1=np.zeros([nT,nn])
for i in range(nT):
    for j in range(nn):
        #ciTr=compute_Tr(Tkins[i],nH2s[j],ci_abundance/Kvirs[max_ll_i[i,j]],Tback,ci_mol)
        #rci21[i,j]=ciTr[1]/ciTr[0]
        #Qul=compute_population(Tkins[i],nH2s[j],ci_mol)
        #rci21[i,j]=Qul[2]/Qul[1]*ci_mol.rad_data[1,2]/ci_mol.rad_data[0,2]*(ci_mol.rad_data[0,3]/ci_mol.rad_data[1,3])**2                                             
        Tb,f_occupation=run_myradex_of(Tkin=Tkins[i],nH2=nH2s[j],abundance_Kvir=co_abundance/Kvirs[max_ll_i[i,j]],molecule=co_mol,Tbg=Tback)
        rci2co1[i,j]=compute_rcico(Tkins[i],nH2s[j],Kvirs[max_ll_i[i,j]],Tb[0])[1] 
os.system('mkdir fit_result')
show_map(rci2co1.T,max_ll.T)
plt.savefig('fit_result/likelyhoodmap_Kvir_%g_%g.pdf'%Kvir_bond)

def log_likelyhood(params):
    Tkin,nH2,Kvir=params
    Tb,f_occupation=run_myradex_of(Tkin=Tkin,nH2=nH2,abundance_Kvir=co_abundance/Kvir,molecule=co_mol,Tbg=Tback)
    ratio_model=Tb[co_Jup-1]/Tb[co_Jup[0]-1]
    ll_co=-0.5*np.sum(((co_ratio-ratio_model)/co_ratio_err)**2)
    rcico=np.array(compute_rcico(Tkin,nH2,Kvir,Tb[0]))
    #rci21=ciTr[1]/ciTr[0]
    #Qul=compute_population(Tkins[i],nH2s[j],ci_mol)
    #rci21=Qul[2]/Qul[1]*ci_mol.rad_data[1,2]/ci_mol.rad_data[0,2]*(ci_mol.rad_data[0,3]/ci_mol.rad_data[1,3])**2
    ll_ci=-0.5*np.sum(((rcico[ci_Jup-1]-ci_ratio)/ci_ratio_err)**2)
    return ll_co+ll_ci
for i in range(nT):
    for j in range(nn):
        for l in range(nK):
            lls[i,j,l]=log_likelyhood([Tkins[i],nH2s[j],Kvirs[l]])

##################### plot best fitted SLED #########################

max_i=np.unravel_index(np.argmax(lls),lls.shape)
Tkins[max_i[0]],nH2s[max_i[1]],Kvirs[max_i[2]]
co_Tr,_=run_myradex_of(Tkin=Tkins[max_i[0]],nH2=nH2s[max_i[1]],abundance_Kvir=co_abundance/Kvirs[max_i[2]],molecule=co_mol,Tbg=Tback)
co_model=co_Tr/co_Tr[0]
ci_model=np.array(compute_rcico(Tkins[max_i[0]],nH2s[max_i[1]],Kvirs[max_i[2]],co_Tr[0]))
#Qul=compute_population(Tkins[max_i[0]],nH2s[max_i[1]],ci_mol)
#ci_model=Qul[2]/Qul[1]*ci_mol.rad_data[1,2]/ci_mol.rad_data[0,2]*(ci_mol.rad_data[0,3]/ci_mol.rad_data[1,3])**2

plt.figure(figsize=[8,6])
plt.title(sourcename+' CO SLED')
plt.errorbar(co_Jup,co_ratio*(co_Jup)**2,yerr=co_ratio_err*(co_Jup)**2,fmt='o',elinewidth=1,ms=1,mfc="w",mec='k',capthick=1,capsize=2)
plt.plot(np.arange(max(co_Jup)+2)+1,co_model[np.arange(max(co_Jup)+2)]*(np.arange(max(co_Jup)+2)+1)**2,'k',\
         label=r'$T_{\rm kin}=%3.1f {\rm K},{\rm log}\,n_{\rm H_2}=%3.1f, K_{\rm vir}=%3.1f$,<$T^{\rm CI}_{\rm 2-1}>/<T^{\rm CI}_{\rm 1-0}$>=%.3f'\
         %(Tkins[max_i[0]],np.log10(nH2s[max_i[1]]),Kvirs[max_i[2]],ci_model[1]/ci_model[0]))
plt.legend()
plt.xlabel('Jup')
plt.ylabel('normalized Flux')
plt.savefig('fit_result/CO_SLED_Kvir_%g_%g.pdf'%Kvir_bond)

ci_freq=np.array([492.160651,809.34197])
plt.figure(figsize=[8,6])
plt.title(sourcename+' CI SLED')
plt.errorbar(ci_Jup,ci_ratio*(ci_freq[ci_Jup-1]/115.2712018)**2,yerr=ci_ratio_err*(ci_freq[ci_Jup-1]/115.2712018)**2,fmt='o',elinewidth=1,ms=1,mfc="w",mec='k',capthick=1,capsize=2)
plt.plot(np.array([1,2]),ci_model*(ci_freq/115.2712018)**2,'k',\
         label=r'$T_{\rm kin}=%3.1f {\rm K},{\rm log}\,n_{\rm H_2}=%3.1f, K_{\rm vir}=%3.1f$,<$T^{\rm CI}_{\rm 2-1}>/<T^{\rm CI}_{\rm 1-0}$>=%.3f'\
         %(Tkins[max_i[0]],np.log10(nH2s[max_i[1]]),Kvirs[max_i[2]],ci_model[1]/ci_model[0]))
plt.legend()
plt.xlabel('Jup')
plt.ylabel(r'normalized Flux $(S_{\rm CI}/S_{\rm CO1-0})$')
plt.savefig('fit_result/CI_SLED_Kvir_%g_%g.pdf'%Kvir_bond)


import emcee,corner
from multiprocessing import Pool
paramguess=np.log10([Tkins[max_i[0]],nH2s[max_i[1]],Kvirs[max_i[2]]])
nwalkers,ndim=10,3
p0=paramguess+0.1* np.random.randn(nwalkers,ndim)
sampfile='fit_result/sample.h5'
overwrite=True

def log_likelyhood(params):
    if log_prior(params)==False:
        return -np.inf
    Tkin,nH2,Kvir=10**params
    Tb,f_occupation=run_myradex_of(Tkin=Tkin,nH2=nH2,abundance_Kvir=co_abundance/Kvir,molecule=co_mol,Tbg=Tback)
    ratio_model=Tb[co_Jup-1]/Tb[co_Jup[0]-1]
    ll_co=-0.5*np.sum(((co_ratio-ratio_model)/co_ratio_err)**2)
    rcico=np.array(compute_rcico(Tkin,nH2,Kvir,Tb[0]))
    ll_ci=-0.5*np.sum(((rcico[ci_Jup-1]-ci_ratio)/ci_ratio_err)**2)
    return ll_co+ll_ci
def log_prior(params):
    Tkin,nH2,Kvir=10**params
    ll=(5<Tkin<1000)*(1e1<nH2<1e7)*(1<Kvir<1e2)
    return ll

if os.path.exists(sampfile) and overwrite==False:
    sampler=emcee.backends.HDFBackend(sampfile, read_only=True)
else:
    backend = emcee.backends.HDFBackend(sampfile)
    backend.reset(nwalkers,p0.shape[1])
    with Pool(14) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelyhood, args=(),pool=pool,backend=backend)
        state=sampler.run_mcmc(p0,20000,progress=True)


samples=sampler.get_chain(discard=7000)
sample_reshape=samples.reshape([samples.shape[0]*samples.shape[1],samples.shape[2]])
log_prob_samples = sampler.get_log_prob(discard=7000)
max_index=np.where(log_prob_samples==np.max(log_prob_samples))
max_params=samples[max_index[0][0],max_index[1][0]]

def draw_corner(sampler,discard=7000):
    samples=sampler.get_chain(discard=discard)
    sample_reshape=samples.reshape([samples.shape[0]*samples.shape[1],samples.shape[2]])
    fig=corner.corner(sample_reshape, labels=['log Tkin','log nH2','log Kvir'])
    fig.savefig('fit_result/corner_Kvir_%g_%g.pdf'%Kvir_bond)
draw_corner(sampler)

import random
randidx=random.sample(list(np.arange(len(sample_reshape))), 1000)

plt.figure(figsize=[8,6])
plt.title(sourcename+' CO SLED')

for i in randidx:
    co_Tr,_=run_myradex_of(Tkin=10**sample_reshape[i,0],nH2=10**sample_reshape[i,1],abundance_Kvir=co_abundance/10**sample_reshape[i,2],molecule=co_mol,Tbg=Tback)
    co_model=co_Tr/co_Tr[0]
    #print(co_model)
    plt.plot(np.arange(max(co_Jup)+2)+1,co_model[np.arange(max(co_Jup)+2)]*(np.arange(max(co_Jup)+2)+1)**2,alpha=0.01,lw=2,color='green')
co_Tr,_=run_myradex_of(Tkin=10**max_params[0],nH2=10**max_params[1],abundance_Kvir=co_abundance/10**max_params[2],molecule=co_mol,Tbg=Tback)
co_model=co_Tr/co_Tr[0]
####
param_pdf=best_fitting(sample_reshape)
term1=r'$T_{\rm kin}=%3.1f_{%3.1f}^{+%3.1f}$K'%(10**max_params[0],10**(max_params[0]+param_pdf[0,1])-10**max_params[0],10**(max_params[0]+param_pdf[0,2])-10**max_params[0])
term2=r'log $n_{\rm H_2}=%3.1f_{%3.1f}^{+%3.1f}$'%(max_params[1],param_pdf[1,1],param_pdf[1,2])
term3=r'$K_{\rm vir}=%3.1f_{%3.1f}^{+%3.1f}$'%(10**max_params[2],10**(max_params[2]+param_pdf[2,1])-10**max_params[2],10**(max_params[2]+param_pdf[2,2])-10**max_params[2])
term4=r'<$T^{\rm CI}_{\rm 2-1}>/<T^{\rm CI}_{\rm 1-0}$>=%.3f'%(ci_model[1]/ci_model[0])
plt.errorbar(co_Jup,co_ratio*(co_Jup)**2,yerr=co_ratio_err*(co_Jup)**2,fmt='o',elinewidth=1,ms=1,mfc="w",mec='k',capthick=1,capsize=2)
plt.plot(np.arange(max(co_Jup)+2)+1,co_model[np.arange(max(co_Jup)+2)]*(np.arange(max(co_Jup)+2)+1)**2,'k',\
         label=term1+','+term2+','+term3+','+term4)
####
print(co_model)
plt.legend()
plt.xlabel('Jup')
plt.ylabel('normalized Flux')
plt.savefig('fit_result/CO_SLED_Kvir_%g_%g.pdf'%Kvir_bond)
