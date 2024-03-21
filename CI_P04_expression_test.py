import numpy as np
import os
from my_script import mol_data
from astropy import constants as const
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

ci_mol=mol_data('./','catom.dat')
def interpolation(T,Ts_arr,values):
    if len(Ts_arr)!=len(values):
        print('Interpolution error: value interpolution not match!')
        return None
    if T<=Ts_arr[0]:
        return values[0]
    if T>=Ts_arr[-1]:
        return values[-1]
    for i in range(len(Ts_arr)-1):
        if Ts_arr[i]<=T<Ts_arr[i+1]:
            Tl,Tr=Ts_arr[i],Ts_arr[i+1]
            dl,dr=(Tr-T)/(Tr-Tl),(T-Tl)/(Tr-Tl)
            result=dl*values[i]+dr*values[i+1]
            return result

h=const.h.cgs.value
c=const.c.cgs.value
k=const.k_B.cgs.value

##### draw population using collision coefficient from LAMDA https://home.strw.leidenuniv.nl/~moldata/ #####

def compute_population(Tkin,n,ci_mol,Tback=2.73,opH2=3):
    Bul=ci_mol.rad_data[:,7]
    Aul=ci_mol.rad_data[:,2]
    freq=ci_mol.rad_data[:,3]
    Blu=ci_mol.rad_data[:,8]
    Eu=ci_mol.rad_data[:,5]
    El=ci_mol.rad_data[:,6]
    E1,E2=Eu[:2]
    E0=El[0]
    g0,g1,g2=ci_mol.level_weight
    npH2=n/(1+opH2)
    noH2=n*opH2/(1+opH2)
    Cul=interpolation(Tkin,ci_mol.colli_data['pH2'].colli_temperature,ci_mol.colli_data['pH2'].Cul[2:]) * npH2 + \
        interpolation(Tkin,ci_mol.colli_data['oH2'].colli_temperature,ci_mol.colli_data['oH2'].Cul[2:]) * noH2
    C10,C20,C21=Cul
    #### revise #####
    #C10,C20,C21 = np.array([1.3e-10, 2.0e-10, 7.28e-11])*n
    #### revise #####
    C01=C10*np.exp(-(E1-E0)/Tkin)*g1/g0
    C02=C20*np.exp(-(E2-E0)/Tkin)*g2/g0
    C12=C21*np.exp(-(E2-E1)/Tkin)*g2/g1
    if Tback>1e-6:
        expTb=np.exp(h*freq/k/Tback)
        ful=expTb/(expTb-1)
    else:
        ful=np.ones(3)
    f10,f21,f20=ful
    
    n10=Aul[0]/(C12+C10)*n
    n21=Aul[1]/(C20+C21)*n
    
    term1=(f10+g1/g0*(f10-1))*(C10+C12)/(C01+C10)*n10/n
    term2=g2/g1*(f21-1)*(C20+C21)/C12*n21/n
    term3=g1/g0*(f10-1)*(C10+C12)/(C20+C01)*n10/n
    # eq. A9
    K=1 + (1+C20/C21)*((f21+g2/g1*(f21-1))*n21/n + C10/C02*(1+f21*n21/n)*(1+C01/C10)*(1+term1)) + \
        + (1+f10*n10/n)*(C12+C10)/C21 + C10/C01*(1+C01/C20)*(1+term2)*(1+term3)
    # A6,A7,A8
    n1_nci_K=1 + f21*(1+C20/C21)*n21/n + C01/C02*(1+C20/C21)*(1+f21*n21/n)*(1+g1/g0*(f10-1)*(C12+C10)/C01*n10/n)
    term4=g2/g1*(f21-1)*(C21+C20)/C12*n21/n
    term5=g1/g0*(f10-1)*(C10+C12)/C01*n10/n
    n2_nci_K=(1+f10*n10/n)*(C10+C12)/C21 + g2/g1*(f21-1)*(1+C20/C21)*n21/n + C10/C20*(1+term4)*(1+term5)
    n0_nci_K=C10/C02*(1+f21*n21/n)*(1+C20/C21)*(1+f10*(1+C12/C10)*n10/n) + C10/C01*(1+term2)
    population=np.array([n0_nci_K,n1_nci_K,n2_nci_K])/K
    return population

os.system('mkdir population_plots')
Tkins=np.logspace(np.log10(1),np.log10(250),100)
nH2s=np.logspace(2,5,7)
cube=np.zeros([len(nH2s),len(Tkins)])
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol)
        cube[i,j]=pop[1]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q10')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q10_coefficient_from_LAMDA.pdf')

cube=np.zeros([len(nH2s),len(Tkins)])
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol)
        cube[i,j]=pop[2]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q21')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q21_coefficient_from_LAMDA.pdf')

redshift=2.15850
Tbg=2.73*(1+redshift)
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol,Tback=Tbg)
        cube[i,j]=pop[1]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q10')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.title('Tbg=%.2fK'%Tbg)
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q10_coefficient_from_LAMDA_Tbg_%.2f.pdf'%Tbg)

for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol,Tback=Tbg)
        cube[i,j]=pop[2]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q21')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.title('Tbg=%.2fK'%Tbg)
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q21_coefficient_from_LAMDA_Tbg_%.2f.pdf'%Tbg)


##### draw population using collision coefficient from Papadopoulos 2004 https://academic.oup.com/mnras/article-lookup/doi/10.1111/j.1365-2966.2004.07762.x #####

def compute_population(Tkin,n,ci_mol,Tback=2.73,opH2=3):
    Bul=ci_mol.rad_data[:,7]
    Aul=ci_mol.rad_data[:,2]
    freq=ci_mol.rad_data[:,3]
    Blu=ci_mol.rad_data[:,8]
    Eu=ci_mol.rad_data[:,5]
    El=ci_mol.rad_data[:,6]
    E1,E2=Eu[:2]
    E0=El[0]
    g0,g1,g2=ci_mol.level_weight
    #gu=ci_mol.level_weight[ci_mol.colli_data[partner].Cul[0].astype('int')-1]
    #gl=ci_mol.level_weight[ci_mol.colli_data[partner].Cul[1].astype('int')-1]
    npH2=n/(1+opH2)
    noH2=n*opH2/(1+opH2)
    Cul=interpolation(Tkin,ci_mol.colli_data['pH2'].colli_temperature,ci_mol.colli_data['pH2'].Cul[2:]) * npH2 + \
        interpolation(Tkin,ci_mol.colli_data['oH2'].colli_temperature,ci_mol.colli_data['oH2'].Cul[2:]) * noH2
    C10,C20,C21=Cul
    #### revise #####
    C10,C20,C21 = np.array([1.3e-10, 2.0e-10, 7.28e-11])*n
    #### revise #####
    C01=C10*np.exp(-(E1-E0)/Tkin)*g1/g0
    C02=C20*np.exp(-(E2-E0)/Tkin)*g2/g0
    C12=C21*np.exp(-(E2-E1)/Tkin)*g2/g1
    if Tback>1e-6:
        expTb=np.exp(h*freq/k/Tback)
        ful=expTb/(expTb-1)
    else:
        ful=1
    f10,f21,f20=ful

    n10=Aul[0]/(C12+C10)*n
    n21=Aul[1]/(C20+C21)*n

    term1=(f10+g1/g0*(f10-1))*(C10+C12)/(C01+C10)*n10/n
    term2=g2/g1*(f21-1)*(C20+C21)/C12*n21/n
    term3=g1/g0*(f10-1)*(C10+C12)/(C20+C01)*n10/n
    # eq. A9
    K=1 + (1+C20/C21)*((f21+g2/g1*(f21-1))*n21/n + C10/C02*(1+f21*n21/n)*(1+C01/C10)*(1+term1)) + \
        + (1+f10*n10/n)*(C12+C10)/C21 + C10/C01*(1+C01/C20)*(1+term2)*(1+term3)
    # A6,A7,A8
    n1_nci_K=1 + f21*(1+C20/C21)*n21/n + C01/C02*(1+C20/C21)*(1+f21*n21/n)*(1+g1/g0*(f10-1)*(C12+C10)/C01*n10/n)
    term4=g2/g1*(f21-1)*(C21+C20)/C12*n21/n
    term5=g1/g0*(f10-1)*(C10+C12)/C01*n10/n
    n2_nci_K=(1+f10*n10/n)*(C10+C12)/C21 + g2/g1*(f21-1)*(1+C20/C21)*n21/n + C10/C20*(1+term4)*(1+term5)
    n0_nci_K=C10/C02*(1+f21*n21/n)*(1+C20/C21)*(1+f10*(1+C12/C10)*n10/n) + C10/C01*(1+term2)
    population=np.array([n0_nci_K,n1_nci_K,n2_nci_K])/K
    return population

cube=np.zeros([len(nH2s),len(Tkins)])
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol)
        cube[i,j]=pop[1]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q10')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q10_coefficient_from_Pa2004.pdf')

cube=np.zeros([len(nH2s),len(Tkins)])
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol)
        cube[i,j]=pop[2]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q21')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q21_coefficient_from_Pa2004.pdf')

redshift=2.15850
Tbg=2.73*(1+redshift)
for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol,Tback=Tbg)
        cube[i,j]=pop[1]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q10')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.title('Tbg=%.2fK'%Tbg)
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q10_coefficient_from_Pa2004_Tbg_%.2f.pdf'%Tbg)

for i in range(len(nH2s)):
    for j in range(len(Tkins)):
        pop=compute_population(Tkins[j],nH2s[i],ci_mol,Tback=Tbg)
        cube[i,j]=pop[2]
fig, ax = plt.subplots()
for i in range(len(nH2s)):
    ax.plot(Tkins,cube[i],label='log nH2='+str(np.log10(nH2s[i])))
ax.legend()

ax.set_ylabel(r'Q21')
ax.set_xlabel(r'$\rm T_{kin}$')
plt.title('Tbg=%.2fK'%Tbg)
plt.xlim(0,150)
plt.ylim(0.05,0.6)
plt.savefig('population_plots/Q21_coefficient_from_Pa2004_Tbg_%.2f.pdf'%Tbg)

