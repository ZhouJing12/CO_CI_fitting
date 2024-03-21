import numpy as np
from astropy import constants as const
from wrapper_my_radex import myradex_wrapper as wrapper
import re

phy_c_cgs=299792458e2
phy_c_SI=299792458
phy_k_cgs=1.3806503e-16
phy_h_cgs=6.62606896e-27
phy_cm2erg=phy_h_cgs*phy_c_cgs
phy_cm2K=phy_cm2erg/phy_k_cgs
class colli_partner:
    def __init__(self,part_name):
        self.name=part_name
    def set_n_trans(self,n_trans):
        self.n_transitions=n_trans
    def set_n_T(self,n_T):
        self.n_T=n_T
    def set_colli_temp(self,col_temp):
        self.colli_temperature=col_temp
    def set_Cul(self,Cul):
        self.Cul=Cul
class mol_data:
    def __init__(self,mol_dir,mol_file):
        self.mol_dir=mol_dir
        self.mol_file=mol_file
        f=open(self.mol_dir+self.mol_file)
        file_list=re.split('\n',f.read())
        f.close()
        if_read=True
        self.mol_name=re.split(' ',file_list[1])
        self.n_level=int(file_list[5])
        self.level_energy=np.zeros(self.n_level)
        self.level_weight=np.zeros(self.n_level)
        nrow=7
        for i in range(self.n_level):
            line=re.split(' |\t',file_list[nrow+i])
            line=[item for item in line if item!='']
            self.level_energy[i]=float(line[1])*phy_cm2K
            self.level_weight[i]=float(line[2])
        nrow+=self.n_level+1
        self.n_transitions=int(file_list[nrow])
        nrow+=2
        self.rad_data=np.zeros([self.n_transitions,9]) # iup, ilow, Aul, freq, lambda, Eup, Elow, Bul, Blu
        for i in range(self.n_transitions):
            line=re.split(' |\t',file_list[nrow+i])
            line=[item for item in line if item!='']
            iu=int(line[1])
            il=int(line[2])
            self.rad_data[i,0]=iu
            self.rad_data[i,1]=il
            self.rad_data[i,2]=float(line[3])
            self.rad_data[i,3]=phy_c_cgs*(self.level_energy[iu-1]-self.level_energy[il-1])/phy_cm2K
            self.rad_data[i,4]=phy_c_SI/self.rad_data[i,3]*1e6  # micron
            self.rad_data[i,5]=self.level_energy[iu-1]
            self.rad_data[i,6]=self.level_energy[il-1]
            self.rad_data[i,7]=self.rad_data[i,2]/((2*phy_h_cgs/phy_c_cgs**2)*self.rad_data[i,3]**3)
            self.rad_data[i,8]=self.rad_data[i,7]*self.level_weight[iu-1]/self.level_weight[il-1]
        nrow+=self.n_transitions+1
        self.n_partner=int(file_list[nrow])
        nrow+=2
        self.colli_data={}
        for i in range(self.n_partner):
            line=re.split(' \+ |-|\t| |:',file_list[nrow])
            line=[item for item in line if item!='']
            part_name=line[2]
            #print(line)
            if part_name=='electron':
                part_name='e'
            elif part_name=='with':
                part_name=line[3]
            elif line[3]=='H2':
                part_name=line[2]+line[3]
            partner=colli_partner(part_name)
            partner.set_n_trans(int(file_list[nrow+2]))
            partner.set_n_T(int(file_list[nrow+4]))
            line=re.split(' |\t',file_list[nrow+6])
            line=np.array([item for item in line if item!=''])
            partner.set_colli_temp(line.astype(float))
            Cul=np.zeros([partner.n_transitions,partner.n_T+2])
            for j in range(partner.n_transitions):
                line=re.split(' |\t',file_list[nrow+8+j])
                line=np.array([item for item in line if item!=''])
                Cul[j]=line[1:].astype(float)
            partner.set_Cul(Cul.T)
            self.colli_data[part_name]=partner
            nrow+=9+partner.n_transitions
        self.data_shape=[self.n_level,self.n_transitions,self.n_partner]
        self.partner_names=list(self.colli_data.keys())
        self.colli_shape=[[self.colli_data[key].n_transitions,self.colli_data[key].n_T] for key in self.colli_data.keys()]
        self.level_data=np.concatenate([[self.level_energy],[self.level_weight]],axis=0).T
        colli_T=np.zeros([self.n_partner,max([self.colli_data[key].n_T for key in self.partner_names])])
        colli_Cul=np.zeros([self.n_partner,max([self.colli_data[key].n_T for key in self.partner_names])+2,max([self.colli_data[key].n_transitions for key in self.partner_names])])
        for i in range(self.n_partner):
            #print(i,self.partner_names)
            partner_name=self.partner_names[i]
            colli_T[i,:self.colli_data[partner_name].n_T]=self.colli_data[partner_name].colli_temperature
            colli_Cul[i,:self.colli_data[partner_name].n_T+2,:self.colli_data[partner_name].n_transitions]=self.colli_data[partner_name].Cul
        self.colli_T=colli_T
        self.colli_Cul=colli_Cul
        space=' '
        self.part_name_str=space.join(self.partner_names)
        self.README='rad_data: [iup, ilow, Aul, freq, lambda, Eup, Elow, Bul, Blu] * transition\ncolli_data: [partner name]: Cul: transition* temperatures\n'


def dvdrvir(nH2):
    return 0.65*1.5**0.5*(nH2/1e3)**0.5
def run_myradex_of(Tkin=0,nH2=0,lamb=None,abundance_Kvir=1e-4,Tbg=2.73,molecule={},ini_occ=[]):
    if len(ini_occ)==0:
        ini_occ=molecule.level_weight*np.exp(-molecule.level_energy/Tkin)
        ini_occ=ini_occ/sum(ini_occ)
    if lamb is None:
        lamb = 3.08e18*abundance_Kvir/dvdrvir(nH2)
    NXcol=nH2*lamb
    params = {'tkin': Tkin,
              'ncol_x_cgs': NXcol,
              'h2_density_cgs': nH2,
              'hi_density_cgs': 0,
              'hp_density_cgs': 0,
              'e_density_cgs': 0,
              'tbg':Tbg,
              'mol_name':molecule.mol_name[0],
              'data_shape':molecule.data_shape,
              'n_transition':molecule.data_shape[1],
              'partner_names':molecule.part_name_str,
              'colli_shape':molecule.colli_shape,
              'level_data':molecule.level_data,
              'rad_data':molecule.rad_data,
              'colli_t':molecule.colli_T,
              'colli_data':molecule.colli_Cul,
              'ini_occ':ini_occ}      #initial occupation
    Tb,f_occupation=wrapper.run_one_params(**params)
    return Tb,f_occupation
h=const.h.cgs.value
c=const.c.cgs.value
k=const.k_B.cgs.value
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

def compute_Tr(Tkin,nH2,ab_kvir,Tback,molecule):                      # from Mangum et al. 2015
    Bul=molecule.rad_data[:,7]
    Aul=molecule.rad_data[:,2]
    freq=molecule.rad_data[:,3]
    Blu=molecule.rad_data[:,8]
    Eu=molecule.rad_data[:,5]
    El=molecule.rad_data[:,6]
    g0,g1,g2=molecule.level_weight
    NXcol=nH2*3.08e18*ab_kvir/dvdrvir(nH2)
    n0_nci,n1_nci,n2_nci=compute_population(Tkin,nH2,molecule,Tback=Tback)
    freq=molecule.rad_data[:,3]
    nu=np.array([n1_nci,n2_nci,n2_nci])
    nl=np.array([n0_nci,n1_nci,n0_nci])
    gu=np.array([g1,g2,g2])
    gl=np.array([g0,g1,g0])
    Tex=h*freq/const.k_B.cgs.value/np.log(nl*gu/nu/gl)
    df=1 /const.c.to('km/s').value*freq    #dv=1
    tau=h*freq/4/np.pi*(nl*Blu-nu*Bul)*NXcol/df
    BvTex=2*h*freq**3/c**2/(np.exp(h*freq/k/Tex)-1)
    BvTbg=2*h*freq**3/c**2/(np.exp(h*freq/k/Tback)-1)
    Tr=c**2/2/k/freq**2*(BvTex-BvTbg)*(1-np.exp(-tau))
    return Tr


def best_fitting(samples):
    if len(samples.shape)==2:
        nparm=samples.shape[-1]
        result=np.zeros([nparm,3])
        for i in range(nparm):
            mcmc=np.percentile(samples[:,i],[16,50,84])
            q=np.diff(mcmc)
            result[i]=[mcmc[1],-q[0],q[1]]
        return result
    elif len(samples.shape)==1:
        result=np.zeros(3)
        mcmc=np.percentile(samples,[16,50,84])
        q=np.diff(mcmc)
        result=[mcmc[1],-q[0],q[1]]
        return result

def log_value(value):
    if len(value.shape)==2:
        N=len(value)
        lg_value=np.zeros([N,2])
        lg_value[:,0]=np.log10(value[:,0])
        lg_value[:,1]=value[:,1]/value[:,0]/np.log(10)
        return lg_value
    elif len(value.shape)==1:
        lg_value=np.zeros(2)
        lg_value[0]=np.log10(value[0])
        lg_value[1]=value[1]/value[0]/np.log(10)
        return lg_value

def divation(aa,bb):
    result=np.zeros_like(aa)
    if len(aa.shape)==2:
        result[:,0]=aa[:,0]/bb[:,0]
        result[:,1]=np.sqrt((aa[:,1]/bb[:,0])**2+(bb[:,1]*aa[:,0]/bb[:,0]**2)**2)
        return result
    elif len(aa.shape)==1:
        result[0]=aa[0]/bb[0]
        result[1]=np.sqrt((aa[1]/bb[0])**2+(bb[1]*aa[0]/bb[0]**2)**2)
        return result

def limit_f(x,llim,ulim):
    if llim<x<ulim:
        return 0
    elif x<llim:
        return -0.5*((x-llim)/0.01)**4
    else:
        return -0.5*((x-ulim)/0.01)**4

def dvdrvir(nH2):
    return 0.65*1.5**0.5*(nH2/1e3)**0.5

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
h=phy_h_cgs
c=phy_c_cgs
k=phy_k_cgs
opH2_ratio=3

def compute_Tr(Tkin,nH2,NXcol,Tback,molecule):                      # from Mangum et al. 2015
    Bul=molecule.rad_data[:,7]
    Aul=molecule.rad_data[:,2]
    freq=molecule.rad_data[:,3]
    Blu=molecule.rad_data[:,8]
    Eu=molecule.rad_data[:,5]
    El=molecule.rad_data[:,6]
    g0,g1,g2=molecule.level_weight
    
    npH2=nH2/(1+opH2_ratio)
    noH2=nH2*opH2_ratio/(1+opH2_ratio)
    Kp,n0_nci_Kp,n1_nci_Kp,n2_nci_Kp=compute_population(Tkin,npH2,molecule,Tback,'pH2')
    Ko,n0_nci_Ko,n1_nci_Ko,n2_nci_Ko=compute_population(Tkin,noH2,molecule,Tback,'oH2')
    n0_nci,n1_nci,n2_nci=np.array([n0_nci_Kp,n1_nci_Kp,n2_nci_Kp])/Kp/(1+opH2_ratio)+np.array([n0_nci_Ko,n1_nci_Ko,n2_nci_Ko])/Ko*opH2_ratio/(1+opH2_ratio)
    
    freq=molecule.rad_data[:,3]
    nu=np.array([n1_nci,n2_nci,n2_nci])
    nl=np.array([n0_nci,n1_nci,n0_nci])
    gu=np.array([g1,g2,g2])
    gl=np.array([g0,g1,g0])
    Tex=h*freq/phy_k_cgs/np.log(nl*gu/nu/gl)
    df=1 /(phy_c_cgs/1e5)*freq    #dv=1
    tau=h*freq/4/np.pi*(nl*Blu-nu*Bul)*NXcol/df
    BvTex=2*h*freq**3/c**2/(np.exp(h*freq/k/Tex)-1)
    BvTbg=2*h*freq**3/c**2/(np.exp(h*freq/k/Tback)-1)
    Tr=c**2/2/k/freq**2*(BvTex-BvTbg)*(1-np.exp(-tau))
    return Tr
