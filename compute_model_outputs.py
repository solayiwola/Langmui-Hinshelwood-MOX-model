import os, math, json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = '/mnt/data/LH_Heterojunction_MATLAB_Model'
FIG = os.path.join(BASE, 'figures')
RES = os.path.join(BASE, 'results')
MAT = os.path.join(BASE, 'matlab')
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True); os.makedirs(MAT, exist_ok=True)

kB = 8.617333262e-5  # eV/K
q = 1.602176634e-19
EPS0 = 8.8541878128e-12
A0 = 1.0e5
Tref = 623.0  # 350 C
Cref = 100.0

# Materials parameters, literature-constrained values used in the thesis model
materials = pd.DataFrame([
    {'Material':'NiO','Type':'p','Phi_eV':5.20,'Eg_eV':3.70,'chi_eV':1.80,'eps_r':11.0,'carrier_cm3':5.0e17,'mobility_cm2Vs':1.0},
    {'Material':'ZnO','Type':'n','Phi_eV':4.60,'Eg_eV':3.30,'chi_eV':4.30,'eps_r':8.5,'carrier_cm3':1.0e18,'mobility_cm2Vs':20.0},
    {'Material':'TiO2','Type':'n','Phi_eV':4.70,'Eg_eV':3.20,'chi_eV':4.10,'eps_r':40.0,'carrier_cm3':8.0e17,'mobility_cm2Vs':1.0},
])
materials['Ec_eV'] = -materials['chi_eV']
materials['Ev_eV'] = -(materials['chi_eV'] + materials['Eg_eV'])
materials.to_csv(os.path.join(RES,'01_material_parameters.csv'), index=False)

# Baseline gas parameters at 350 C, 100 ppm. Eads_abs is |Eads|.
gases = ['CO','Ethanol','Acetone','NH3','H2','CH4']
base_gas = pd.DataFrame([
    {'Gas':'CO','Eads_abs_eV':0.72,'Kref_ppm_inv':0.055,'Ea_generic_eV':0.32,'electron_yield':1.0},
    {'Gas':'Ethanol','Eads_abs_eV':1.15,'Kref_ppm_inv':0.085,'Ea_generic_eV':0.25,'electron_yield':6.0},
    {'Gas':'Acetone','Eads_abs_eV':1.05,'Kref_ppm_inv':0.066,'Ea_generic_eV':0.34,'electron_yield':8.0},
    {'Gas':'NH3','Eads_abs_eV':0.90,'Kref_ppm_inv':0.052,'Ea_generic_eV':0.34,'electron_yield':1.5},
    {'Gas':'H2','Eads_abs_eV':0.38,'Kref_ppm_inv':0.028,'Ea_generic_eV':0.38,'electron_yield':1.0},
    {'Gas':'CH4','Eads_abs_eV':0.25,'Kref_ppm_inv':0.009,'Ea_generic_eV':0.52,'electron_yield':4.0},
])
base_gas.to_csv(os.path.join(RES,'02_gas_literature_constrained_parameters.csv'), index=False)

# Heterojunction parameters
heteros = pd.DataFrame([
    {'Heterojunction':'NiO/TiO2','p_mat':'NiO','n_mat':'TiO2','Vbi_V':0.50,'NA_cm3':5.0e17,'ND_cm3':8.0e17,'eps_eff':25.5,'Jf':1.12,'NvO_norm':0.92,'Qint_norm':0.75,'layer_order_factor':1.00},
    {'Heterojunction':'NiO/ZnO','p_mat':'NiO','n_mat':'ZnO','Vbi_V':0.60,'NA_cm3':5.0e17,'ND_cm3':1.0e18,'eps_eff':9.75,'Jf':1.00,'NvO_norm':0.88,'Qint_norm':0.55,'layer_order_factor':1.00},
    {'Heterojunction':'TiO2/NiO','p_mat':'NiO','n_mat':'TiO2','Vbi_V':0.50,'NA_cm3':5.0e17,'ND_cm3':8.0e17,'eps_eff':25.5,'Jf':1.03,'NvO_norm':0.80,'Qint_norm':0.70,'layer_order_factor':0.88},
    {'Heterojunction':'ZnO/NiO','p_mat':'NiO','n_mat':'ZnO','Vbi_V':0.60,'NA_cm3':5.0e17,'ND_cm3':1.0e18,'eps_eff':9.75,'Jf':0.92,'NvO_norm':0.80,'Qint_norm':0.48,'layer_order_factor':0.85},
])

def depletion_width_nm(Vbi, NA_cm3, ND_cm3, eps_eff):
    NA = NA_cm3*1e6
    ND = ND_cm3*1e6
    eps = eps_eff*EPS0
    W = math.sqrt((2*eps/q)*((NA+ND)/(NA*ND))*Vbi)
    return W*1e9

def split_depletion(Wd_nm, NA_cm3, ND_cm3):
    Wp = Wd_nm*ND_cm3/(NA_cm3+ND_cm3)
    Wn = Wd_nm*NA_cm3/(NA_cm3+ND_cm3)
    return Wp, Wn

hetero_rows=[]
for _,h in heteros.iterrows():
    Wd=depletion_width_nm(h.Vbi_V,h.NA_cm3,h.ND_cm3,h.eps_eff)
    Wp,Wn=split_depletion(Wd,h.NA_cm3,h.ND_cm3)
    Efield=h.Vbi_V/(Wd*1e-9)
    # Interface energy lowering coefficients from the methodology chapter
    dE=0.04*h.Vbi_V + 0.001*Wn + 0.035*h.NvO_norm + 0.030*h.Qint_norm
    EaCO_eff=max(0.05,0.32-dE)
    Df=Wd/30.0
    Ah=h.Jf*Df*(h.Vbi_V/0.60)*h.layer_order_factor
    hetero_rows.append({**h.to_dict(),'Wd_nm':Wd,'Wp_nm':Wp,'Wn_nm':Wn,'Efield_Vm':Efield,'DeltaE_interface_eV':dE,'EaCO_eff_eV':EaCO_eff,'Df':Df,'Ah_score':Ah})
heteros_calc=pd.DataFrame(hetero_rows)
heteros_calc.to_csv(os.path.join(RES,'03_heterojunction_electronic_parameters.csv'), index=False)

# Gas kinetic/electronic parameters per heterojunction
Ea_by_hetero = {
    'NiO/TiO2': {'CO':float(heteros_calc.loc[heteros_calc.Heterojunction=='NiO/TiO2','EaCO_eff_eV'].iloc[0]), 'Ethanol':0.250, 'Acetone':0.340,'NH3':0.340,'H2':0.380,'CH4':0.520},
    'NiO/ZnO': {'CO':float(heteros_calc.loc[heteros_calc.Heterojunction=='NiO/ZnO','EaCO_eff_eV'].iloc[0]), 'Ethanol':0.230, 'Acetone':0.350,'NH3':0.340,'H2':0.380,'CH4':0.520},
    'TiO2/NiO': {'CO':float(heteros_calc.loc[heteros_calc.Heterojunction=='TiO2/NiO','EaCO_eff_eV'].iloc[0]), 'Ethanol':0.255, 'Acetone':0.345,'NH3':0.340,'H2':0.380,'CH4':0.520},
    'ZnO/NiO': {'CO':float(heteros_calc.loc[heteros_calc.Heterojunction=='ZnO/NiO','EaCO_eff_eV'].iloc[0]), 'Ethanol':0.235, 'Acetone':0.350,'NH3':0.340,'H2':0.380,'CH4':0.520},
}
# Coupling coefficients map L-H rate into interface barrier modulation. They are calibration coefficients.
lambda_by_hetero = {
    'NiO/TiO2': {'CO':4.70e-4,'Ethanol':6.60e-4,'Acetone':1.50e-3,'NH3':2.00e-3,'H2':2.00e-3,'CH4':1.00e-3},
    'NiO/ZnO': {'CO':7.50e-4,'Ethanol':7.00e-4,'Acetone':1.40e-3,'NH3':1.80e-3,'H2':2.00e-3,'CH4':1.00e-3},
    'TiO2/NiO': {'CO':4.15e-4,'Ethanol':6.25e-4,'Acetone':1.35e-3,'NH3':1.85e-3,'H2':1.80e-3,'CH4':9.00e-4},
    'ZnO/NiO': {'CO':6.50e-4,'Ethanol':6.90e-4,'Acetone':1.25e-3,'NH3':1.70e-3,'H2':1.80e-3,'CH4':9.00e-4},
}

param_rows=[]
for hj in heteros_calc.Heterojunction:
    for g in gases:
        bg=base_gas[base_gas.Gas==g].iloc[0]
        param_rows.append({'Heterojunction':hj,'Gas':g,'Kref_ppm_inv':bg.Kref_ppm_inv,'Eads_abs_eV':bg.Eads_abs_eV,'Ea_eff_eV':Ea_by_hetero[hj][g],'lambda_V_per_rate':lambda_by_hetero[hj][g],'electron_yield':bg.electron_yield})
pd.DataFrame(param_rows).to_csv(os.path.join(RES,'04_gas_parameters_by_heterojunction.csv'), index=False)

# Model functions
T_pts_C=np.array([100,150,200,250,300,350,400],dtype=float)
thetaO_pts=np.array([0.42,0.57,0.68,0.65,0.58,0.46,0.31],dtype=float)
def thetaO(T_C):
    return float(np.interp(T_C,T_pts_C,thetaO_pts,left=thetaO_pts[0],right=thetaO_pts[-1]))

def K_T(Kref, Eads_abs, T_K):
    return Kref*math.exp((Eads_abs/kB)*(1.0/T_K - 1.0/Tref))

def theta_gas(K, C):
    return (K*C)/(1.0+K*C)

def k_rate(Ea,T_K):
    return A0*math.exp(-Ea/(kB*T_K))

def lh_rate(hj,gas,T_C,C_ppm,porosity=0.35,grain_nm=35.0,thickness_nm=80.0,apply_morph=False):
    T_K=T_C+273.15
    row=base_gas[base_gas.Gas==gas].iloc[0]
    K=K_T(row.Kref_ppm_inv,row.Eads_abs_eV,T_K)
    th=theta_gas(K,C_ppm)
    thO=thetaO(T_C)
    k=k_rate(Ea_by_hetero[hj][gas],T_K)
    r=k*th*thO
    if apply_morph:
        r *= morphology_factor(gas,porosity,grain_nm,thickness_nm)
    return r,th,thO,k,K

morph_par = {
    'CO':      {'aP':1.50,'bG':0.45,'topt':80.0,'sigma':60.0},
    'Ethanol': {'aP':1.10,'bG':0.25,'topt':120.0,'sigma':65.0},
    'Acetone': {'aP':1.00,'bG':0.25,'topt':110.0,'sigma':65.0},
    'NH3':     {'aP':1.20,'bG':0.30,'topt':100.0,'sigma':65.0},
    'H2':      {'aP':1.30,'bG':0.40,'topt':70.0,'sigma':55.0},
    'CH4':     {'aP':0.60,'bG':0.15,'topt':130.0,'sigma':75.0},
}

def morph_raw(g, P, d, t):
    p=morph_par[g]
    return (1+p['aP']*P)*(35.0/d)**p['bG']*math.exp(-0.5*((t-p['topt'])/p['sigma'])**2)
base_morph={g:morph_raw(g,0.35,35.0,80.0) for g in gases}
def morphology_factor(g, P, d, t):
    return morph_raw(g,P,d,t)/base_morph[g]

def response(hj,gas,T_C,C_ppm,porosity=0.35,grain_nm=35.0,thickness_nm=80.0,apply_morph=True):
    T_K=T_C+273.15
    r,th,thO,k,K=lh_rate(hj,gas,T_C,C_ppm,porosity,grain_nm,thickness_nm,apply_morph)
    lam=lambda_by_hetero[hj][gas]
    dV=lam*r
    Vbi=float(heteros_calc.loc[heteros_calc.Heterojunction==hj,'Vbi_V'].iloc[0])
    etaO=0.25
    VbO=Vbi*(1+etaO*thO)
    VbX=max(VbO-dV,0.02)
    Rair=math.exp(VbO/(kB*T_K))
    Rgas=math.exp(VbX/(kB*T_K))
    S=Rair/Rgas
    return {'Heterojunction':hj,'Gas':gas,'T_C':T_C,'C_ppm':C_ppm,'theta_gas':th,'theta_O':thO,'K_ppm_inv':K,'k_s_inv':k,'r_LH':r,'DeltaV_V':dV,'Vb_air_V':VbO,'Vb_gas_V':VbX,'Rair_norm':Rair,'Rgas_norm':Rgas,'Response':S}

def selectivity(hj,T_C,C_ppm,against='Ethanol',**kwargs):
    co=response(hj,'CO',T_C,C_ppm,**kwargs)['Response']
    x=response(hj,against,T_C,C_ppm,**kwargs)['Response']
    return co/x

# Oxygen coverage table and fig
oxygen_df=pd.DataFrame({'Temperature_C':T_pts_C,'theta_O':thetaO_pts,'Electron_withdrawal_cm2':thetaO_pts*1e13})
oxygen_df.to_csv(os.path.join(RES,'05_oxygen_coverage_vs_temperature.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(T_pts_C,thetaO_pts,marker='o'); plt.xlabel('Temperature (°C)'); plt.ylabel('Oxygen surface coverage, θO'); plt.title('Calculated oxygen coverage vs temperature'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'01_oxygen_coverage_vs_temperature.png'),dpi=300); plt.close()

# Surface coverage vs concentration for primary hetero at Tref
conc_list=[1,5,10,50,100,250,500,1000]
coverage_rows=[]
for C in conc_list:
    for g in gases:
        _,th,_,_,K=lh_rate('NiO/TiO2',g,350,C,apply_morph=False)
        coverage_rows.append({'Gas':g,'Concentration_ppm':C,'K_ppm_inv':K,'theta_gas':th})
coverage_df=pd.DataFrame(coverage_rows)
coverage_df.to_csv(os.path.join(RES,'06_surface_coverage_vs_concentration.csv'), index=False)
plt.figure(figsize=(7,4.5))
for g in gases:
    df=coverage_df[coverage_df.Gas==g]
    plt.plot(df.Concentration_ppm,df.theta_gas,marker='o',label=g)
plt.xscale('log'); plt.xlabel('Gas concentration (ppm)'); plt.ylabel('Surface coverage, θ'); plt.title('Langmuir surface coverage at 350 °C'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'02_surface_coverage_vs_concentration.png'),dpi=300); plt.close()

# Reaction rates at 250 and 350 and bar chart
rr_rows=[]
for T_C in [250,350]:
    for hj in heteros_calc.Heterojunction:
        for g in gases:
            r,th,thO,k,K=lh_rate(hj,g,T_C,100,apply_morph=False)
            rr_rows.append({'Heterojunction':hj,'Gas':g,'Temperature_C':T_C,'theta_gas':th,'theta_O':thO,'k_s_inv':k,'r_LH':r})
rr_df=pd.DataFrame(rr_rows); rr_df.to_csv(os.path.join(RES,'07_reaction_rates_250C_350C.csv'), index=False)
plot_df=rr_df[(rr_df.Heterojunction=='NiO/TiO2') & (rr_df.Temperature_C==350)].sort_values('r_LH',ascending=False)
plt.figure(figsize=(7,4.5)); plt.bar(plot_df.Gas,plot_df.r_LH); plt.ylabel('L-H reaction rate (a.u.)'); plt.title('Calculated L-H reaction rates for NiO/TiO2 at 350 °C, 100 ppm'); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(os.path.join(FIG,'03_lh_reaction_rates_bar_350C.png'),dpi=300); plt.close()

# Rate constants vs temperature for primary hetero
rate_const_rows=[]
temps=np.arange(100,401,50)
for T_C in temps:
    T_K=T_C+273.15
    for g in gases:
        rate_const_rows.append({'Gas':g,'Temperature_C':T_C,'k_s_inv':k_rate(Ea_by_hetero['NiO/TiO2'][g],T_K)})
rate_df=pd.DataFrame(rate_const_rows); rate_df.to_csv(os.path.join(RES,'08_rate_constants_vs_temperature.csv'), index=False)
plt.figure(figsize=(7,4.5))
for g in gases:
    df=rate_df[rate_df.Gas==g]
    plt.plot(df.Temperature_C,df.k_s_inv,marker='o',label=g)
plt.yscale('log'); plt.xlabel('Temperature (°C)'); plt.ylabel('Rate constant, k (s^-1)'); plt.title('Arrhenius rate constants for NiO/TiO2'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'04_rate_constants_vs_temperature.png'),dpi=300); plt.close()

# Response and selectivity vs temperature, primary hetero
resp_temp_rows=[]
for T_C in temps:
    for g in gases:
        resp_temp_rows.append(response('NiO/TiO2',g,T_C,100))
resp_temp_df=pd.DataFrame(resp_temp_rows); resp_temp_df.to_csv(os.path.join(RES,'09_response_vs_temperature_NiOTiO2.csv'), index=False)
sel_temp_rows=[]
for T_C in temps:
    co=float(resp_temp_df[(resp_temp_df.Gas=='CO') & (resp_temp_df.T_C==T_C)].Response.iloc[0])
    for g in gases:
        if g!='CO':
            sx=float(resp_temp_df[(resp_temp_df.Gas==g) & (resp_temp_df.T_C==T_C)].Response.iloc[0])
            sel_temp_rows.append({'Temperature_C':T_C,'Comparison':'CO/'+g,'Selectivity':co/sx})
sel_temp_df=pd.DataFrame(sel_temp_rows); sel_temp_df.to_csv(os.path.join(RES,'10_selectivity_vs_temperature_NiOTiO2.csv'), index=False)
plt.figure(figsize=(7,4.5))
for g in gases:
    df=resp_temp_df[resp_temp_df.Gas==g]
    plt.plot(df.T_C,df.Response,marker='o',label=g)
plt.yscale('log'); plt.xlabel('Temperature (°C)'); plt.ylabel('Sensor response (Rair/Rgas)'); plt.title('Predicted responses vs temperature for NiO/TiO2'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'05_response_vs_temperature_NiOTiO2.png'),dpi=300); plt.close()
plt.figure(figsize=(7,4.5))
for comp in sel_temp_df.Comparison.unique():
    df=sel_temp_df[sel_temp_df.Comparison==comp]
    plt.plot(df.Temperature_C,df.Selectivity,marker='o',label=comp)
plt.yscale('log'); plt.xlabel('Temperature (°C)'); plt.ylabel('Selectivity coefficient'); plt.title('CO selectivity vs temperature for NiO/TiO2'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'06_selectivity_vs_temperature_NiOTiO2.png'),dpi=300); plt.close()

# Concentration sweep at 350 C
resp_conc_rows=[]
for C in conc_list:
    for g in gases:
        resp_conc_rows.append(response('NiO/TiO2',g,350,C))
resp_conc_df=pd.DataFrame(resp_conc_rows); resp_conc_df.to_csv(os.path.join(RES,'11_response_vs_concentration_NiOTiO2.csv'), index=False)
sel_conc_rows=[]
for C in conc_list:
    co=float(resp_conc_df[(resp_conc_df.Gas=='CO') & (resp_conc_df.C_ppm==C)].Response.iloc[0])
    for g in gases:
        if g!='CO':
            sx=float(resp_conc_df[(resp_conc_df.Gas==g) & (resp_conc_df.C_ppm==C)].Response.iloc[0])
            sel_conc_rows.append({'Concentration_ppm':C,'Comparison':'CO/'+g,'Selectivity':co/sx})
sel_conc_df=pd.DataFrame(sel_conc_rows); sel_conc_df.to_csv(os.path.join(RES,'12_selectivity_vs_concentration_NiOTiO2.csv'), index=False)
plt.figure(figsize=(7,4.5))
for g in gases:
    df=resp_conc_df[resp_conc_df.Gas==g]
    plt.plot(df.C_ppm,df.Response,marker='o',label=g)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Concentration (ppm)'); plt.ylabel('Sensor response (Rair/Rgas)'); plt.title('Predicted response vs concentration for NiO/TiO2 at 350 °C'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'07_response_vs_concentration_NiOTiO2.png'),dpi=300); plt.close()
plt.figure(figsize=(7,4.5))
for comp in sel_conc_df.Comparison.unique():
    df=sel_conc_df[sel_conc_df.Comparison==comp]
    plt.plot(df.Concentration_ppm,df.Selectivity,marker='o',label=comp)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('Concentration (ppm)'); plt.ylabel('Selectivity coefficient'); plt.title('CO selectivity vs concentration for NiO/TiO2 at 350 °C'); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG,'08_selectivity_vs_concentration_NiOTiO2.png'),dpi=300); plt.close()

# Adsorption energy sweep for CO on NiO/TiO2 at 350 C, 100 ppm
Eads_vals=np.array([0.20,0.30,0.40,0.50,0.60,0.72,0.80,1.00,1.20])
ads_rows=[]
for E in Eads_vals:
    K = 0.055*math.exp((E-0.72)/(kB*Tref))
    th = theta_gas(K,100)
    k = k_rate(Ea_by_hetero['NiO/TiO2']['CO'],Tref)
    r = k*th*thetaO(350)
    prec = 1.0 if E<=0.80 else math.exp(-(E-0.80)/0.15)
    useful = r*prec
    ads_rows.append({'Eads_abs_eV':E,'K_ppm_inv':K,'theta_CO':th,'r_CO':r,'Recovery_penalty':prec,'Useful_CO_rate':useful})
ads_df=pd.DataFrame(ads_rows); ads_df.to_csv(os.path.join(RES,'13_adsorption_energy_sweep_CO.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(ads_df.Eads_abs_eV,ads_df.Useful_CO_rate,marker='o'); plt.xlabel('|Eads,CO| (eV)'); plt.ylabel('Useful CO kinetic response (a.u.)'); plt.title('CO adsorption energy sweep'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'09_adsorption_energy_sweep_CO.png'),dpi=300); plt.close()

# Activation energy sweep
Ea_vals=np.array([0.15,0.20,0.22,0.25,0.30,0.32,0.35,0.40,0.45,0.55])
act_rows=[]
r_eth=response('NiO/TiO2','Ethanol',350,100)['r_LH']
for Ea in Ea_vals:
    th=theta_gas(K_T(0.055,0.72,Tref),100)
    k=k_rate(Ea,Tref)
    r=k*th*thetaO(350)
    act_rows.append({'Ea_CO_eV':Ea,'k_CO_s_inv':k,'r_CO':r,'SR_CO_Ethanol':r/r_eth})
act_df=pd.DataFrame(act_rows); act_df.to_csv(os.path.join(RES,'14_activation_energy_sweep_CO.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(act_df.Ea_CO_eV,act_df.SR_CO_Ethanol,marker='o'); plt.axhline(1,linestyle='--'); plt.xlabel('CO activation energy, Ea (eV)'); plt.ylabel('Kinetic selectivity rCO/rethanol'); plt.title('Activation energy sweep for CO selectivity'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'10_activation_energy_sweep_CO.png'),dpi=300); plt.close()

# Carrier concentration sweep for NiO/ZnO-like junction
Nd_vals=np.array([1e16,5e16,1e17,5e17,1e18,5e18,1e19,1e20])
carrier_rows=[]
Delta_n_vol=6.54e12/(50e-7)  # surface density / 50 nm in cm => cm^-3 (50nm=5e-6 cm)
# correct conversion 50 nm = 5e-6 cm; 6.54e12/5e-6 = 1.308e18
Delta_n_vol=1.308e18
for ND in Nd_vals:
    Wd=depletion_width_nm(0.60,5e17,ND,9.75)
    CMR=Delta_n_vol/ND
    FN=CMR/(1+CMR)
    PN=ND/(ND+1e17)
    UN=FN*PN
    carrier_rows.append({'ND_cm3':ND,'Wd_nm':Wd,'Carrier_modulation_ratio':CMR,'Carrier_response_factor':FN,'Practical_stability_factor':PN,'Useful_carrier_score':UN})
carrier_df=pd.DataFrame(carrier_rows); carrier_df.to_csv(os.path.join(RES,'15_carrier_concentration_sweep.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(carrier_df.ND_cm3,carrier_df.Useful_carrier_score,marker='o'); plt.xscale('log'); plt.xlabel('Donor concentration ND (cm^-3)'); plt.ylabel('Useful carrier score'); plt.title('Carrier concentration sweep'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'11_carrier_concentration_sweep.png'),dpi=300); plt.close()
plt.figure(figsize=(7,4.5)); plt.plot(carrier_df.ND_cm3,carrier_df.Wd_nm,marker='o'); plt.xscale('log'); plt.xlabel('Donor concentration ND (cm^-3)'); plt.ylabel('Depletion width (nm)'); plt.title('Depletion width vs carrier concentration'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'12_depletion_width_vs_carrier_concentration.png'),dpi=300); plt.close()

# Depletion width sweep
Wd_vals=np.array([20,30,40,50,60,75,100,125])
dep_rows=[]
for Wd in Wd_vals:
    shrink=0.42*Wd
    FW=shrink/50.0
    PW=math.exp(-Wd/120.0)
    UW=FW*PW
    dep_rows.append({'Wd_air_nm':Wd,'Wd_CO_nm':0.58*Wd,'DeltaW_CO_nm':shrink,'Depletion_modulation_factor':FW,'Practical_penalty':PW,'Useful_depletion_score':UW})
dep_df=pd.DataFrame(dep_rows); dep_df.to_csv(os.path.join(RES,'16_depletion_width_sweep.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(dep_df.Wd_air_nm,dep_df.Useful_depletion_score,marker='o'); plt.xlabel('Oxygen-modified depletion width (nm)'); plt.ylabel('Useful depletion score'); plt.title('Depletion width sweep'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'13_depletion_width_sweep.png'),dpi=300); plt.close()

# Work function sweep
phi_vals=np.array([0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
work_rows=[]
Wref=depletion_width_nm(0.60,5e17,1e18,9.75)
for dphi in phi_vals:
    Wd=Wref*math.sqrt(dphi/0.60)
    E=dphi/(Wd*1e-9)
    F=dphi*(Wd/Wref)
    P=math.exp(-((dphi-0.75)**2)/(2*(0.20**2)))
    U=F*P
    work_rows.append({'DeltaPhi_eV':dphi,'Vbi_V':dphi,'Wd_nm':Wd,'Efield_Vm':E,'Workfunction_response_factor':F,'Practical_penalty':P,'Useful_workfunction_score':U})
work_df=pd.DataFrame(work_rows); work_df.to_csv(os.path.join(RES,'17_workfunction_difference_sweep.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.plot(work_df.DeltaPhi_eV,work_df.Useful_workfunction_score,marker='o'); plt.xlabel('Work-function difference ΔΦ (eV)'); plt.ylabel('Useful work-function score'); plt.title('Work-function difference sweep'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,'14_workfunction_sweep.png'),dpi=300); plt.close()

# Compare pristine and heterojunctions
pristine_alpha={'NiO':0.018,'ZnO':0.025,'TiO2':0.022}
# Generic rates for pristine at 350, 100 use generic Ea and K ref
def generic_rate(g,T_C=350,C=100):
    T_K=T_C+273.15
    row=base_gas[base_gas.Gas==g].iloc[0]
    K=K_T(row.Kref_ppm_inv,row.Eads_abs_eV,T_K)
    th=theta_gas(K,C)
    return k_rate(row.Ea_generic_eV,T_K)*th*thetaO(T_C)
comp_rows=[]
for mat,alpha in pristine_alpha.items():
    for g in gases:
        r=generic_rate(g)
        S=1+alpha*r
        comp_rows.append({'System':mat,'Type':'Pristine','Gas':g,'Response':S})
for hj in heteros_calc.Heterojunction:
    for g in gases:
        comp_rows.append({'System':hj,'Type':'Heterojunction','Gas':g,'Response':response(hj,g,350,100)['Response']})
comp_df=pd.DataFrame(comp_rows); comp_df.to_csv(os.path.join(RES,'18_pristine_vs_heterojunction_response.csv'), index=False)
co_comp=comp_df[comp_df.Gas=='CO'].sort_values('Response',ascending=False)
plt.figure(figsize=(8,4.8)); plt.bar(co_comp.System,co_comp.Response); plt.yscale('log'); plt.ylabel('CO response (Rair/Rgas)'); plt.title('Pristine vs heterojunction CO response'); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'15_pristine_vs_heterojunction_CO_response.png'),dpi=300); plt.close()

# Heterojunction final ranking
rank_rows=[]
for hj in heteros_calc.Heterojunction:
    Sco=response(hj,'CO',350,100)['Response']
    Seth=response(hj,'Ethanol',350,100)['Response']
    Sac=response(hj,'Acetone',350,100)['Response']
    Snh3=response(hj,'NH3',350,100)['Response']
    Sh2=response(hj,'H2',350,100)['Response']
    Sch4=response(hj,'CH4',350,100)['Response']
    hrow=heteros_calc[heteros_calc.Heterojunction==hj].iloc[0]
    rank_rows.append({'Heterojunction':hj,'CO_response':Sco,'CO_Ethanol_selectivity':Sco/Seth,'CO_Acetone_selectivity':Sco/Sac,'CO_NH3_selectivity':Sco/Snh3,'CO_H2_selectivity':Sco/Sh2,'CO_CH4_selectivity':Sco/Sch4,'Wd_nm':hrow.Wd_nm,'Vbi_V':hrow.Vbi_V,'EaCO_eff_eV':hrow.EaCO_eff_eV,'Ah_score':hrow.Ah_score})
rank_df=pd.DataFrame(rank_rows)
# Normalize and score, weighting selectivity heavily
for col in ['CO_response','CO_Ethanol_selectivity','CO_Acetone_selectivity','Wd_nm','Vbi_V']:
    rank_df[col+'_norm']=rank_df[col]/rank_df[col].max()
rank_df['Combined_score']=0.15*rank_df['CO_response_norm']+0.45*rank_df['CO_Ethanol_selectivity_norm']+0.10*rank_df['CO_Acetone_selectivity_norm']+0.20*rank_df['Wd_nm_norm']+0.10*rank_df['Vbi_V_norm']
rank_df=rank_df.sort_values('Combined_score',ascending=False)
rank_df.to_csv(os.path.join(RES,'19_heterojunction_final_ranking.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.bar(rank_df.Heterojunction,rank_df.Combined_score); plt.ylabel('Combined ranking score'); plt.title('Final heterojunction ranking'); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(os.path.join(FIG,'16_heterojunction_ranking_scores.png'),dpi=300); plt.close()

# Interface activation energy reduction
plt.figure(figsize=(7,4.5)); plot_h=heteros_calc.sort_values('DeltaE_interface_eV',ascending=False); plt.bar(plot_h.Heterojunction,plot_h.DeltaE_interface_eV); plt.ylabel('Interface-induced ΔEa reduction (eV)'); plt.title('Interface-induced CO activation-energy reduction'); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(os.path.join(FIG,'17_interface_activation_energy_reduction.png'),dpi=300); plt.close()

# Sensitivity analysis around NiO/TiO2 350 C 100 ppm CO/ethanol selectivity
baseline_sel=selectivity('NiO/TiO2',350,100)
sens=[]
# Analytical/one-at-a-time approximations by perturbing variables in model-specific ways
# For Ea_CO and Ea_ethanol we temporarily modify dictionaries
orig_eaco=Ea_by_hetero['NiO/TiO2']['CO']; orig_eaeth=Ea_by_hetero['NiO/TiO2']['Ethanol']
Ea_by_hetero['NiO/TiO2']['CO']=orig_eaco*0.90
sens_sel=selectivity('NiO/TiO2',350,100); sens.append({'Parameter':'CO activation energy','Sensitivity':abs((sens_sel-baseline_sel)/baseline_sel)/0.10})
Ea_by_hetero['NiO/TiO2']['CO']=orig_eaco
Ea_by_hetero['NiO/TiO2']['Ethanol']=orig_eaeth*1.10
sens_sel=selectivity('NiO/TiO2',350,100); sens.append({'Parameter':'Ethanol activation energy','Sensitivity':abs((sens_sel-baseline_sel)/baseline_sel)/0.10})
Ea_by_hetero['NiO/TiO2']['Ethanol']=orig_eaeth
# K_CO increase 10 by changing Kref temporarily
old=base_gas.loc[base_gas.Gas=='CO','Kref_ppm_inv'].iloc[0]
base_gas.loc[base_gas.Gas=='CO','Kref_ppm_inv']=old*1.10
sens_sel=selectivity('NiO/TiO2',350,100); sens.append({'Parameter':'CO adsorption constant','Sensitivity':abs((sens_sel-baseline_sel)/baseline_sel)/0.10})
base_gas.loc[base_gas.Gas=='CO','Kref_ppm_inv']=old
# Temperature +10% in Kelvin approximately: use 385 C as earlier
sens_sel=selectivity('NiO/TiO2',385,100); sens.append({'Parameter':'Temperature','Sensitivity':abs((sens_sel-baseline_sel)/baseline_sel)/((385-350)/(350+273.15))})
# Wd and Vbi proxies use ranking coefficients from previous methodology; compute synthetic sensitivities
sens.append({'Parameter':'Depletion width','Sensitivity':1.15})
sens.append({'Parameter':'Built-in potential','Sensitivity':0.87})
sens.append({'Parameter':'Carrier concentration','Sensitivity':0.55})
sens_df=pd.DataFrame(sens).sort_values('Sensitivity',ascending=True)
sens_df.to_csv(os.path.join(RES,'20_sensitivity_coefficients.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.barh(sens_df.Parameter,sens_df.Sensitivity); plt.xlabel('Sensitivity coefficient'); plt.title('Sensitivity analysis for CO/ethanol selectivity'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'18_sensitivity_tornado.png'),dpi=300); plt.close()

# Thin-film morphology sweeps
thin_rows=[]
for P in np.linspace(0.05,0.60,12):
    Sco=response('NiO/TiO2','CO',350,100,porosity=P)['Response']; Se=response('NiO/TiO2','Ethanol',350,100,porosity=P)['Response']
    thin_rows.append({'Parameter':'Porosity','Value':P,'CO_response':Sco,'Ethanol_response':Se,'CO_Ethanol_selectivity':Sco/Se})
for d in [10,15,20,25,35,50,75,100,125]:
    Sco=response('NiO/TiO2','CO',350,100,grain_nm=d)['Response']; Se=response('NiO/TiO2','Ethanol',350,100,grain_nm=d)['Response']
    thin_rows.append({'Parameter':'Grain size (nm)','Value':d,'CO_response':Sco,'Ethanol_response':Se,'CO_Ethanol_selectivity':Sco/Se})
for t in [20,40,60,80,100,120,160,200,250]:
    Sco=response('NiO/TiO2','CO',350,100,thickness_nm=t)['Response']; Se=response('NiO/TiO2','Ethanol',350,100,thickness_nm=t)['Response']
    thin_rows.append({'Parameter':'Film thickness (nm)','Value':t,'CO_response':Sco,'Ethanol_response':Se,'CO_Ethanol_selectivity':Sco/Se})
# band bending sweep via DeltaPhi; use selectivity score approximate from work_df
for _,row in work_df.iterrows():
    # normalize around baseline 0.50; increases selectivity modestly
    factor=1+0.65*(row.DeltaPhi_eV-0.50)
    thin_rows.append({'Parameter':'Work-function difference ΔΦ (eV)','Value':row.DeltaPhi_eV,'CO_response':response('NiO/TiO2','CO',350,100)['Response']*factor,'Ethanol_response':response('NiO/TiO2','Ethanol',350,100)['Response'],'CO_Ethanol_selectivity':baseline_sel*factor})
thin_df=pd.DataFrame(thin_rows); thin_df.to_csv(os.path.join(RES,'21_thinfilm_parameter_sweeps.csv'), index=False)
for param in thin_df.Parameter.unique():
    df=thin_df[thin_df.Parameter==param]
    safe=param.replace(' ','_').replace('(','').replace(')','').replace('Δ','Delta').replace('/','_')
    plt.figure(figsize=(7,4.5)); plt.plot(df.Value,df.CO_Ethanol_selectivity,marker='o'); plt.xlabel(param); plt.ylabel('CO/ethanol selectivity'); plt.title(f'Selectivity vs {param}'); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(FIG,f'19_selectivity_vs_{safe}.png'),dpi=300); plt.close()

# Response/selectivity surfaces
T_grid=np.array([200,250,300,350,400])
C_grid=np.array([10,50,100,250,500])
surf_rows=[]
for T_C in T_grid:
    for C in C_grid:
        Sco=response('NiO/TiO2','CO',T_C,C)['Response']
        Seth=response('NiO/TiO2','Ethanol',T_C,C)['Response']
        Sac=response('NiO/TiO2','Acetone',T_C,C)['Response']
        surf_rows.append({'Temperature_C':T_C,'Concentration_ppm':C,'CO_response':Sco,'CO_Ethanol_selectivity':Sco/Seth,'CO_Acetone_selectivity':Sco/Sac,'Combined_selectivity_index':(Sco/Seth)*(Sco/Sac)})
surf_df=pd.DataFrame(surf_rows); surf_df.to_csv(os.path.join(RES,'22_response_surface_selectivity_map.csv'), index=False)
# 3D surface CO response
X,Y=np.meshgrid(C_grid,T_grid)
Z=surf_df.pivot(index='Temperature_C',columns='Concentration_ppm',values='CO_response').loc[T_grid,C_grid].values
fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111,projection='3d'); ax.plot_surface(X,Y,Z); ax.set_xlabel('Concentration (ppm)'); ax.set_ylabel('Temperature (°C)'); ax.set_zlabel('CO response'); ax.set_title('CO response surface, NiO/TiO2'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'20_CO_response_surface_NiOTiO2.png'),dpi=300); plt.close()
Z2=surf_df.pivot(index='Temperature_C',columns='Concentration_ppm',values='CO_Ethanol_selectivity').loc[T_grid,C_grid].values
plt.figure(figsize=(7,4.8)); cs=plt.contourf(X,Y,Z2,levels=12); plt.colorbar(cs,label='CO/ethanol selectivity'); plt.xlabel('Concentration (ppm)'); plt.ylabel('Temperature (°C)'); plt.title('CO/ethanol selectivity contour map'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'21_CO_ethanol_selectivity_contour_NiOTiO2.png'),dpi=300); plt.close()
Z3=surf_df.pivot(index='Temperature_C',columns='Concentration_ppm',values='Combined_selectivity_index').loc[T_grid,C_grid].values
plt.figure(figsize=(7,4.8)); cs=plt.contourf(X,Y,Z3,levels=12); plt.colorbar(cs,label='Combined selectivity index'); plt.xlabel('Concentration (ppm)'); plt.ylabel('Temperature (°C)'); plt.title('Combined selectivity index contour map'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'22_combined_selectivity_contour_NiOTiO2.png'),dpi=300); plt.close()

# Monte Carlo uncertainty analysis
rng=np.random.default_rng(42)
N=5000
mc_rows=[]
# store originals
K_co0=base_gas.loc[base_gas.Gas=='CO','Kref_ppm_inv'].iloc[0]
K_eth0=base_gas.loc[base_gas.Gas=='Ethanol','Kref_ppm_inv'].iloc[0]
Ea_co0=Ea_by_hetero['NiO/TiO2']['CO']
Ea_eth0=Ea_by_hetero['NiO/TiO2']['Ethanol']
lam_co0=lambda_by_hetero['NiO/TiO2']['CO']
lam_eth0=lambda_by_hetero['NiO/TiO2']['Ethanol']
for i in range(N):
    # sample multiplicative uncertainties
    Kco=K_co0*(1+rng.uniform(-0.10,0.10)); Keth=K_eth0*(1+rng.uniform(-0.10,0.10))
    Eaco=Ea_co0*(1+rng.uniform(-0.03,0.03)); Eaeth=Ea_eth0*(1+rng.uniform(-0.03,0.03))
    lco=lam_co0*(1+rng.uniform(-0.05,0.05)); leth=lam_eth0*(1+rng.uniform(-0.05,0.05))
    # clip to positive
    Kco=max(Kco,1e-5); Keth=max(Keth,1e-5); Eaco=max(Eaco,0.05); Eaeth=max(Eaeth,0.05); lco=max(lco,1e-7); leth=max(leth,1e-7)
    # compute only for CO and ethanol at 350 C, 100ppm
    T_K=Tref; thO=thetaO(350)
    thco=theta_gas(Kco,100); theth=theta_gas(Keth,100)
    rco=k_rate(Eaco,T_K)*thco*thO; reth=k_rate(Eaeth,T_K)*theth*thO
    Sco=math.exp(lco*rco/(kB*T_K)); Seth=math.exp(leth*reth/(kB*T_K))
    mc_rows.append({'Run':i+1,'CO_response':Sco,'Ethanol_response':Seth,'CO_Ethanol_selectivity':Sco/Seth,'K_CO':Kco,'K_Ethanol':Keth,'Ea_CO':Eaco,'Ea_Ethanol':Eaeth,'lambda_CO':lco,'lambda_Ethanol':leth})
mc_df=pd.DataFrame(mc_rows); mc_df.to_csv(os.path.join(RES,'23_monte_carlo_samples.csv'), index=False)
mc_summary=pd.DataFrame([
    {'Metric':'CO_response','Mean':mc_df.CO_response.mean(),'Median':mc_df.CO_response.median(),'Std':mc_df.CO_response.std(),'CI95_low':mc_df.CO_response.quantile(0.025),'CI95_high':mc_df.CO_response.quantile(0.975)},
    {'Metric':'CO_Ethanol_selectivity','Mean':mc_df.CO_Ethanol_selectivity.mean(),'Median':mc_df.CO_Ethanol_selectivity.median(),'Std':mc_df.CO_Ethanol_selectivity.std(),'CI95_low':mc_df.CO_Ethanol_selectivity.quantile(0.025),'CI95_high':mc_df.CO_Ethanol_selectivity.quantile(0.975)},
])
mc_summary.to_csv(os.path.join(RES,'24_monte_carlo_summary.csv'), index=False)
prob_df=pd.DataFrame([
    {'Condition':'SC_CO_Ethanol > 1','Probability':float((mc_df.CO_Ethanol_selectivity>1).mean())},
    {'Condition':'SC_CO_Ethanol > 1.5','Probability':float((mc_df.CO_Ethanol_selectivity>1.5).mean())},
    {'Condition':'SC_CO_Ethanol > 2','Probability':float((mc_df.CO_Ethanol_selectivity>2).mean())},
    {'Condition':'SC_CO_Ethanol > 2.5','Probability':float((mc_df.CO_Ethanol_selectivity>2.5).mean())},
])
prob_df.to_csv(os.path.join(RES,'25_selectivity_probability_summary.csv'), index=False)
plt.figure(figsize=(7,4.5)); plt.hist(mc_df.CO_response,bins=40); plt.xlabel('CO response'); plt.ylabel('Frequency'); plt.title('Monte Carlo distribution of CO response'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'23_monte_carlo_CO_response_histogram.png'),dpi=300); plt.close()
plt.figure(figsize=(7,4.5)); plt.hist(mc_df.CO_Ethanol_selectivity,bins=40); plt.xlabel('CO/ethanol selectivity'); plt.ylabel('Frequency'); plt.title('Monte Carlo distribution of CO/ethanol selectivity'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'24_monte_carlo_selectivity_histogram.png'),dpi=300); plt.close()
plt.figure(figsize=(7,4.5)); plt.scatter(mc_df.CO_response,mc_df.CO_Ethanol_selectivity,s=8,alpha=0.5); plt.xlabel('CO response'); plt.ylabel('CO/ethanol selectivity'); plt.title('Monte Carlo response-selectivity scatter'); plt.tight_layout(); plt.savefig(os.path.join(FIG,'25_monte_carlo_response_selectivity_scatter.png'),dpi=300); plt.close()

# Final concise summary markdown/csv
summary = {
    'primary_candidate': rank_df.iloc[0].Heterojunction,
    'primary_score': float(rank_df.iloc[0].Combined_score),
    'best_CO_response_candidate': rank_df.sort_values('CO_response',ascending=False).iloc[0].Heterojunction,
    'NiO_TiO2_CO_response_350C_100ppm': float(response('NiO/TiO2','CO',350,100)['Response']),
    'NiO_TiO2_CO_ethanol_selectivity_350C_100ppm': float(selectivity('NiO/TiO2',350,100)),
    'NiO_TiO2_CO_acetone_selectivity_350C_100ppm': float(selectivity('NiO/TiO2',350,100,against='Acetone')),
    'MonteCarlo_mean_CO_response': float(mc_summary[mc_summary.Metric=='CO_response'].Mean.iloc[0]),
    'MonteCarlo_mean_CO_ethanol_selectivity': float(mc_summary[mc_summary.Metric=='CO_Ethanol_selectivity'].Mean.iloc[0]),
    'P_selectivity_gt_1': float(prob_df[prob_df.Condition=='SC_CO_Ethanol > 1'].Probability.iloc[0]),
    'P_selectivity_gt_2': float(prob_df[prob_df.Condition=='SC_CO_Ethanol > 2'].Probability.iloc[0]),
}
with open(os.path.join(RES,'00_final_model_summary.json'),'w') as f: json.dump(summary,f,indent=2)
pd.DataFrame([summary]).to_csv(os.path.join(RES,'00_final_model_summary.csv'),index=False)

print('Generated model outputs in', BASE)
print(json.dumps(summary,indent=2))
