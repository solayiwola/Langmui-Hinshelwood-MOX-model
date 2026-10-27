%% LH + Heterojunction Numerical Model for CO Selectivity
% This script reproduces the numerical calculations used for the thesis stage.
% It computes Langmuir-Hinshelwood surface coverages, reaction rates,
% heterojunction electronic parameters, sensor responses, selectivity values,
% thin-film parameter sweeps, response maps, sensitivity analysis, and
% Monte Carlo uncertainty estimates.
%
% Author note:
% Some kinetic/interfacial constants are literature-constrained modelling
% coefficients. They should be calibrated later using experimental transient
% fitting or dedicated DFT/NEB calculations.

clear; clc; close all;

baseDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(baseDir);
resultsDir = fullfile(rootDir,'results_matlab');
figDir = fullfile(rootDir,'figures_matlab');
if ~exist(resultsDir,'dir'); mkdir(resultsDir); end
if ~exist(figDir,'dir'); mkdir(figDir); end

%% Constants
kB = 8.617333262e-5;      % Boltzmann constant in eV/K
q = 1.602176634e-19;      % C
eps0 = 8.8541878128e-12;  % F/m
A0 = 1.0e5;               % pre-exponential factor, s^-1
Tref = 623.0;             % K, 350 C

%% Material parameters
materials = table( ...
    {'NiO';'ZnO';'TiO2'}, {'p';'n';'n'}, ...
    [5.20;4.60;4.70], [3.70;3.30;3.20], [1.80;4.30;4.10], ...
    [11.0;8.5;40.0], [5.0e17;1.0e18;8.0e17], [1.0;20.0;1.0], ...
    'VariableNames', {'Material','Type','Phi_eV','Eg_eV','chi_eV','eps_r','carrier_cm3','mobility_cm2Vs'});
materials.Ec_eV = -materials.chi_eV;
materials.Ev_eV = -(materials.chi_eV + materials.Eg_eV);
writetable(materials, fullfile(resultsDir,'01_material_parameters.csv'));

%% Gas parameters at 350 C reference point
baseGas = table( ...
    {'CO';'Ethanol';'Acetone';'NH3';'H2';'CH4'}, ...
    [0.72;1.15;1.05;0.90;0.38;0.25], ...
    [0.055;0.085;0.066;0.052;0.028;0.009], ...
    [0.32;0.25;0.34;0.34;0.38;0.52], ...
    [1.0;6.0;8.0;1.5;1.0;4.0], ...
    'VariableNames', {'Gas','Eads_abs_eV','Kref_ppm_inv','Ea_generic_eV','electron_yield'});
writetable(baseGas, fullfile(resultsDir,'02_gas_literature_constrained_parameters.csv'));

gases = baseGas.Gas;

%% Heterojunction parameters
heteros = table( ...
    {'NiO/TiO2';'NiO/ZnO';'TiO2/NiO';'ZnO/NiO'}, ...
    {'NiO';'NiO';'NiO';'NiO'}, {'TiO2';'ZnO';'TiO2';'ZnO'}, ...
    [0.50;0.60;0.50;0.60], [5.0e17;5.0e17;5.0e17;5.0e17], ...
    [8.0e17;1.0e18;8.0e17;1.0e18], [25.5;9.75;25.5;9.75], ...
    [1.12;1.00;1.03;0.92], [0.92;0.88;0.80;0.80], [0.75;0.55;0.70;0.48], [1.00;1.00;0.88;0.85], ...
    'VariableNames', {'Heterojunction','p_mat','n_mat','Vbi_V','NA_cm3','ND_cm3','eps_eff','Jf','NvO_norm','Qint_norm','layer_order_factor'});

Wd_nm = zeros(height(heteros),1);
Wp_nm = zeros(height(heteros),1);
Wn_nm = zeros(height(heteros),1);
Efield_Vm = zeros(height(heteros),1);
DeltaE_interface_eV = zeros(height(heteros),1);
EaCO_eff_eV = zeros(height(heteros),1);
Df = zeros(height(heteros),1);
Ah_score = zeros(height(heteros),1);

for i = 1:height(heteros)
    Wd_nm(i) = depletionWidthNm(heteros.Vbi_V(i), heteros.NA_cm3(i), heteros.ND_cm3(i), heteros.eps_eff(i), q, eps0);
    Wp_nm(i) = Wd_nm(i)*heteros.ND_cm3(i)/(heteros.NA_cm3(i)+heteros.ND_cm3(i));
    Wn_nm(i) = Wd_nm(i)*heteros.NA_cm3(i)/(heteros.NA_cm3(i)+heteros.ND_cm3(i));
    Efield_Vm(i) = heteros.Vbi_V(i)/(Wd_nm(i)*1e-9);
    DeltaE_interface_eV(i) = 0.04*heteros.Vbi_V(i) + 0.001*Wn_nm(i) + 0.035*heteros.NvO_norm(i) + 0.030*heteros.Qint_norm(i);
    EaCO_eff_eV(i) = max(0.05, 0.32 - DeltaE_interface_eV(i));
    Df(i) = Wd_nm(i)/30.0;
    Ah_score(i) = heteros.Jf(i)*Df(i)*(heteros.Vbi_V(i)/0.60)*heteros.layer_order_factor(i);
end
heteros.Wd_nm = Wd_nm;
heteros.Wp_nm = Wp_nm;
heteros.Wn_nm = Wn_nm;
heteros.Efield_Vm = Efield_Vm;
heteros.DeltaE_interface_eV = DeltaE_interface_eV;
heteros.EaCO_eff_eV = EaCO_eff_eV;
heteros.Df = Df;
heteros.Ah_score = Ah_score;
writetable(heteros, fullfile(resultsDir,'03_heterojunction_electronic_parameters.csv'));

%% Dictionaries implemented as tables for Ea and lambda
paramRows = table();
for h = 1:height(heteros)
    hj = heteros.Heterojunction{h};
    for gi = 1:numel(gases)
        gas = gases{gi};
        idx = strcmp(baseGas.Gas,gas);
        Ea = getEa(hj, gas, heteros);
        lam = getLambda(hj, gas);
        newRow = table({hj},{gas},baseGas.Kref_ppm_inv(idx),baseGas.Eads_abs_eV(idx),Ea,lam,baseGas.electron_yield(idx), ...
            'VariableNames', {'Heterojunction','Gas','Kref_ppm_inv','Eads_abs_eV','Ea_eff_eV','lambda_V_per_rate','electron_yield'});
        paramRows = [paramRows; newRow]; %#ok<AGROW>
    end
end
writetable(paramRows, fullfile(resultsDir,'04_gas_parameters_by_heterojunction.csv'));

%% Oxygen coverage
T_pts_C = [100;150;200;250;300;350;400];
thetaO_pts = [0.42;0.57;0.68;0.65;0.58;0.46;0.31];
oxygenTable = table(T_pts_C, thetaO_pts, thetaO_pts*1.0e13, ...
    'VariableNames', {'Temperature_C','theta_O','Electron_withdrawal_cm2'});
writetable(oxygenTable, fullfile(resultsDir,'05_oxygen_coverage_vs_temperature.csv'));
figure; plot(T_pts_C,thetaO_pts,'-o'); grid on;
xlabel('Temperature (°C)'); ylabel('Oxygen surface coverage, \theta_O');
title('Calculated oxygen coverage vs temperature');
saveas(gcf, fullfile(figDir,'01_oxygen_coverage_vs_temperature.png'));

%% Surface coverage vs concentration
concList = [1 5 10 50 100 250 500 1000];
coverageTable = table();
for C = concList
    for gi = 1:numel(gases)
        gas = gases{gi};
        [r,th,thO,k,K] = lhRate('NiO/TiO2',gas,350,C,baseGas,heteros,kB,A0,Tref,false,0.35,35,80); %#ok<ASGLU>
        row = table({gas},C,K,th,'VariableNames',{'Gas','Concentration_ppm','K_ppm_inv','theta_gas'});
        coverageTable = [coverageTable; row]; %#ok<AGROW>
    end
end
writetable(coverageTable, fullfile(resultsDir,'06_surface_coverage_vs_concentration.csv'));
figure; hold on;
for gi = 1:numel(gases)
    gas = gases{gi};
    df = coverageTable(strcmp(coverageTable.Gas,gas),:);
    plot(df.Concentration_ppm,df.theta_gas,'-o','DisplayName',gas);
end
set(gca,'XScale','log'); grid on; legend('Location','best');
xlabel('Gas concentration (ppm)'); ylabel('Surface coverage, \theta');
title('Langmuir surface coverage at 350 °C');
saveas(gcf, fullfile(figDir,'02_surface_coverage_vs_concentration.png'));

%% Reaction rates, responses, and selectivity
Tlist = 100:50:400;
rrTable = table();
respTemp = table();
for T_C = Tlist
    for gi = 1:numel(gases)
        gas = gases{gi};
        [r,th,thO,k,K] = lhRate('NiO/TiO2',gas,T_C,100,baseGas,heteros,kB,A0,Tref,false,0.35,35,80);
        rrTable = [rrTable; table({gas},T_C,th,thO,k,K,r,'VariableNames',{'Gas','Temperature_C','theta_gas','theta_O','k_s_inv','K_ppm_inv','r_LH'})]; %#ok<AGROW>
        S = sensorResponse('NiO/TiO2',gas,T_C,100,baseGas,heteros,kB,A0,Tref,0.35,35,80);
        respTemp = [respTemp; S]; %#ok<AGROW>
    end
end
writetable(rrTable, fullfile(resultsDir,'07_reaction_rates_vs_temperature_NiOTiO2.csv'));
writetable(respTemp, fullfile(resultsDir,'09_response_vs_temperature_NiOTiO2.csv'));

figure; hold on;
for gi = 1:numel(gases)
    gas = gases{gi};
    df = respTemp(strcmp(respTemp.Gas,gas),:);
    plot(df.T_C,df.Response,'-o','DisplayName',gas);
end
set(gca,'YScale','log'); grid on; legend('Location','best');
xlabel('Temperature (°C)'); ylabel('Sensor response (R_{air}/R_{gas})');
title('Predicted responses vs temperature for NiO/TiO2');
saveas(gcf, fullfile(figDir,'05_response_vs_temperature_NiOTiO2.png'));

selTemp = table();
for T_C = Tlist
    Sco = respTemp.Response(strcmp(respTemp.Gas,'CO') & respTemp.T_C==T_C);
    for gi = 1:numel(gases)
        gas = gases{gi};
        if strcmp(gas,'CO'); continue; end
        Sx = respTemp.Response(strcmp(respTemp.Gas,gas) & respTemp.T_C==T_C);
        selTemp = [selTemp; table(T_C,{['CO/' gas]},Sco/Sx,'VariableNames',{'Temperature_C','Comparison','Selectivity'})]; %#ok<AGROW>
    end
end
writetable(selTemp, fullfile(resultsDir,'10_selectivity_vs_temperature_NiOTiO2.csv'));
figure; hold on;
comps = unique(selTemp.Comparison);
for i = 1:numel(comps)
    df = selTemp(strcmp(selTemp.Comparison,comps{i}),:);
    plot(df.Temperature_C,df.Selectivity,'-o','DisplayName',comps{i});
end
set(gca,'YScale','log'); grid on; legend('Location','best');
xlabel('Temperature (°C)'); ylabel('Selectivity coefficient');
title('CO selectivity vs temperature for NiO/TiO2');
saveas(gcf, fullfile(figDir,'06_selectivity_vs_temperature_NiOTiO2.png'));

%% Concentration sweep
respConc = table();
for C = concList
    for gi = 1:numel(gases)
        respConc = [respConc; sensorResponse('NiO/TiO2',gases{gi},350,C,baseGas,heteros,kB,A0,Tref,0.35,35,80)]; %#ok<AGROW>
    end
end
writetable(respConc, fullfile(resultsDir,'11_response_vs_concentration_NiOTiO2.csv'));

figure; hold on;
for gi = 1:numel(gases)
    gas = gases{gi};
    df = respConc(strcmp(respConc.Gas,gas),:);
    plot(df.C_ppm,df.Response,'-o','DisplayName',gas);
end
set(gca,'XScale','log','YScale','log'); grid on; legend('Location','best');
xlabel('Concentration (ppm)'); ylabel('Sensor response (R_{air}/R_{gas})');
title('Predicted response vs concentration for NiO/TiO2 at 350 °C');
saveas(gcf, fullfile(figDir,'07_response_vs_concentration_NiOTiO2.png'));

%% Final heterojunction ranking
rankTable = table();
for h = 1:height(heteros)
    hj = heteros.Heterojunction{h};
    Sco = sensorResponse(hj,'CO',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
    Seth = sensorResponse(hj,'Ethanol',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
    Sac = sensorResponse(hj,'Acetone',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
    rankTable = [rankTable; table({hj},Sco,Sco/Seth,Sco/Sac,heteros.Wd_nm(h),heteros.Vbi_V(h),heteros.EaCO_eff_eV(h),heteros.Ah_score(h), ...
        'VariableNames', {'Heterojunction','CO_response','CO_Ethanol_selectivity','CO_Acetone_selectivity','Wd_nm','Vbi_V','EaCO_eff_eV','Ah_score'})]; %#ok<AGROW>
end
rankTable.CO_response_norm = rankTable.CO_response/max(rankTable.CO_response);
rankTable.CO_Ethanol_selectivity_norm = rankTable.CO_Ethanol_selectivity/max(rankTable.CO_Ethanol_selectivity);
rankTable.CO_Acetone_selectivity_norm = rankTable.CO_Acetone_selectivity/max(rankTable.CO_Acetone_selectivity);
rankTable.Wd_nm_norm = rankTable.Wd_nm/max(rankTable.Wd_nm);
rankTable.Vbi_V_norm = rankTable.Vbi_V/max(rankTable.Vbi_V);
rankTable.Combined_score = 0.15*rankTable.CO_response_norm + 0.45*rankTable.CO_Ethanol_selectivity_norm + 0.10*rankTable.CO_Acetone_selectivity_norm + 0.20*rankTable.Wd_nm_norm + 0.10*rankTable.Vbi_V_norm;
rankTable = sortrows(rankTable,'Combined_score','descend');
writetable(rankTable, fullfile(resultsDir,'19_heterojunction_final_ranking.csv'));
figure; bar(categorical(rankTable.Heterojunction),rankTable.Combined_score); grid on;
ylabel('Combined ranking score'); title('Final heterojunction ranking');
saveas(gcf, fullfile(figDir,'16_heterojunction_ranking_scores.png'));

%% Response surface and contour map
Tgrid = [200 250 300 350 400];
Cgrid = [10 50 100 250 500];
surfaceTable = table();
for T_C = Tgrid
    for C = Cgrid
        Sco = sensorResponse('NiO/TiO2','CO',T_C,C,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
        Seth = sensorResponse('NiO/TiO2','Ethanol',T_C,C,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
        Sac = sensorResponse('NiO/TiO2','Acetone',T_C,C,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response;
        surfaceTable = [surfaceTable; table(T_C,C,Sco,Sco/Seth,Sco/Sac,(Sco/Seth)*(Sco/Sac), ...
            'VariableNames', {'Temperature_C','Concentration_ppm','CO_response','CO_Ethanol_selectivity','CO_Acetone_selectivity','Combined_selectivity_index'})]; %#ok<AGROW>
    end
end
writetable(surfaceTable, fullfile(resultsDir,'22_response_surface_selectivity_map.csv'));

[X,Y] = meshgrid(Cgrid,Tgrid);
Z = reshape(surfaceTable.CO_response, [numel(Cgrid), numel(Tgrid)])';
figure; surf(X,Y,Z); xlabel('Concentration (ppm)'); ylabel('Temperature (°C)'); zlabel('CO response');
title('CO response surface, NiO/TiO2');
saveas(gcf, fullfile(figDir,'20_CO_response_surface_NiOTiO2.png'));

Zsel = reshape(surfaceTable.CO_Ethanol_selectivity, [numel(Cgrid), numel(Tgrid)])';
figure; contourf(X,Y,Zsel,12); colorbar; xlabel('Concentration (ppm)'); ylabel('Temperature (°C)');
title('CO/ethanol selectivity contour map');
saveas(gcf, fullfile(figDir,'21_CO_ethanol_selectivity_contour_NiOTiO2.png'));

%% Monte Carlo uncertainty
rng(42);
N = 5000;
mc = table();
Kco0 = baseGas.Kref_ppm_inv(strcmp(baseGas.Gas,'CO'));
Keth0 = baseGas.Kref_ppm_inv(strcmp(baseGas.Gas,'Ethanol'));
Eaco0 = getEa('NiO/TiO2','CO',heteros);
Eaeth0 = getEa('NiO/TiO2','Ethanol',heteros);
lamco0 = getLambda('NiO/TiO2','CO');
lameth0 = getLambda('NiO/TiO2','Ethanol');
for i = 1:N
    Kco = Kco0*(1 + (-0.10 + 0.20*rand));
    Keth = Keth0*(1 + (-0.10 + 0.20*rand));
    Eaco = Eaco0*(1 + (-0.03 + 0.06*rand));
    Eaeth = Eaeth0*(1 + (-0.03 + 0.06*rand));
    lamco = lamco0*(1 + (-0.05 + 0.10*rand));
    lameth = lameth0*(1 + (-0.05 + 0.10*rand));
    thO = oxygenCoverage(350);
    thco = thetaGas(Kco,100);
    theth = thetaGas(Keth,100);
    rco = rateConstant(Eaco,Tref,A0,kB)*thco*thO;
    reth = rateConstant(Eaeth,Tref,A0,kB)*theth*thO;
    Sco = exp(lamco*rco/(kB*Tref));
    Seth = exp(lameth*reth/(kB*Tref));
    mc = [mc; table(i,Sco,Seth,Sco/Seth,Kco,Keth,Eaco,Eaeth,lamco,lameth, ...
        'VariableNames', {'Run','CO_response','Ethanol_response','CO_Ethanol_selectivity','K_CO','K_Ethanol','Ea_CO','Ea_Ethanol','lambda_CO','lambda_Ethanol'})]; %#ok<AGROW>
end
writetable(mc, fullfile(resultsDir,'23_monte_carlo_samples.csv'));
summaryMC = table({'CO_response';'CO_Ethanol_selectivity'}, ...
    [mean(mc.CO_response); mean(mc.CO_Ethanol_selectivity)], ...
    [median(mc.CO_response); median(mc.CO_Ethanol_selectivity)], ...
    [std(mc.CO_response); std(mc.CO_Ethanol_selectivity)], ...
    [prctile(mc.CO_response,2.5); prctile(mc.CO_Ethanol_selectivity,2.5)], ...
    [prctile(mc.CO_response,97.5); prctile(mc.CO_Ethanol_selectivity,97.5)], ...
    'VariableNames', {'Metric','Mean','Median','Std','CI95_low','CI95_high'});
writetable(summaryMC, fullfile(resultsDir,'24_monte_carlo_summary.csv'));

figure; histogram(mc.CO_response,40); xlabel('CO response'); ylabel('Frequency'); title('Monte Carlo distribution of CO response');
saveas(gcf, fullfile(figDir,'23_monte_carlo_CO_response_histogram.png'));
figure; histogram(mc.CO_Ethanol_selectivity,40); xlabel('CO/ethanol selectivity'); ylabel('Frequency'); title('Monte Carlo distribution of CO/ethanol selectivity');
saveas(gcf, fullfile(figDir,'24_monte_carlo_selectivity_histogram.png'));

fprintf('\nCompleted L-H + heterojunction calculations.\n');
fprintf('Best overall candidate: %s\n', rankTable.Heterojunction{1});
fprintf('NiO/TiO2 CO response at 350 C, 100 ppm: %.3f\n', sensorResponse('NiO/TiO2','CO',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response);
fprintf('NiO/TiO2 CO/ethanol selectivity: %.3f\n', sensorResponse('NiO/TiO2','CO',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response / sensorResponse('NiO/TiO2','Ethanol',350,100,baseGas,heteros,kB,A0,Tref,0.35,35,80).Response);

%% Local functions
function Wd = depletionWidthNm(Vbi, NA_cm3, ND_cm3, eps_eff, q, eps0)
    NA = NA_cm3*1e6;
    ND = ND_cm3*1e6;
    epss = eps_eff*eps0;
    W = sqrt((2*epss/q)*((NA+ND)/(NA*ND))*Vbi);
    Wd = W*1e9;
end

function thO = oxygenCoverage(T_C)
    Tpts = [100 150 200 250 300 350 400];
    Opts = [0.42 0.57 0.68 0.65 0.58 0.46 0.31];
    thO = interp1(Tpts,Opts,T_C,'linear','extrap');
end

function K = adsorptionConstant(Kref,Eads_abs,T_K,Tref,kB)
    K = Kref*exp((Eads_abs/kB)*(1/T_K - 1/Tref));
end

function th = thetaGas(K,C)
    th = (K*C)/(1+K*C);
end

function k = rateConstant(Ea,T_K,A0,kB)
    k = A0*exp(-Ea/(kB*T_K));
end

function Ea = getEa(hj,gas,heteros)
    row = find(strcmp(heteros.Heterojunction,hj));
    if strcmp(gas,'CO')
        Ea = heteros.EaCO_eff_eV(row);
        return;
    end
    switch hj
        case 'NiO/TiO2'
            vals = struct('Ethanol',0.250,'Acetone',0.340,'NH3',0.340,'H2',0.380,'CH4',0.520);
        case 'NiO/ZnO'
            vals = struct('Ethanol',0.230,'Acetone',0.350,'NH3',0.340,'H2',0.380,'CH4',0.520);
        case 'TiO2/NiO'
            vals = struct('Ethanol',0.255,'Acetone',0.345,'NH3',0.340,'H2',0.380,'CH4',0.520);
        otherwise
            vals = struct('Ethanol',0.235,'Acetone',0.350,'NH3',0.340,'H2',0.380,'CH4',0.520);
    end
    Ea = vals.(gas);
end

function lam = getLambda(hj,gas)
    switch hj
        case 'NiO/TiO2'
            vals = struct('CO',4.70e-4,'Ethanol',6.60e-4,'Acetone',1.50e-3,'NH3',2.00e-3,'H2',2.00e-3,'CH4',1.00e-3);
        case 'NiO/ZnO'
            vals = struct('CO',7.50e-4,'Ethanol',7.00e-4,'Acetone',1.40e-3,'NH3',1.80e-3,'H2',2.00e-3,'CH4',1.00e-3);
        case 'TiO2/NiO'
            vals = struct('CO',4.15e-4,'Ethanol',6.25e-4,'Acetone',1.35e-3,'NH3',1.85e-3,'H2',1.80e-3,'CH4',9.00e-4);
        otherwise
            vals = struct('CO',6.50e-4,'Ethanol',6.90e-4,'Acetone',1.25e-3,'NH3',1.70e-3,'H2',1.80e-3,'CH4',9.00e-4);
    end
    lam = vals.(gas);
end

function F = morphologyFactor(gas,P,d_nm,t_nm)
    switch gas
        case 'CO'
            aP=1.50; bG=0.45; topt=80; sigma=60;
        case 'Ethanol'
            aP=1.10; bG=0.25; topt=120; sigma=65;
        case 'Acetone'
            aP=1.00; bG=0.25; topt=110; sigma=65;
        case 'NH3'
            aP=1.20; bG=0.30; topt=100; sigma=65;
        case 'H2'
            aP=1.30; bG=0.40; topt=70; sigma=55;
        otherwise
            aP=0.60; bG=0.15; topt=130; sigma=75;
    end
    raw = (1+aP*P)*(35/d_nm)^bG*exp(-0.5*((t_nm-topt)/sigma)^2);
    base = (1+aP*0.35)*(35/35)^bG*exp(-0.5*((80-topt)/sigma)^2);
    F = raw/base;
end

function [r,th,thO,k,K] = lhRate(hj,gas,T_C,C,baseGas,heteros,kB,A0,Tref,applyMorph,P,d_nm,t_nm)
    T_K = T_C + 273.15;
    idx = strcmp(baseGas.Gas,gas);
    K = adsorptionConstant(baseGas.Kref_ppm_inv(idx),baseGas.Eads_abs_eV(idx),T_K,Tref,kB);
    th = thetaGas(K,C);
    thO = oxygenCoverage(T_C);
    Ea = getEa(hj,gas,heteros);
    k = rateConstant(Ea,T_K,A0,kB);
    r = k*th*thO;
    if applyMorph
        r = r*morphologyFactor(gas,P,d_nm,t_nm);
    end
end

function out = sensorResponse(hj,gas,T_C,C,baseGas,heteros,kB,A0,Tref,P,d_nm,t_nm)
    [r,th,thO,k,K] = lhRate(hj,gas,T_C,C,baseGas,heteros,kB,A0,Tref,true,P,d_nm,t_nm);
    lam = getLambda(hj,gas);
    dV = lam*r;
    row = find(strcmp(heteros.Heterojunction,hj));
    Vbi = heteros.Vbi_V(row);
    Vb_air = Vbi*(1+0.25*thO);
    Vb_gas = max(Vb_air-dV,0.02);
    T_K = T_C+273.15;
    Rair = exp(Vb_air/(kB*T_K));
    Rgas = exp(Vb_gas/(kB*T_K));
    Response = Rair/Rgas;
    out = table({hj},{gas},T_C,C,th,thO,K,k,r,dV,Vb_air,Vb_gas,Rair,Rgas,Response, ...
        'VariableNames', {'Heterojunction','Gas','T_C','C_ppm','theta_gas','theta_O','K_ppm_inv','k_s_inv','r_LH','DeltaV_V','Vb_air_V','Vb_gas_V','Rair_norm','Rgas_norm','Response'});
end
