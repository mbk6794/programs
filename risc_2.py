import os

import numpy as np
import pandas as pd
from scipy.constants import pi, k, hbar, c, Planck
import get_info
import argparse
import matplotlib.pyplot as plt
import eckart

bohr_to_ang = 0.529177249 # [Angstrom / Bohr radius]
bohr_to_meter = bohr_to_ang * pow(10,-10)
hartree_to_wavenumber = 219474.63 # [cm**-1 / Hartrees]
amu_to_kg = 1.660538921*pow(10,-27) # kg
amu_to_m_e = 1822.89
kg_to_m_e = pow(9.1093837015*pow(10,-31),-1)
joule_to_wavenum = 5.034*pow(10,22)
#k = k * 1/kg_to_m_e * 1/np.power(bohr_to_meter,2) # Boltzmann constant, mass of electron * bohr^2 * s^-2
#hbar = hbar * 1/np.power(bohr_to_meter,2) * 1/kg_to_m_e # hbar, bohr^2 * mass of electron * s^-1
#Planck = Planck * 1/np.power(bohr_to_meter,2) * 1/kg_to_m_e # Planck constant, bohr^2 * mass of electron * s^-1

def get_ener(fin):
    energy = get_info.get_energy(fin)
    return energy

def get_soc(fin): # It requires the name of spin-orbit coupling csv file
    with open(fin,'r') as f:
        flines = f.readlines()
    data = []
    for i in range(2):
        data.append(flines[2*i+1].split(','))

    soc = np.zeros((2,3),dtype=complex)
    for i in range(len(data)):
        for j in range(len(data[i])):
            soc[i,j] = eval(data[i][j])

    return soc 

def amplitude(freq):
    freq = freq * 100 * c * 2*pi # 2pi/s
    E_vib = 0.5 * hbar * freq # 2pi*J 
    sq_amplitude = []
    for i in range(len(freq)):
        sq_amplitude.append(E_vib[i] * 2 / np.power(freq[i], 2) * pow(1/bohr_to_meter,2) * kg_to_m_e) # 2pi*kg*m^2/(2pi)^2 * (bohr/meter)^2 * (m_e/kg) = 1/2pi * bohr^2 * m_e
    sq_amplitude = np.array(sq_amplitude)
    amp = np.sqrt(sq_amplitude) # m_e^0.5 * bohr

    return amp

def displacement(amplitude, normal_mode):
    disp = []
    for i in range(len(normal_mode)):
        disp.append(amplitude[i] * normal_mode[i])
    disp = np.array(disp)
    return disp

def duschinsky(mass,initial, final, i_cart, f_cart):#, f_amplitude, i_amplitude): # It needs mass-weighted normal modes (row) vectors (initial(T1) & final(S0))
    T = eckart.Eckart(mass,i_cart,f_cart) # Eckart rotation matrix
    M, num = mass_weighted(mass, 'mass') # mass of electron
    Li = mass_weighted(initial, 'normal', M, num, T) 
    Lf = mass_weighted(final, 'normal', M, num) # mass-weighted normal modes
    mass_i_cart = mass_weighted(i_cart, 'cart', M, num, T)
    mass_f_cart = mass_weighted(f_cart, 'cart', M, num) # mass-weighted Cartesian coordinate, root(M)*|r>, root(a.u.)*Bohr
    dist_cart = (mass_f_cart - mass_i_cart).flatten() # the difference between mass-weighted Cartesian coordiantes.
    S = np.matmul(Li,np.transpose(Lf)) # Duschinsky rotation matrix 
    d = np.matmul(Li,dist_cart)
    #print("Li\n",Li)
    #print("S0-T1 mw Cartesian\n",dist_cart)
    #print("mw cartesian norm\n",np.linalg.norm(dist_cart))
    #print("Displacement\n",d)
    d = d.reshape((len(d),1)) # m_e^0.5 * Ang
    '''
    #i_disp = displacement(i_amplitude, Li)
    #i_disp = i_disp.reshape((len(Li),-1,3))
    #f_disp = displacement(f_amplitude, Lf)
    #print("f_amplitude\n",f_amplitude)
    #print("Lf\n",Lf)
    #print("f_disp\n",f_disp)
    #f_disp = f_disp.reshape((len(Lf),-1,3))
    
    Qf_2_temp = []
    for i in range(len(f_disp)):
        Qf_2_temp.append(np.matmul(M, f_disp[i]))
    Qf_2_temp = np.array(Qf_2_temp)
    flat_Qf_temp = []
    for i in range(len(Qf_2_temp)):
        flat_Qf_temp.append(Qf_2_temp[i].flatten())
    flat_Qf_temp = np.array(flat_Qf_temp)
    Qf = np.matmul(Lf, np.transpose(flat_Qf_temp))

    Qi_1 = np.matmul(S,Qf)+d

    Qi_2_temp = []
    for i in range(len(i_disp)):
        Qi_2_temp.append(np.matmul(M, i_disp[i]))
    Qi_2_temp = np.array(Qi_2_temp)
    flat_Qi_temp = []
    for i in range(len(Qi_2_temp)):
        flat_Qi_temp.append(Qi_2_temp[i].flatten())
    flat_Qi_temp = np.array(flat_Qi_temp)
    Qi_2 = np.matmul(Li, np.transpose(flat_Qi_temp))
    #print("Qf from transpose(L)(q-q0)\n",Qf)
    #print("Qi from transpose(L)(q-q0)\n",Qi_2)
    #print("Qi from SQf + d\n",Qi_1)
    '''
    return S, d, Lf, M

def mass_weighted(fin,option,mass=None, num=None, T=np.eye(3)):
    if option == 'mass':
        mass_matrix, num = get_info.get_mass(fin) # mass [amu]
        mass_matrix = mass_matrix * amu_to_m_e # mass of electron
        return mass_matrix, num
    elif option == 'normal':
        root_mass = np.sqrt(mass)
        root_mass_inv = np.sqrt(np.linalg.inv(mass))
        mass_unweighted_normal_mode = get_info.get_normal(fin) # [a.u.], QChem output, |n'>
        m_normal_mode = []
        for i in range(len(mass_unweighted_normal_mode)):
            for j in range(num):
                mass_unweighted_normal_mode[i,j] = np.matmul(T, mass_unweighted_normal_mode[i,j])
            m_normal_mode.append(np.matmul(mass, mass_unweighted_normal_mode[i])) # |Mn'>, a.u.
        m_normal_mode = np.array(m_normal_mode)

        flat_normal_mode = []
        for i in range(len(m_normal_mode)):
            flat_normal_mode.append(m_normal_mode[i].flatten())
        flat_normal_mode = np.array(flat_normal_mode)

        C = []
        for i in range(len(flat_normal_mode)):
            C.append(np.inner(flat_normal_mode[i], flat_normal_mode[i]))
        C = np.array(C) # a.u.

        ori_eig_normal = []
        for i in range(len(flat_normal_mode)):
            ori_eig_normal.append(1/np.sqrt(C[i])*flat_normal_mode[i]) # a.u.^-0.5
        ori_eig_normal = np.array(ori_eig_normal)

        mass_eig_normal = []
        for i in range(len(ori_eig_normal)):
            mass_eig_normal.append(np.matmul(root_mass_inv,ori_eig_normal[i].reshape((num,3))).flatten())
        mass_eig_normal = np.array(mass_eig_normal)

        flat_final_normal = []
        for i in range(len(mass_eig_normal)):
            flat_final_normal.append(mass_eig_normal[i]/np.linalg.norm(mass_eig_normal[i]))
        flat_final_normal = np.array(flat_final_normal) 

        return flat_final_normal
    elif option == 'cart':
        root_mass = np.sqrt(mass)
        cartesian = get_info.get_cart(fin,num) # cartesian geometry
        #cartesian = cartesian * 1/bohr_to_ang
        for i in range(num):
            cartesian[i] = np.matmul(T, cartesian[i])
        m_cart = np.matmul(root_mass, cartesian)

        return m_cart

def basic(ffreq,ifreq,S,d,tau):
    tau = tau * pow(10,-15)
    tau_prime = -tau-1j/(k*300)
    ffreq = ffreq * 100 * c * 2*pi # 2pi / s, angular frequency
    ifreq = ifreq * 100 * c * 2*pi
    bf, af = np.zeros((len(ffreq),len(ffreq)), dtype=np.complex_), np.zeros((len(ffreq),len(ffreq)), dtype=np.complex_)
    for i in range(len(ffreq)):
        bf[i,i] = ffreq[i]
        af[i,i] = ffreq[i]
    bi, ai = np.zeros((len(ifreq),len(ifreq)), dtype=np.complex_), np.zeros((len(ifreq),len(ifreq)), dtype=np.complex_)
    for i in range(len(ifreq)):
        bi[i,i] = ifreq[i]
        ai[i,i] = ifreq[i]
    for i in range(len(bf)):
        ai[i,i] = ai[i,i] / np.sin(hbar * ai[i,i] * tau)
        af[i,i] = af[i,i] / np.sin(hbar * af[i,i] * tau_prime)
        bi[i,i] = bi[i,i] / np.tan(hbar * bi[i,i] * tau)
        bf[i,i] = bf[i,i] / np.tan(hbar * bf[i,i] * tau_prime) # 2pi/s * 1/sin(or tan)(hbar * 2pi/s * s/hbar)

    E = bf - af 
    A = ai + np.matmul(np.matmul(np.transpose(S),af),S)
    B = bi + np.matmul(np.matmul(np.transpose(S),bf),S)
    
    F = np.array([np.matmul(np.matmul(np.transpose(d),E),S),np.matmul(np.matmul(np.transpose(d),E),S)])
    F = F.reshape((2*len(ffreq),1))
    return E, A, B, bf, bi, af, ai, F

def dso(af,ai,A,B,F,E,d,e_gap,t,dt):
    t = t * pow(10,-15)
    dt = dt * pow(10,-15)
    e_gap_term = np.exp(1j*e_gap*t)
    K = np.vstack((np.hstack((B,-A)), np.hstack((-A,B))))
    #### Way 1 ###
    _, K_vector = np.linalg.eig(K)
    cc_K_vector = np.transpose(K_vector.conjugate())
    diag_K = np.matmul(np.matmul(cc_K_vector,K),K_vector)
    in_sqrt = 1
    for i in range(len(af)):
        in_sqrt *= np.sqrt(af[i,i]*ai[i,i]/(diag_K[2*i,2*i]*diag_K[2*i+1,2*i+1]))
    ##############
    #### Way 2 ###
    #mul = np.matmul(np.matmul(ai,af), np.linalg.inv(np.matmul(B,(B-np.matmul(np.matmul(A,np.linalg.inv(B)),A)))))
    #in_sqrt = np.sqrt(np.linalg.det(mul))
    #############
    a = -1/2*np.matmul(np.matmul(np.transpose(F),np.linalg.inv(K)),F)
    b = np.matmul(np.matmul(np.transpose(d),E),d)
    exponent = -1j/hbar*(a+b)
    expo_func = 0
    for i in range(7):
        expo_func += np.power(exponent,i)/np.math.factorial(i)
    corr_fun_dso = in_sqrt*expo_func*e_gap_term*dt
    return corr_fun_dso

def H_matrix(bf, af, i, j, d, E, S):
    kth_num = len(bf)
    H_kl1 = np.zeros((len(bf),1), dtype=np.complex_)
    H_kl2 = np.zeros((len(af),1), dtype=np.complex_)

    dTES = np.matmul(np.matmul(np.transpose(d),E),S)

    H_kl1[i,0] = bf[i,i]*dTES[0,j]
    H_kl2[i,0] = -af[i,i]*dTES[0,j]
    H_kl = np.vstack((H_kl1,H_kl2))
 
    return H_kl

def G_matrix(bf,af,S,bi,ai,i,j):
    lenifreq = len(ai)
    lenffreq = len(bi)
    G11,G12 = np.zeros((lenifreq,lenifreq), dtype=np.complex_),np.zeros((lenifreq,lenifreq), dtype=np.complex_)
    G21,G22 = np.zeros((lenffreq,lenffreq), dtype=np.complex_),np.zeros((lenffreq,lenffreq), dtype=np.complex_)
    sas = np.matmul(np.matmul(np.transpose(S),ai),S)
    sbs = np.matmul(np.matmul(np.transpose(S),bi),S)

    G11[i] = -bf[i,i]*sas[j]
    G12[i] = bf[i,i]*sbs[j]
    G21[i] = af[i,i]*sas[j]
    G22[i] = -af[i,i]*sbs[j]
    G = np.vstack((np.hstack((G11,G12)),np.hstack((G21,G22))))
            
    return G

def nBO(deriv, nac_e_gap, M, Lf):
    nonadia_matrix = np.zeros(3*len(M)-6, dtype=np.complex_)
    Lf = Lf.reshape((-1,len(M),3))
    inv_M = np.linalg.inv(M)

    for k in range(len(nonadia_matrix)):
        nonadiabatic = 0
        for sigma in range(len(M)):
            for j in range(3):
                nonadiabatic += np.sqrt(inv_M)[sigma,sigma]*deriv[sigma,j]*Lf[k,sigma,j]
        nonadia_matrix[k] = -nonadiabatic * 1j * hbar / nac_e_gap

    return nonadia_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RISC rate constant')
    parser.add_argument('-soc','--soc',default=None,help='soc file name')
    parser.add_argument('-mass','--mass',default=None,help='mass file name')
    parser.add_argument('-f_v','--finalvib',default=None,help='normal mode of final state')
    parser.add_argument('-i_v','--initialvib',default=None,help='normal mode of initial state')
    parser.add_argument('-f_c','--finalcart',default=None,help='cartesian coordinates of final state')
    parser.add_argument('-i_c','--initialcart',default=None,help='cartesian coordinates of initial state')
    parser.add_argument('-nac','--nac',default=None,help='nac file name')
    parser.add_argument('-nd','--ndata',type=int,help='the number of data that will be used. It determines the number of packets')
    parser.add_argument('-idx','--index',type=int,help='index of packet')

    args = parser.parse_args()
    soc_file = args.soc
    mass_file = args.mass
    initial_vib_file = args.initialvib
    final_vib_file = args.finalvib
    initial_cart_file = args.initialcart
    final_cart_file = args.finalcart

    s0_energy = get_ener(final_vib_file)
    t1_energy = get_ener(initial_vib_file)
    e_gap = (t1_energy-s0_energy) # hartree
    if e_gap < 0:
        e_gap = -e_gap
    SOC = get_soc(soc_file) # cm^-1

    ffreq = get_info.get_freq(final_vib_file) # Frequecies of S0 state, cm^-1
    ifreq = get_info.get_freq(initial_vib_file) # Frequecies of T1 state
    #famp = amplitude(ffreq)
    #iamp = amplitude(ifreq)
    S, d, Lf, M = duschinsky(mass_file, initial_vib_file, final_vib_file, initial_cart_file, final_cart_file)#, famp, iamp)
    deriv, nac_e_gap = get_info.get_nac(args.nac, len(M)) # derivative coupling (inverse Bohr), energy gap (hartree)
    raw_sv_term = nBO(deriv, nac_e_gap, M, Lf) 
    t_matrix = np.zeros((3,len(raw_sv_term)), dtype=np.complex_)

    for m in range(len(SOC[1])):
        t_matrix[m] = SOC[1,m]*raw_sv_term

    T_matrix_0 = np.matmul(t_matrix[0].reshape((-1,1)),np.conjugate(np.transpose(t_matrix[0].reshape((-1,1)))))
    T_matrix_1 = np.matmul(t_matrix[1].reshape((-1,1)),np.conjugate(np.transpose(t_matrix[1].reshape((-1,1)))))
    T_matrix = 1/3 * (T_matrix_0 + 2*T_matrix_1).real

    E_zeropoint = 0
    total_mul = 1
    for i in range(len(ifreq)):
        E_zeropoint += 0.5 * hbar * ifreq[i]*100*c*2*pi
        prefactor = np.exp(-E_zeropoint/(k*300))
        in_denominator = -hbar * ifreq[i]*100*c*2*pi/(k*300)
        total_mul *= 1/(1-np.exp(in_denominator))
    Z = prefactor*total_mul
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(abs(S), interpolation='nearest', cmap=plt.get_cmap('Greys'))
    #fig.colorbar(cax)
    #plt.barh(range(len(d.reshape(-1))),abs(d.reshape(-1)))
    #plt.show()
    t_border_1, t_border_2 = -6553.6, 6553.6
    dt = 0.0125
    tau = np.array(np.arange(t_border_1,t_border_2,dt))/hbar
    tau = tau.reshape((-1,args.ndata))

    corr_fun_dso = 0
    spin_vib_term = 0
    cnt = 0
    for t in tau[args.index]:
        E, A, B, bf, bi, af, ai, F = basic(ffreq, ifreq, S, d, t)
        corr_fun_dso += dso(af, ai, A, B, F, E, d, e_gap, t, dt)
        K = np.vstack((np.hstack((B,-A)), np.hstack((-A,B))))
        K_inv = np.linalg.inv(K)
        for i in range(len(ffreq)):
            for j in range(len(ifreq)):
                H_kl = H_matrix(bf, af, i, j, d, E, S)
                G = G_matrix(bf, af, S, bi, ai, i, j)
                chi = 1j*hbar*np.trace(np.matmul(G,K_inv))+np.matmul(np.matmul(np.transpose(np.matmul(K_inv,F)),G),np.matmul(K_inv,F))-np.matmul(np.matmul(np.transpose(H_kl),K_inv),F)
                spin_vib_term += T_matrix[i,j]*chi

        cnt += 1
        if cnt % 100 == 0:
            print(cnt)

    Hso_sq_s0t1 = 0
    for j in range(len(SOC[0])):
        Hso_sq_s0t1 += 1/3 * np.power(abs(SOC[0,j]),2) # Hso^2

    kdso = 1/(hbar*Z)*corr_fun_dso*(Hso_sq_s0t1 + spin_vib_term)
    print(kdso)
    #with open("summary{:03d}.txt".format(args.index),'w') as f:
    #    f.write("spin-orbit coupling\n"+str(SOC)+"\n")
    #    f.write("hbar\n"+str(hbar)+"\n")
    #    f.write("Z inverse\n"+str(1/Z)+"\n")
    #    f.write("integrated term\n"+str(corr_fun_dso)+"\n")
    #    f.write("kdso\n"+str(kdso)+"\n")
