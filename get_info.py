import numpy as np

def get_freq(fin):
    freq_lines = []
    with open(fin,'r') as f:
        flines = f.readlines()
    for i in range(len(flines)):
        if 'Frequency' in flines[i]:
            freq_lines.append(flines[i])

    for i in range(len(freq_lines)):
        freq_lines[i] = freq_lines[i].split(' ')
        while '' in freq_lines[i]:
            freq_lines[i].remove('')

    freq = []
    for i in range(len(freq_lines)):
        for j in range(1,len(freq_lines[i])):
            freq.append(eval(freq_lines[i][j])) # [cm**-1]

    freq = np.array(freq)
    return freq

def get_normal(fin):
    starting_index = []
    end_index = []
    with open(fin,'r') as f:
        flines = f.readlines()
    for i in range(len(flines)):
        if 'Raman Active' in flines[i]:
            starting_index.append(i+2)
        if 'TransDip' in flines[i]:
            end_index.append(i)
    raw = []
    for i in range(len(starting_index)):
        raw.append(flines[starting_index[i]:end_index[i]])

    normal_mode = []
    for i in range(len(raw)):
        temp_normal_mode_1 = []
        temp_normal_mode_2 = []
        temp_normal_mode_3 = []
        for j in range(len(raw[i])):
            raw[i][j] = raw[i][j].split(' ')
            while '' in raw[i][j]:
                raw[i][j].remove('')
            tem = []
            for k in range(1,len(raw[i][j])):
                tem.append(eval(raw[i][j][k]))
            if len(tem) == 9:
                temp_normal_mode_3.append(tem[6:])
            if len(tem) >= 6:
                temp_normal_mode_2.append(tem[3:6])
            if len(tem) >= 3:
                temp_normal_mode_1.append(tem[:3])

        for j in [temp_normal_mode_1, temp_normal_mode_2, temp_normal_mode_3]:
            if len(j) >= 1:
                normal_mode.append(j) # mass unweighted normal mode

    normal_mode = np.array(normal_mode)    
    return normal_mode

def get_mass(fin):
    with open(fin, 'r') as f:
        flines = f.readlines()
    
    mass_weighted = np.zeros((len(flines), len(flines)))
    for i in range(len(flines)):
        flines[i] = flines[i].split(' ')
        while '' in flines[i]:
            flines[i].remove('')
        mass_weighted[i,i] = eval(flines[i][-1]) # diagonal matrix, [amu]
    
    atomnum = len(mass_weighted)
    return mass_weighted, atomnum

def get_nac(fin, num):
    with open(fin,'r') as f:
        flines = f.readlines()
    
    for i in range(len(flines)):
        if "Ej-Ei" in flines[i]:
            e_gap = i
        elif "CIS derivative coupling with ETF" in flines[i]:
            break
    
    e_g = flines[e_gap].split('=')
    e_g = eval(e_g[1]) # Hartrees
    data = flines[i+3:i+3+num]
    nacm = np.zeros((len(data),3)) 
    
    for i in range(len(data)):
        data[i] = data[i].split(' ')
        while '' in data[i]:
            data[i].remove('')
        nacm[i,0] = eval(data[i][1]) # [1 / Bohr radius] 
        nacm[i,1] = eval(data[i][2]) # [1 / Bohr radius]
        nacm[i,2] = eval(data[i][3]) # [1 / Bohr radius]
    
    deriv_coupling = nacm # [1 / Bohr radius ]

    return deriv_coupling, e_g

def get_cart(fin, num):
    geo = np.zeros((num,3))

    with open(fin,'r') as f:
        flines = f.readlines()

    for i in range(len(flines)):
        flines[i] = flines[i].split(' ')
        while '' in flines[i]:
            flines[i].remove('')
        for j in range(2,5):
            geo[i,j-2] = eval(flines[i][j])

    return geo

def get_energy(fin):
    with open(fin,'r') as f:
        flines = f.readlines()

    for i in range(len(flines)):
        if "SCF   energy in the final basis set" in flines[i]:
            break
    flines[i] = flines[i].split('=')
    energy = eval(flines[i][-1])

    return energy
