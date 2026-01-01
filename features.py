from pymatgen.core import Composition, Element
from mendeleev import element
import numpy as np
import json


with open("structures_data.txt", "r") as f:
    data = json.load(f)
    
    
##Setting up features with elemental properties
#Atomic Mass
#Atomic Radius
#Mendeleev Number
#Electronegativity
#Ionization Energy

def features(formula):
    #Set up objects and get properties for all elements in each element
    features = []
    mn = []
    mass = []
    ar = []
    ie = []
    en = []
    frac = []
    comp = Composition(formula)
    d = comp.fractional_composition.as_dict()
    for i,f in d.items():
        elm = Element(i)
        el = element(i)
        mn.append(el.mendeleev_number)
        mass.append(elm.atomic_mass)
        ar.append(elm.atomic_radius)
        ie.append(elm.ionization_energy)
        en.append(elm.X)
        frac.append(f)
        
    
    mend = np.array(mn)
    masses = np.array(mass)
    rad = np.array(ar)
    ion_e = np.array(ie)
    electro = np.array(en)
    fracs = np.array(frac, dtype = float)

    def stats(array):
        #Weighted mean Variance and Range for 5 properties
        mean = np.sum(fracs*array)
        var = np.sum(fracs *(array - mean)**2)
        rng = np.max(array) - np.min(array)
        ls = [mean,var,rng]
        return ls

    features += stats(mend)
    features += stats(masses)
    features += stats(rad)
    features += stats(ion_e)
    features += stats(electro)
    
    #Add sodium fraction and number of elements in Material
    na = comp.get_atomic_fraction("Na")
    num = len(comp.elements)

    features.append(na)
    features.append(num)

    return [float(i) for i in features]

import csv

with open("features.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for m, n in enumerate(data):
        i = n["formula"]
        e = n["formation_energy"]
        row = [i] + features(i) + [e]
        writer.writerow(row)

        if m % 50 == 0:
            d = m / 50
            print(f"{10 * d} % done")
        