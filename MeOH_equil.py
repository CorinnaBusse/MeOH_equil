import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from thermo.chemical import Chemical

# Set page config
st.set_page_config(page_title="Methanol Equilibrium", layout="wide")

st.title("Methanol Synthesis Equilibrium: CO + 2H₂ ⇌ CH₃OH")

# Sidebar controls
st.sidebar.header("Eingabe Parameter")

P_ges_list = st.sidebar.multiselect(
    "Wähle Drücke in atm:", [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500], default=[1, 10, 100]
)

T_min = st.sidebar.slider("Min Temperatur (K)", 298, 873, 300)
T_max = st.sidebar.slider("Max Temperatur (K)", 298, 873, 800)

K_model = st.sidebar.radio("Equilibrium Constant Model", ["van't Hoff", "Ullmann"], index=0)

# Chemical definitions
CO = Chemical('carbon monoxide')
H2 = Chemical('H2')
CH3OH = Chemical('methanol')

H_f_i = np.array([CO.Hfm, H2.Hfm, CH3OH.Hfgm])
G_f_i = np.array([CO.Gfm, H2.Gfm, CH3OH.Gfgm])

nu_i = np.array([-1, -2, 1])
sum_nu = float(np.sum(nu_i))

dH_r = nu_i @ H_f_i
dG_r = nu_i @ G_f_i

K_stp = np.exp(-1 / 8.314 * dG_r / 298)

def K_vantHoff(T):
    return np.exp(np.log(K_stp) + dH_r / 8.314 * (1 / 298 - 1 / T))

def K_Ullmann(T):
    return 10 ** (3921 / T - 7.971 * np.log10(T) + 2.499 / 1000 * T - 2.953E-7 * T**2 + 10.2)

def Funk(xi, P_ges, T, K_func):
    n_CO = 1 + nu_i[0] * xi
    n_H2 = 2 + nu_i[1] * xi
    n_CH3OH = 0 + nu_i[2] * xi

    n_total = n_CO + n_H2 + n_CH3OH
    x_CO = n_CO / n_total
    x_H2 = n_H2 / n_total
    x_CH3OH = n_CH3OH / n_total

    return x_CH3OH / x_CO / (x_H2 ** 2) * P_ges**sum_nu - K_func(T)

T_array = np.linspace(T_min, T_max, 100)
fig, ax = plt.subplots()

for P in P_ges_list:
    X_GGW_Result = []
    for T in T_array:
        try:
            xi = bisect(Funk, 1E-9, 1 - 1E-9, args=(P, T, K_vantHoff if K_model == "van't Hoff" else K_Ullmann))
            X_GGW = xi / 1 * 100
        except:
            X_GGW = np.nan
        X_GGW_Result.append(X_GGW)

    ax.plot(T_array, X_GGW_Result, label=f'{P} / atm')

ax.set_xlabel("Temperatur $T$ / K")
ax.set_ylabel("Conversion $X_{CO}$ / %")
ax.set_title("Gleichgewichtsumsatzgrad vs. Temperatur")
ax.set_xlim(T_min, T_max)
ax.set_ylim(0, 100)
ax.grid(True)
ax.legend()

st.pyplot(fig)
