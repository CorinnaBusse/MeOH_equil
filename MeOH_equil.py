import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from thermo.chemical import Chemical

st.set_page_config(page_title="Gleichgewicht Methanolsynthese", layout="wide")
st.title("Gleichgewicht der Methanolsynthese")

# Seitenleiste: Auswahl der Reaktion
reaktionswahl = st.sidebar.selectbox("Wähle das Edukt:", ["CO", "CO₂"])

# Seitenleiste: Eingaben
P_ges_list = st.sidebar.multiselect(
    "Wähle Drücke (in atm):", [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500], default=[1, 10, 100]
)
T_min = st.sidebar.slider("Minimale Temperatur (K)", 298, 900, 300)
T_max = st.sidebar.slider("Maximale Temperatur (K)", 298, 900, 800)

# Reaktionsdefinitionen und stöchiometrische Koeffizienten
if reaktionswahl == "CO":
    species = ['carbon monoxide', 'H2', 'methanol']
    nu_i = np.array([-1, -2, 1])
    edukt_label = "CO"
else:
    species = ['carbon dioxide', 'H2', 'methanol', 'water']
    nu_i = np.array([-1, -3, 1, 1])
    edukt_label = "CO₂"

sum_nu = float(np.sum(nu_i))

# Chemikalien definieren
chemicals = [Chemical(s) for s in species]
H_f_i = np.array([chem.Hfm if hasattr(chem, 'Hfm') else chem.Hfgm for chem in chemicals])
G_f_i = np.array([chem.Gfm if hasattr(chem, 'Gfm') else chem.Gfgm for chem in chemicals])

# ΔH und ΔG berechnen
dH_r = nu_i @ H_f_i
dG_r = nu_i @ G_f_i
K_stp = np.exp(-1 / 8.314 * dG_r / 298)

# Gleichgewichtskonstante nach van’t Hoff
def K_vantHoff(T):
    return np.exp(np.log(K_stp) + dH_r / 8.314 * (1 / 298 - 1 / T))

# Temperaturbereich
T_array = np.linspace(T_min, T_max, 100)

# Gleichung für das Gleichgewicht
def Funk(xi, P_ges, T):
    if reaktionswahl == "CO":
        n_0 = np.array([1, 2, 0])
    else:
        n_0 = np.array([1, 3, 0, 0])

    n = n_0 + nu_i * xi
    n_total = np.sum(n)
    x = n / n_total

    if reaktionswahl == "CO":
        Q = x[2] / (x[0] * x[1]**2)
    else:
        Q = (x[2] * x[3]) / (x[0] * x[1]**3)

    return Q * P_ges**sum_nu - K_vantHoff(T)

# Plot
fig, ax = plt.subplots()

for P in P_ges_list:
    X_GGW_Result = []
    for T in T_array:
        try:
            xi = bisect(Funk, 1E-9, 1 - 1E-9, args=(P, T))
            X_GGW = xi / 1 * 100
        except:
            X_GGW = np.nan
        X_GGW_Result.append(X_GGW)

    ax.plot(T_array, X_GGW_Result, label=f'{P} atm')

ax.set_xlabel("Temperatur $T$ / K")
ax.set_ylabel(f"Umsatz $X_{{{edukt_label}}}$ / %")
ax.set_title(f"Gleichgewichtsumsatz von {edukt_label} in Abhängigkeit von der Temperatur")
ax.set_xlim(T_min, T_max)
ax.set_ylim(0, 100)
ax.grid(True)
ax.legend(title="Druck")

st.pyplot(fig)
