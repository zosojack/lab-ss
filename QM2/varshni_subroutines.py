import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import chi2

# conversione lambda (nm) in E (eV)
def lambda_to_E (wavelenght):
    h_in_ev = 4.1357e-15
    c_luce  = 299792458
    
    if wavelenght != 0:
        return h_in_ev * c_luce / (wavelenght*1e-09)
    else:
        return 0

# propagazione errore lambda (nm) in E (eV)
def err_lambda_to_E (wavelenght, err_wavelenght):
    h_in_ev = 4.1357e-15
    c_luce  = 299792458
    
    if wavelenght != 0:
        return h_in_ev * c_luce * err_wavelenght / (wavelenght*wavelenght*1e-09)
    else:
        return 0

# legge varshni
def varshni (T, E_0, alpha, beta):
    T = np.asarray(T)  
    return E_0 - (alpha * T*T) / (beta + T)

# doppia gaussiana + polinomio quadratico
def doppia_gaussiana_polinomio(x, A1, mu1, sigma1, A2, mu2, sigma2, a, b, c):
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    rumore = a * x**2 + b * x + c
    return gauss1 + gauss2 + rumore

# gaussiana semplice
'''
def gaussiana(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
'''

# costruisce automaticamente il nome del file
def build_name (campione, d_o_emi, d_o_ass, temperatura):
    name = '../data/rampa_T/' + campione + d_o_emi + 'OD_' + d_o_ass + 'OD_orangeF_' + str(temperatura) + 'K.txt'
    return name

# legge il file, estrapola ascissa(wavelength) e ordinata (counts)
def leggi_file (nomefile):
    # Leggi il file ignorando righe iniziali e usando il separatore corretto
    file = pd.read_csv(nomefile, skiprows=6, delimiter=';', engine='python', index_col=False)

    # Rimuovi spazi extra nei nomi delle colonne
    file.columns = file.columns.str.strip()

    # Ignora l'ultima colonna se presente
    if file.shape[1] > 4:
        file = file.iloc[:, :-1]

    # Converti le colonne in valori numerici
    file = file.apply(pd.to_numeric, errors='coerce')

    # Verifica le colonne disponibili
    #print(file.head())

    # Assicurati di avere lunghezza d'onda e conteggi
    wavelength = file.iloc[:, 0]  # Prima colonna
    sample_counts = file.iloc[:, 1]  # Seconda colonna

    return wavelength, sample_counts

# prendi i picchi: ce ne sono 2 nel reference e 4 nel campione 2
# è intelligente selezionare solo le regioni dello spettro che ci interessano:
# ~ 710/740 picco AlGaAs che scompare 
# ~ 775 picco qd che scompare 
# ~ 820 picco più alto
# ~ 830 picchetto a destra molto soppresso 

# Funzione per rilevare i picchi in due regioni
def find_peak_in_regions(asse_x, asse_y):
    regioni = [
        (690, 800, {'height': 15, 'distance': 10, 'prominence': 6}),
        (800, 880, {'height': 20, 'distance': 5, 'prominence': 5}),
    ]

    picchi_regioni = []

    for limite_inferiore, limite_superiore, parametri in regioni:
        maschera = (asse_x >= limite_inferiore) & (asse_x <= limite_superiore)
        x_region = asse_x[maschera]
        y_region = asse_y[maschera]

        # Trova i picchi nella regione corrente
        picchi, _ = find_peaks(y_region, **parametri)
        
        # Converti indici locali in indici globali rispetto a asse_x
        picchi_globali = np.where(maschera)[0][picchi]
        picchi_regioni.append(picchi_globali)

    return picchi_regioni
    

# ora ho i picchi, per ciascun picco bisogna fittare con gauss
# Funzione per il fit locale attorno a ciascun picco
def fit_gaussiani(asse_x, asse_y, picchi_regioni, emi, ass, temp):
    """
    Funzione per effettuare il fit delle regioni selezionate con gaussiane e plottare i risultati.
    Restituisce i risultati del fit per ciascuna regione.
    """

    # Definizione delle funzioni per il fit
    def gaussiana(x, a, mu, sigma):
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def somma_gaussiane(x, *params):
        n = len(params) // 3
        y = np.zeros_like(x)
        for i in range(n):
            a, mu, sigma = params[3 * i: 3 * (i + 1)]
            y += gaussiana(x, a, mu, sigma)
        return y

    # Dizionario per memorizzare i risultati
    risultati = {"Regione 1": [], "Regione 2": []}

    plt.figure(figsize=(5, 3))
    plt.plot(asse_x, asse_y, color='blue') 

    # Ciclo sulle regioni e sui picchi
    for i, picchi in enumerate(picchi_regioni):
        if len(picchi) == 0:
            print(f"Nessun picco nella regione {i + 1}, skip fit.")
            continue

        # Estrai i dati per la regione corrispondente
        limiti = (asse_x[picchi[0]] - 10, asse_x[picchi[-1]] + 10)  # Range della regione
        maschera = (asse_x >= limiti[0]) & (asse_x <= limiti[1])
        x_region = asse_x[maschera]
        y_region = asse_y[maschera]

        if len(x_region) == 0:
            print(f"Nessun dato nella regione {i + 1}, skip fit.")
            continue

        # Aggiungi i marker dei picchi
        if i == 0:
            plt.scatter(asse_x[picchi], asse_y[picchi], color='green', s=10, label=f"Picchi Regione {i + 1}")
        else:
            plt.scatter(asse_x[picchi], asse_y[picchi], color='red', s=10, label=f"Picchi Regione {i + 1}")

        # Parametri iniziali per il fit
        p0 = []
        bounds = ([], [])
        for picco in picchi:
            altezza = asse_y[picco]
            posizione = asse_x[picco]
            larghezza_iniziale = 5  # Stima iniziale della larghezza

            # Aggiungi parametri iniziali
            p0.extend([altezza, posizione, larghezza_iniziale])
            bounds[0].extend([0, posizione - 10, 0])  # Limiti inferiori
            bounds[1].extend([np.inf, posizione + 10, np.inf])  # Limiti superiori

        try:
            # Fit con somma di gaussiane
            popt, pcov = curve_fit(somma_gaussiane, x_region, y_region, p0=p0, bounds=bounds)

            # Plot della somma delle gaussiane
            x_fit = np.linspace(x_region.min(), x_region.max(), 500)
            y_fit = somma_gaussiane(x_fit, *popt)
            plt.plot(x_fit, y_fit, linestyle='--', color='orange', label=f"Somma Fit Regione {i + 1}")

            # Memorizza i parametri dei picchi nel dizionario
            region_name = f"Regione {i + 1}"
            for j in range(len(picchi)):
                mu = popt[3 * j + 1]  # Media
                sigma = popt[3 * j + 2]  # Deviazione standard

                # Estrai errore sulla media dalla matrice di covarianza
                errore_media = np.sqrt(np.diag(pcov))[3 * j + 1]

                risultati[region_name].append((mu, errore_media, sigma))

            # Debug: stampa i parametri del fit
            n_gaussiane = len(popt) // 3
            fit_type = "doppia gaussiana" if n_gaussiane > 1 else "singola gaussiana"
            print(f"Fit ({fit_type}) per Regione {i + 1}:")
            for j in range(n_gaussiane):
                print(f"  Gaussiana {j + 1}: a={popt[3 * j]:.2f}, mu={popt[3 * j + 1]:.2f}, sigma={popt[3 * j + 2]:.2f}")

        except RuntimeError:
            print(f"Fit non riuscito per la regione {i + 1}.")
        except ValueError as e:
            print(f"Errore nei dati per la regione {i + 1}: {e}")

    # Titolo e legenda
    plt.title(f"emi={emi} | ass={ass} | T={temp}")
    plt.xlabel("Lunghezza d'onda (nm)")
    plt.ylabel("Conteggi")
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

    return risultati


def print_results(raccoglitore_1):
    for ass_key, emi_data in raccoglitore_1.items():
        print(f"{ass_key}:")
        for emi_key, fit_results in emi_data.items():
            print(f"  {emi_key}:")
            for result in fit_results:
                result = [float(val) if isinstance(val, np.float64) else val for val in result]
                print(f"    Media: {result[0]}, Sigma: {result[1]}, Chi^2: {result[2]}, GDL: {result[3]}, Chi ridotto: {result[4]}, p-value: {result[5]}")
                
       
def media_pesata (values, errors):
    num, den = 0, 0
    for val, err in zip(values, errors):
        w = 1/(err*err)
        num += val * w
        den += w
    
    return num/den

