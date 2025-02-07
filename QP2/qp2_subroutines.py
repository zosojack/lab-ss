import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import chi2

import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

# per un tocco di colore
col = ['red', 'green', 'magenta', 'purple', 'peru', 'cyan', 
       'olive', 'goldenrod', 'black', 'sienna', 'steelblue', 'crimson']

# per leggere i file .asc con due colonne di QP2
def leggi_file_asc (nomefile):

    data = np.loadtxt(nomefile)
    
    # colonna 0: lunghezze d'onda
    # colonna 1: counts
    
    waveln = data[:, 0]
    counts = np.where(data[:, 1] < 0, 0, data[:, 1])
   
    return waveln, counts

# questo pre-processa i counts: 
# va applicato quando i picchi senza senso sono alle estremità del plot
# viene applicato rep volte, nel caso in cui ci siano più anomalie in ciascuna regione
def pre_clean_counts(counts, n_primi=20, n_ultimi=20, rep=1):
    
    #  definisce quale valore sostituirà il massimo
    n_2_primi = round(n_primi/5)
    n_2_ultimi = round(n_ultimi/5)
    
    j = 0
    while j < rep:
        # sostituisco il massimo nei primi 20 elementi di counts
        primi = counts[:n_primi]
        i_max_primi = np.argmax(primi)
        i_2_max_primi = np.argsort(primi)[-n_2_primi]
        counts[i_max_primi] = counts[i_2_max_primi]

        # sostituisco il massimo negli ultimi 20 elementi di counts
        ultimi = counts[-n_ultimi:]
        i_max_ultimi = np.argmax(ultimi)
        i_2_max_ultimi = np.argsort(ultimi)[-n_2_ultimi]
        counts[-n_ultimi + i_max_ultimi] = counts[-n_ultimi + i_2_max_ultimi]
        
        j += 1
        
    return counts

# pulisce il vettore dei conteggi controllando localmente i valori 
def clean_counts(counts, n=1):
    filtered_counts = np.copy(counts)
    
    for i in range(len(counts)):
        # mi concentro su una regione localizzata        
        left = max(0, i - 20)
        right = min(len(counts), i + 20)
        # calcolo media e deviazione standard locali
        local_mean = np.mean(counts[left:right])
        local_std = np.std(counts[left:right])
        # stabilisco i valori di soglia
        upp = local_mean + (n * local_std)
        low = local_mean - (n * local_std)
        
        # se il conteggio i-esimo è oltre i limiti, viene sostituito con la media locale
        if counts[i] > upp or counts[i] < low:
            filtered_counts[i] = local_mean
        # se invece è nullo, viene sostituito col precedente (se è il primo rimane 0)
        elif counts[i] == 0 and i < len(counts) - 1 and i!=0:
            filtered_counts[i] = counts[i + 1]
    
    return filtered_counts

# singola gaussiana
def gaussiana(x, a, mu, sigma):
        if sigma == 0:
            sigma = 0.1
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# somma di N gaussiane
def N_gaussiane (x, *params):
    # Inizializzo il risultato
    y = np.zeros_like(x)
    # Sommo una gaussiana per ciascuna entrata della list
    for i in range(0, len(params), 3):  # Scandisco i parametri a gruppi di 3
        A, mu, sigma = params[i], params[i+1], params[i+2]
        y += gaussiana(x, A, mu, sigma)
        
    return y

## TUTTO BEN SPEZZETTATO ##
##   fit di N gaussiane  ##

# Funzione per il calcolo del chi2 ridotto      
def chi2_N_gaussiane (x_fit, y_fit, popt, n_acq=5):
    ## CHI 2 RID ##
    y_fit_values = N_gaussiane(x_fit, *popt)
    
    # rimuovo elementi nulli
    mask = (y_fit != 0) & (y_fit_values != 0)  # Maschera per tenere solo elementi non nulli
    y_fit = y_fit[mask]
    y_fit_values = y_fit_values[mask]
    
    # calcolo residui
    residuals = y_fit - y_fit_values
    # errore poissoniano (radice del valore) / 160 perché è una media di 160 valori!
    sigma = np.sqrt(y_fit) / np.sqrt(40*n_acq)  
        
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(y_fit) - len(popt)
    
    # restituisce anche l'indice del massimo residuo
    # servirà per impostare la media della prossima gaussiana
    argmax = np.argmax(residuals)
    
    return chi2, dof, argmax

# Funzione per eseguire il fit
def esegui_fit (x_fit, y_fit, p0, bounds):
    # esecuzione fit
    try:
        # Fit gaussiano
        popt, pcov = curve_fit(N_gaussiane, x_fit, y_fit, p0=p0, bounds=bounds)
        return True, popt, pcov
    except:
        print(f"Fit non riuscito: ritento")
        try:
            # provo a toccacciare i parametri
            N = int(len(p0)/3) # - numero di gaussiane
            p0[3*N+1] = p0[3*N+1] + 2*p0[3*N+2] # sposto di poco la media 
            # Fit gaussiano
            popt, pcov = curve_fit(N_gaussiane, x_fit, y_fit, p0=p0, bounds=bounds)
            return True, popt, pcov
        except:
            print(f"Fit non riuscito nuovamente!")
            # restituisco i parametri del fit riuscito precedente
            return False, p0, [np.zeros_like(p0),np.zeros_like(p0)]
    
# Funzione per il fit di N gaussiane
def fit_N_gaussiane (x_fit, y_fit, params, bounds, N_MAX_GAUSS=5, n_acq=5):
    '''
    -----------------------------------------------------------------------
    Funzione che fitta uno spettro usando N gaussiane basandosi sul chi2rid
    -----------------------------------------------------------------------
    • x_fit: vettore delle lunghezze d'onda
    • y_fit: vettore dei conteggi
    • params: parametri iniziali per la/le prima/e gaussiana/e [(a1,mu1,sigma1), ...]
    • bounds: limiti per i parametri [(low_a1, low_mu1, low_sigma1), ...]
    • N_MAX_GAUSS: numero massimo di gaussiane usate
    • n_acq: numero di acquisizioni in laboratorio
    
    COMMENTO su n_acq:
    
    - ciascuna acquisizione è composta di 40 misure lunghe t_acq
    - il risultato della singola acquisizione è la media di queste 40 misure
    - il count è la media delle medie delle singole acquisizioni
    - il numero totale di misure è 40*n_acq
    
    l'errore poissoniano viene corretto in funzione di questo dettaglio:
    err = sqrt(count) / sqrt(40*n_acq)
    
    '''
    par_flattened = np.array(params).flatten()
    
    success, popt, pcov = esegui_fit(x_fit, y_fit, par_flattened, bounds)
    if not success:
        raise Exception('PRIMO fit non riuscito')
    
    chi2, dof, argmax = chi2_N_gaussiane (x_fit, y_fit, popt)
    chi2_rid = chi2/dof
    
    ## AGGIUNTA DI GAUSSIANE IN BASE AL CHI2RID ##
    k=1
    while chi2_rid > 1.2 and k < N_MAX_GAUSS: # gaussiane massime di default: 5
        
        # inizializzo nuovi parametri con media mu+2sigma del primo picco 
        # li attacco ai valori restituiti dal fit precedente
        p0 = np.concatenate([popt, np.array([1, x_fit[argmax], 50])])
        # sistemo i bounds
        low = [bounds[0][0], bounds[0][1], bounds[0][2]] * round(np.size(p0)/3)
        upp = [bounds[1][0], bounds[1][1], bounds[1][2]] * round(np.size(p0)/3)
        
        k+=1
    
        # ri-esecuzione fit
        success, popt, pcov = esegui_fit(x_fit, y_fit, p0=p0, bounds=(low,upp))
        
        if not success:
            print('Aggiunta di gaussiane interrotta')
            k=N_MAX_GAUSS+1 # analogo a break
        else:
            # ri-calcolo chi2 ridotto
            chi2, dof, argmax = chi2_N_gaussiane (x_fit, y_fit, popt)
            chi2_rid = chi2/dof
            
    return popt, pcov, chi2_rid