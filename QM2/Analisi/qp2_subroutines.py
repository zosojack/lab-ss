import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.stats import chi2

# per sistemare gli errori dei dati troppo smussati
import random
import time
random.seed(time.time())


import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

# per un tocco di colore
col = ['red', 'green', 'magenta', 'purple', 'peru', 'cyan', 'olive', 
       'goldenrod', 'black', 'sienna', 'steelblue', 'crimson', 'violet',
       'darkorange', 'rosybrown', 'cadetblue', 'navy', 'royalblue', 
       'slategrey', 'forestgreen', 'lightcoral', 'hotpink']

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
def clean_counts(counts, n=1, peak: bool = True):
    '''
    - counts: vettore dei counts nei bin
    - n: multiplo della deviazione standard da considerare
    - peak: se True pulisco anche il picco, 
            se False l'intorno del picco viene preservato 
    '''
    filtered_counts = np.copy(counts)
    
    
    
    # escludo degli indici se non pulisco il picco
    if peak is not True:
        i_max = np.argmax(counts)
        lista_indici = np.concatenate((np.arange(0, i_max - 20), np.arange(i_max + 20, len(counts))))
    # tutti gli indici se pulisco anche il picco
    else:
        lista_indici = np.arange(len(counts))
        
    for i in lista_indici:
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

        '''# così la curva viene oltremodo smussata:
        # va bene togliere punti inutili ma meglio re-inserire un po' di casualità
        caso = random.uniform(-0.1, 0.1) * (abs(counts[i] - local_mean) / local_std)
        filtered_counts[i] += caso*local_std
        # per evitare che sia nullo
        filtered_counts[i] = max(filtered_counts[i], local_mean * 0.9)
        # per evitare che sia enorme
        filtered_counts[i] = min(filtered_counts[i], local_mean * 1.1)'''
        
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

# somma di N gaussiane con traslazione verticale (per il bulk rumoroso)
def N_gaussiane_traslate(x, *params):
    """
    Somma N gaussiane e aggiunge una traslazione verticale,
    dove gli ultimi parametri in `params` sono:
      [A1, mu1, sigma1, A2, mu2, sigma2, ..., delta_y]
    """
    y = np.zeros_like(x)
    # Calcola il numero di gaussiane: i primi len(params)-1 parametri
    for i in range(0, len(params) - 1, 3):
        A, mu, sigma = params[i], params[i+1], params[i+2]
        y += gaussiana(x, A, mu, sigma)
    # Aggiungi il parametro di traslazione verticale (l'ultimo elemento)
    y += params[-1]
    return y

## TUTTO BEN SPEZZETTATO ##
##   fit di N gaussiane  ##

# Funzione per il calcolo del chi2 ridotto      
def chi2_N_gaussiane (x_fit, y_fit, popt, n_acq=5, err_counts=None, DELTA_Y=False):
    
    ## CHI 2 RID ##
    if DELTA_Y:
        y_fit_values = N_gaussiane_traslate(x_fit, *popt)
    else:
        y_fit_values = N_gaussiane(x_fit, *popt)
    
    # rimuovo elementi nulli
    mask = (y_fit > 0) & (y_fit_values != 0)  # Maschera per tenere solo elementi non nulli
    y_fit = y_fit[mask]
    y_fit_values = y_fit_values[mask]
    
    # calcolo residui
    residuals = y_fit - y_fit_values
   
    if err_counts is not None:
        sigma = err_counts[mask]
    else: 
        # errore poissoniano (radice del valore) / 160 perché è una media di 160 valori!
        sigma = np.sqrt(y_fit)

        if n_acq != 0:
            sigma = sigma / np.sqrt(40*n_acq)  
        
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(y_fit) - len(popt)
    
    # restituisce anche l'indice del massimo residuo
    # servirà per impostare la media della prossima gaussiana
    argmax = np.argmax(residuals)
    
    return chi2, dof, argmax

# Funzione per eseguire il fit
def esegui_fit (x_fit, y_fit, p0, bounds, DELTA_Y=False, err_counts=None):
    # esecuzione fit
    try:
        # Fit gaussiano
        if DELTA_Y:
            popt, pcov = curve_fit(N_gaussiane_traslate, x_fit, y_fit, sigma=err_counts, p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(N_gaussiane, x_fit, y_fit, sigma=err_counts, p0=p0, bounds=bounds)
        return True, popt, pcov
    except:
        print(f"Fit non riuscito: ritento")
        try:
            # provo a toccacciare i parametri
            N = int(len(p0)/3) # - numero di gaussiane
            p0[3*N+1] = p0[3*N+1] + 2*p0[3*N+2] # sposto di poco la media 
            # Fit gaussiano
            if DELTA_Y:
                popt, pcov = curve_fit(N_gaussiane_traslate, x_fit, y_fit, sigma=err_counts, p0=p0, bounds=bounds)
            else:
                popt, pcov = curve_fit(N_gaussiane, x_fit, y_fit, sigma=err_counts, p0=p0, bounds=bounds)
            return True, popt, pcov
        except:
            print(f"Fit non riuscito nuovamente!")
            # restituisco i parametri del fit riuscito precedente
            return False, p0, [np.zeros_like(p0),np.zeros_like(p0)]
        
# Per la prima T viene fornito un parametro, dopo di che ogni T successiva riceve
# i parametri della T precedente. 
# Bisogna evitare che si accumulino gaussiane inutilmente, introducendo un controllo 
# che escluda eventuali gaussiane superflue
def prevent_overfitting (x_fit, y_fit, p0, bounds, n_acq):

    # controllo il chi2rid non appena vengono passati i parametri
    chi2, dof, _ = chi2_N_gaussiane (x_fit, y_fit, p0, n_acq)
    chi2rid  = chi2 / dof
    # se è minore di 1 rimuovo un picco (ultimi tre valori) dai parametri
    contatore = 0
    while chi2rid < 2:
        p0 = p0[:-3]
        bounds = (bounds[0][:-3],bounds[1][:-3]) # anche i bounds vanno ridimensionati
        chi2, dof, _ = chi2_N_gaussiane (x_fit, y_fit, p0, n_acq)
        chi2rid  = chi2 / dof
        
        contatore += 1
    
    print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('prevent_overfitting ha RIMOSSO', contatore, 'picchi') 
    print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
    
    return p0, bounds
    
# Funzione per il fit di N gaussiane
def fit_N_gaussiane (x_fit, y_fit, params, bounds, N_MAX_GAUSS=5, n_acq=5, PREVENT_OVERFIT=True, 
                     err_counts=None, tolleranza=1.2, DELTA_Y=False):
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
    • PREVENT_OVERFIT: bool che indica se avviare un controllo sulle gaussiane iniziali
    • err_counts: vettore che contiene gli errori sui counts, se None viene calcolato default
    • tolleranza: valore del chi2rid sotto al quale termina il fit
    • DELTA_Y: se True viene aggiunto una traslazione di ordinata nel fit 
    
    COMMENTO su n_acq:
    
    - ciascuna acquisizione è composta di 40 misure lunghe t_acq
    - il risultato della singola acquisizione è la media di queste 40 misure
    - il count è la media delle medie delle singole acquisizioni
    - il numero totale di misure è 40*n_acq
    
    l'errore poissoniano viene corretto in funzione di questo dettaglio:
    err = sqrt(count) / sqrt(40*n_acq)
    '''
    
    # i parametri vanno fatti collassare ad un array 1D
    par_flattened = np.array(params).flatten()
    
    # se serve un parametro di traslazione verticale va aggiunto subito
    if DELTA_Y:
        par_flattened = np.append(par_flattened, 0)
        bounds[0].append(-1)
        bounds[1].append(+1)
        bounds = (np.array(bounds[0]), np.array(bounds[1]))
    
    
    # PRIMA DI TUTTO: evito l'overfitting, rimuovendo eventuali gaussiane superflue
    if PREVENT_OVERFIT:
        par_flattened, bounds = prevent_overfitting (x_fit, y_fit, par_flattened, bounds, n_acq)
    
    success, popt, pcov = esegui_fit(x_fit, y_fit, par_flattened, bounds, DELTA_Y=DELTA_Y, err_counts=err_counts)
    
    if not success:
        raise Exception('PRIMO fit non riuscito')
    
    chi2, dof, argmax = chi2_N_gaussiane (x_fit, y_fit, popt, n_acq, err_counts=err_counts, DELTA_Y=DELTA_Y)
    chi2_rid = chi2/dof
    
    ## AGGIUNTA DI GAUSSIANE IN BASE AL CHI2RID ##
    k=round(len(popt)/3)
    while chi2_rid > tolleranza and k < N_MAX_GAUSS: # gaussiane massime di default: 5
        
        # inizializzo nuovi parametri con media mu+2sigma del primo picco 
        # li attacco ai valori restituiti dal fit precedente
        a_fit = np.max(y_fit)
        if DELTA_Y:
            p0 = np.concatenate([popt[:-1], np.array([a_fit, x_fit[argmax], 50, 0])])
        else:
            p0 = np.concatenate([popt, np.array([a_fit, x_fit[argmax], 50])])
            
            
        # sistemo i bounds    
        low = [bounds[0][0], bounds[0][1], bounds[0][2]] * round(np.size(p0)/3)
        upp = [bounds[1][0], bounds[1][1], bounds[1][2]] * round(np.size(p0)/3)
        
        if DELTA_Y:
            low.append(-0.0001)
            upp.append(+0.0001)
        
        k+=1
    
        # ri-esecuzione fit
        success, popt, pcov = esegui_fit(x_fit, y_fit, p0=p0, bounds=(low,upp), DELTA_Y=DELTA_Y, err_counts=err_counts)
        
        if not success:
            print('Aggiunta di gaussiane interrotta')
            k=N_MAX_GAUSS+1 # analogo a break
        else:
            # ri-calcolo chi2 ridotto
            chi2, dof, argmax = chi2_N_gaussiane (x_fit, y_fit, 
                                                  popt, n_acq, err_counts=err_counts, 
                                                  DELTA_Y=DELTA_Y)
            chi2_rid = chi2/dof
            
    return popt, pcov, chi2_rid

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

## CONVOLUZIONE DI DUE GAUSSIANE ##
def conv_2_gauss (x, A, mu1, sigma1, mu2, sigma2):
    sigma_quadro = sigma1*sigma1 + sigma2*sigma2
    num = np.exp( -(x-mu1-mu2)*(x-mu1-mu2)/(2*sigma_quadro) )
    denom = np.sqrt(2*np.pi*sigma_quadro)
    
    return A * num / denom

# chi2 per la convoluzione
def chi2_convoluzione (x_fit, y_fit, popt, n_acq=5):
    ## CHI 2 RID ##
    y_fit_values = conv_2_gauss (x_fit, *popt)
    
    # rimuovo elementi nulli
    mask = (y_fit != 0) & (y_fit_values != 0)  # Maschera per tenere solo elementi non nulli
    y_fit = y_fit[mask]
    y_fit_values = y_fit_values[mask]
    
    # calcolo residui
    residuals = y_fit - y_fit_values
    # errore poissoniano (radice del valore) / 160 perché è una media di 160 valori!
    sigma = np.sqrt(y_fit)
    
    if n_acq != 0:
        sigma = sigma / np.sqrt(40*n_acq)  
        
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(y_fit) - len(popt)
    
    return chi2, dof

## FIT CONVOLUZIONE ##
def fit_convoluzione (x_fit, y_fit, params, bounds, n_acq=5):
    
    par_flattened = np.array(params).flatten()
    
    if n_acq != 0:
        err_y = np.sqrt( y_fit / (40*n_acq) )
    else:
        err_y = np.sqrt(y_fit)
        
    try: 
        popt, pcov = curve_fit(conv_2_gauss, x_fit, y_fit, sigma=err_y, p0=par_flattened, bounds=bounds)
    except:
        print(f"Fit non riuscito nuovamente!")
        
    chi2, dof = chi2_convoluzione (x_fit, y_fit, popt, n_acq)
    chi2_rid = chi2/dof
            
    return popt, pcov, chi2_rid

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# legge varshni
def varshni (x, E_0, alpha, beta):
     # x == T
    return E_0 - (alpha * x*x) / (beta + x)

# Funzione per il calcolo del chi2 ridotto per fit varshni  
def chi2_varshni (x_fit, y_fit, err_y, popt):
   
    y_fit_values = varshni(x_fit, *popt)
    
    # calcolo residui
    residuals = y_fit - y_fit_values
        
    chi2 = np.sum((residuals / err_y) ** 2)
    dof = len(y_fit) - len(popt)
    
    return chi2, dof

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PER L'ANALISI DEL PbS IN T.R.P.L.
# TIME RESOLVED PHOTOLUMINESCENCE
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# per la TRPL! seleziona solo la prima e le ultime due colonne
def read_trpl_csv (nomefile):

    data = np.loadtxt(nomefile, skiprows=1, delimiter=';')
    
    # colonna 0: lunghezze d'onda
    # colonna 1: counts
    
    waveln = data[:, 0]
    emissione = np.where(data[:, 3] < 0, 0, data[:, 3])
    trasmissione = np.where(data[:, 4] < 0, 0, data[:, 4])
    
    assorbimento = emissione - trasmissione
   
    return waveln, assorbimento, emissione, trasmissione

# doppio esponenziale per estrarre tau
# con traslazione sia su asse x che su asse y
def doppio_esponenziale (x, a1, tau1, a2, tau2, delta_x, delta_y):
    x = x - delta_x
    return a1 * np.exp(-x/tau1) + a2 * np.exp(-x/tau2) + delta_y

# Funzione per il calcolo del chi2 ridotto per fit doppio esp
def chi2_doppio_esponenziale (x_fit, y_fit, popt, err_y=None):
    ## CHI 2 RID ##
    y_fit_values = doppio_esponenziale(x_fit, *popt)
    
    # rimuovo elementi nulli
    mask = (y_fit != 0) & (y_fit_values != 0)  # Maschera per tenere solo elementi non nulli
    y_fit = y_fit[mask]
    y_fit_values = y_fit_values[mask]
    
    # calcolo residui
    residuals = y_fit - y_fit_values
    # errore poissoniano (radice del valore) 
    if err_y is not None:
        sigma = err_y[mask]
    else:
        sigma = np.sqrt(y_fit) # quante acquisizioni?
        
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(y_fit) - len(popt)
    
    return chi2, dof

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# PER L'ANALISI DELL'ASSORBIMENTO PbS 2000 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# stavolta uso pandas perché ci sono scritte in fondo al file
def leggi_file_assorbimento(nomefile):
    df = pd.read_csv(nomefile, delim_whitespace=True, skiprows=19, header=None, on_bad_lines='skip')

    # Converte le colonne in numeri, mettendo NaN se ci sono errori
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df[1] = pd.to_numeric(df[1], errors='coerce')

    # Rimuove righe con NaN
    df = df.dropna()

    waveln = df.iloc[:, 0].values
    #assorbimento = df.iloc[:, 1].values
    assorbimento = np.where(df.iloc[:, 1] < 0, 0, df.iloc[:, 1].values)

    return np.array(waveln), np.array(assorbimento)


# in alcune regioni il conteggio si alza in maniera anomala
# probabilmente dovuto a cambiamento lampada nello strumento
# funzione che pulisce le crescite vertiginose
def clean_counts_assorbimento(counts, n=1, n_primi=20, n_ultimi=20):
    filtered_counts = np.copy(counts)
    
    for i in range(len(counts)):
        # mi concentro su una regione localizzata        
        left = max(0, i - n_primi)
        right = min(len(counts), i + n_ultimi)
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


# Funzione per il calcolo del chi2 ridotto per fit gaussiani  
def chi2_gaussiana (x_fit, y_fit, popt, n_acq=5):
    ## CHI 2 RID ##
    y_fit_values = gaussiana(x_fit, *popt)
    
    # rimuovo elementi nulli
    mask = (y_fit != 0) & (y_fit_values != 0)  # Maschera per tenere solo elementi non nulli
    y_fit = y_fit[mask]
    y_fit_values = y_fit_values[mask]
    
    # calcolo residui
    residuals = y_fit - y_fit_values
    # errore poissoniano (radice del valore) 
    sigma = np.sqrt(y_fit)   
    
    if n_acq != 0:
        sigma = sigma / np.sqrt(40*n_acq)
        
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(y_fit) - len(popt)
    
    return chi2, dof

