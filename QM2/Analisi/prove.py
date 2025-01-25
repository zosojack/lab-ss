import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ora ho i picchi, per ciascun picco bisogna fittare con gauss
# Funzione per il fit locale attorno a ciascun picco
def fit_gaussiani(asse_x, asse_y, picchi_regioni, emi, ass, temp, option=0):
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
        return y + 13 # 13 è la base di noise

    # Dizionario per memorizzare i risultati
    risultati = {"Regione 1": [], "Regione 2": []}

    plt.figure(figsize=(5, 3))
    plt.plot(asse_x, asse_y, color='blue') 

    # Margini personalizzati per ciascuna regione
    margini_regioni = [
        {"sx": 20, "dx": 25},  # Margini per Regione 1
        {"sx": 9, "dx": 9}  # Margini per Regione 2
    ]

    # Ciclo sulle regioni e sui picchi
    for i, picchi in enumerate(picchi_regioni):
        if len(picchi) == 0:
            print(f"Nessun picco nella regione {i + 1}, skip fit.")
            continue

        # Estrai i margini specifici per la regione
        margine_sx = margini_regioni[i]["sx"]
        margine_dx = margini_regioni[i]["dx"]

        # Estrai i dati per la regione corrispondente
        limiti = (asse_x[picchi[0]] - margine_sx, asse_x[picchi[-1]] + margine_dx)
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
    if option == 0:
        plt.title(f"emi={emi} | ass={ass} | T={temp}")
    else:
        plt.title(f"emi={emi} | ass={ass} | Spot {temp}")
    plt.xlabel("Lunghezza d'onda (nm)")
    plt.ylabel("Conteggi")
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

    return risultati