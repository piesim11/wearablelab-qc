import numpy as np

def calculate_rms(signal):
    quadrati= signal**2
    mean= np.mean(quadrati)
    rms= np.sqrt(mean)
    return rms

if __name__ == "__main__":
    valori = np.array([3, 4])

    risultato = calculate_rms(valori)

    print("RMS calcolato:", risultato)