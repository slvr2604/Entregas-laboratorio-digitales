# Entregas-laboratorio-digitales
Prácticos del laboratorio de procesamiento digital de señales.



## Laboratorio 1: Análisis estadístico de una señal.
### Parte A:

Se utilizó la base de datos "Physionet" para adquirir una señal fisiológica, esta a su vez fue importada y graficada.
La señal es la siguiente:


### Parte B:
Para esta parte se realizo la adquisicion de una señal real utilizando la tarjeta *NI DAQ* conectada a el generador y al osciloscopio.  
import numpy as np
import matplotlib.pyplot as plt

ruta = "/content/drive/MyDrive/Colab Notebooks/senal_DAQ.txt"

try:

    datos = np.loadtxt(ruta, skiprows=1)

    print("Datos cargados correctamente. Primeros valores:")
    print(datos[:10])

    tiempo = datos[:, 0]  
    senal = datos[:, 1]    

    plt.figure(figsize=(12, 4))
    plt.plot(tiempo, senal)
    plt.title('Señal adquirida con DAQ')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (mV)')  
    plt.grid(True)

    plt.xlim(0, 0.1)  
    plt.show()

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta {ruta}")
    print("Por favor, verifica que el archivo esté en la ruta correcta o súbelo de nuevo.")
except Exception as e:
    print(f"Ocurrió un error: {e}")

<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/ad34853e-3d75-43e0-8770-55b782b5b61c" />
