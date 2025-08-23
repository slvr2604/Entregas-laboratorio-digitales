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

El código carga un archivo de texto que contiene los datos de la señal adquirida con el DAQ. Primero importa las librerías necesarias (NumPy y Matplotlib), después abre el archivo y guarda la primera columna como tiempo y la segunda como la amplitud de la señal. Luego grafica la señal en un intervalo de 0 a 0.1 segundos. 
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/ad34853e-3d75-43e0-8770-55b782b5b61c" />
La gráfica muestra la señal que se obtuvo con el DAQ y que luego se trabajó en Python. En el eje del tiempo se ve desde 0 hasta 0.1 segundos y en el eje vertical la amplitud en mV. Se notan picos que se repiten de manera clara, lo que confirma que el código cargó bien los datos y que la gráfica permite visualizar la señal de forma ordenada.


    media = np.mean(senal)
    desviacion_estandar = np.std(senal)
    min_valor = np.min(senal)
    max_valor = np.max(senal)
    rango = max_valor - min_valor

    if media != 0:
    coeficiente_variacion = (desviacion_estandar / abs(media)) * 100
    else:
    coeficiente_variacion = float('inf') 

    print(f"Estadísticas Descriptivas de la Señal:")
    print(f"  Media: {media:.4f}")
    print(f"  Desviación Estándar: {desviacion_estandar:.4f}")
    print(f"  Valor Mínimo: {min_valor:.4f}")
    print(f"  Valor Máximo: {max_valor:.4f}")
    print(f"  Rango: {rango:.4f}")
    print(f"  Coeficiente de Variación: {coeficiente_variacion:.4f}%")


    from scipy.stats import kurtosis

    curt = kurtosis(senal)
    print(f"Curtosis de la señal: {curt:.4f}")

## Datos estadisticos de la parte B adquiridos con el DAQ:

  Media: -1.2833
  
  Desviación Estándar: 0.6300
  
  Valor Mínimo: -2.3909
  
  Valor Máximo: 2.4551
  
  Rango: 4.8460
  
  Coeficiente de Variación: 49.0944%

  Curtosis de la señal: 19.1574

## Datos estadisticos de la parte A adquiridos por funciones:

  Media: -0.014106341399538005
  
Desviacion estandar: 0.18098538862422828

Coeficiente de variacion (REVISAR): 1283.007290821387

Curtosis calculada con SciPy: 19.1574

En la parte B con el DAQ se ve que la señal es más amplia y con más variación en los valores. En cambio, en la parte A, al generarla con funciones, la señal sale más ajustada alrededor de cero, pero con picos muy marcados que hacen que los datos se concentren más en ciertos puntos. Esto muestra que la señal real tiene un comportamiento más extendido, mientras que la generada se ve más limitada pero con valores extremos.

    # Función de Probabilidad (usando histograma con density=True)
    plt.figure(figsize=(12, 4))
    plt.hist(senal, bins=50, density=True, edgecolor='black')
    plt.title('Función de Probabilidad de la Señal')
    plt.xlabel('Amplitud (mV)')
    plt.ylabel('Densidad de Probabilidad')
    plt.grid(True)
    plt.show()

La gráfica muestra la función de probabilidad de la señal obtenida. Aquí se puede ver cómo se distribuyen los valores de amplitud, es decir, qué tan seguido aparecen ciertos valores dentro de la señal. Esto permite identificar si los datos se concentran más en un rango específico o si están más dispersos.

<img width="988" height="393" alt="image" src="https://github.com/user-attachments/assets/ea953cc5-9eca-4a61-8ef5-69ba1b7f007a" />  <img width="996" height="394" alt="image" src="https://github.com/user-attachments/assets/f9bffa09-c2d0-446f-9dae-22cea307781f" />
La gráfica de la parte B muestra que la mayoría de valores de la señal se concentran cerca de -1 mV con un pico bien definido, mientras que en la parte A los datos se agrupan más alrededor de 0 mV y con una forma menos simétrica. Esto refleja que aunque ambas señales tienen concentraciones marcadas, el comportamiento de la adquirida con el DAQ es distinto al de la generada por funciones.

    # Histograma
    plt.figure(figsize=(12, 4))
    plt.hist(senal, bins=50, edgecolor='black')
    plt.title('Histograma de la Señal')
    plt.xlabel('Amplitud (mV)')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()
El histograma muestra la frecuencia con la que aparecen los valores de amplitud en la señal, permitiendo identificar en qué rangos se concentran más los datos y cómo se distribuyen a lo largo de la señal.
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/2ffc5d41-adc7-4b37-aae4-5625f9188db4" />   <img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/522e1a86-58a0-43b6-a406-f01b2ffe29d0" />



