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

## Datos estadisticos de la parte B adquiridos con el DAQ:

  Media: -1.2833
  
  Desviación Estándar: 0.6300
  
  Valor Mínimo: -2.3909
  
  Valor Máximo: 2.4551
  
  Rango: 4.8460
  
  Coeficiente de Variación: 49.0944%

## Datos estadisticos de la parte A adquiridos por funciones:

  Media: -0.014106341399538005
  
Desviacion estandar: 0.18098538862422828

Coeficiente de variacion (REVISAR): 1283.007290821387

Curtosis calculada con SciPy: 19.1574


