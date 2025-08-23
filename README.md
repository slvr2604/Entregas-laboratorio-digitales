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
En este apartado se calcularon estadísticas importantes de la señal.

La **media** muestra el valor promedio, sirviendo como referencia de la tendencia central de los datos.

La **desviación estándar** refleja qué tanto se alejan los valores respecto al promedio; si es grande significa que hay más variación.

El **coeficiente de variación** relaciona la desviación estándar con la media y es útil para comparar la variabilidad entre señales de distinta escala.

La **curtosis** mide qué tan “picuda” o plana es la distribución de los datos, mostrando si predominan valores extremos o si los datos se agrupan más cerca del promedio.


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

<img width="988" height="393" alt="image" src="https://github.com/user-attachments/assets/ea953cc5-9eca-4a61-8ef5-69ba1b7f007a" /> }
<img width="996" height="394" alt="image" src="https://github.com/user-attachments/assets/f9bffa09-c2d0-446f-9dae-22cea307781f" />
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
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/2ffc5d41-adc7-4b37-aae4-5625f9188db4" />
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/522e1a86-58a0-43b6-a406-f01b2ffe29d0" />
La primera gráfica, que es la del DAQ, muestra la señal más dispersa y cargada hacia valores negativos, porque viene directamente de la adquisición real. En cambio, la segunda, hecha con funciones, se ve más concentrada cerca de cero y con menos ruido. Se posria observar que la señal con el DAQ sale mas desordenada y sacada por funciones se ve un poco mas limpia.


### Parte C:
La **Relación Señal-Ruido (SNR)** es una medida fundamental en el análisis de señales. Indica cuán fuerte es una señal en comparación con el ruido de fondo que la acompaña, y permite evaluar qué tan clara o distinguible resulta frente a las interferencias que pueden distorsionarla.  

```math
SNR = Potenciaseñal / Potenciaruido
```

Pero como esta relación se expresa en decibelios (dB) la expresión queda como:  

```math
SNR_dB = (10 · log10(Potenciaseñal / Potenciaruido))
```

Un SNR alto indica que la señal es fuerte y clara respecto al ruido, lo que facilita su análisis, transmisión o interpretación. Por el contrario, un SNR bajo sugiere que el ruido interfiere significativamente, dificultando la detección de patrones o eventos relevantes.

**a. Contaminar la señal con ruido gaussiano y medir el SNR**

El ruido gaussiano es un tipo de ruido aleatorio cuya distribución de probabilidad sigue una distribución normal (también llamada gaussiana). Este tipo de ruido es común en sistemas físicos y biomédicos debido a múltiples fuentes de interferencia pequeñas e independientes (por ejemplo, ruido térmico, eléctrico, ambiental).

           import numpy as np
           import matplotlib.pyplot as plt
           snr_deseado_db = 10
           potencia_senal = np.mean(senal** 2)
           potencia_ruido = potencia_senal / (10 **(snr_deseado_db / 10))
           desv_esta_ruido = np.sqrt(potencia_ruido)
           ruido = np.random.normal(0, desv_esta_ruido, len(senal))
           
           senal_con_ruido = senal + ruido
           
           potencia_ruido_real = np.mean(ruido**2)
           snr_real = potencia_senal / potencia_ruido_real
           snr_real_db = 10 * np.log10(snr_real)
           
           print(f"SNR deseado: {snr_deseado_db:.2f} dB")
           print(f"Desviación Estándar del Ruido Calculado: {desv_esta_ruido:.4f}")
           print(f"SNR real de la señal con ruido: {snr_real_db:.2f} dB")

Se tomo 10 dB como SNR deseado para que fuera más fácil visualizar el ruido en la señal original al graficarlo, luego usamos una función para sacr la medio de los cuadrados de la señal lo que nos iba a dar como resultado la potencia de la señal original. Luego se despejo la formula del SNR en dB para saber la potencia del ruido y también se hallo la desviación estandar del mismo, gracias a esto se uso una función que generó un vector de ruido gaussiano con la misma longitud que la señal original y se sumo punto a punto. Se volvio a hallar la potencial del ruido, ya que en el vector se daban números aleatorios; se usan las dos formulas de SNR y finalmente se grafica.  
  
SNR deseado: 10.00 dB  
Desviación Estándar del Ruido Calculado: 0.4521  
SNR real de la señal con ruido: 10.05 dB
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/9197a065-4b45-4fcf-9ec0-0dbe6b9580d2" />
La simulación de ruido gaussiano permite modelar interferencias realistas en señales biomédicas como el ECG, facilitando el estudio de su impacto sobre la calidad de la señal. También permite ver como un ECG se ve afectado por interferencias en un entorno clínico o experimental

**b. Contaminar la señal con ruido impulso y medir el SNR**  
El ruido impulso es una interferencia compuesta por picos breves y de alta intensidad que aparecen de forma repentina en una señal. A diferencia del ruido gaussiano, no es continuo ni predecible, y puede distorsionar gravemente puntos específicos de la señal. Este tipo de ruido es común en entornos con interferencias electromagnéticas y puede ser particularmente problemático porque introduce distorsiones significativas en la señal.


     import random # Importa el módulo random
     porcentaje_ruido_impulso = 0.01 # 1% ruido de impulso
     amplitud_impulso = np.max(np.abs(senal)) * 2 # Dos veces la amplitud absoluta máxima de la señal original
     
     senal_ruidosa_impulso = np.copy(senal)
     
    num_muestras_ruido = int(len(senal) * porcentaje_ruido_impulso)
    indices_ruido = np.random.choice(len(senal), num_muestras_ruido, replace=False)
    
    for indice in indices_ruido:
    valor_impulso = amplitud_impulso if random.random() > 0.5 else -amplitud_impulso
    senal_ruidosa_impulso[indice] += valor_impulso
    
    potencia_senal_original = np.mean(senal**2)
    
    ruido_impulso = senal_ruidosa_impulso - senal
    potencia_ruido_impulso = np.mean(ruido_impulso**2)
    
    if potencia_ruido_impulso > 0:
    snr_real_impulso = potencia_senal_original / potencia_ruido_impulso
    snr_real_impulso_db = 10 * np.log10(snr_real_impulso)
    print(f"SNR real de la señal con ruido de impulso: {snr_real_impulso_db:.2f} dB")  
    SNR real de la señal con ruido de impulso: 9.28 dB
    
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/6538bc69-37b1-4015-86d2-097534f1ddbe" />


**c.Contaminar la señal con ruido artefacto y medir el SNR**   

     frecuencia_interferencia = 60
     amplitud_interferencia = (np.max(senal) - np.min(senal)) * 0.1 
     interferencia_linea_potencia = amplitud_interferencia * np.sin(2 * np.pi * frecuencia_interferencia * tiempo)
     
     senal_con_interferencia = senal + interferencia_linea_potencia
     potencia_senal_original = np.mean(senal**2)
     
     ruido_interferencia = senal_con_interferencia - senal
     potencia_ruido_interferencia = np.mean(ruido_interferencia**2)
     
     if potencia_ruido_interferencia > 0:
     snr_real_interferencia = potencia_senal_original / potencia_ruido_interferencia
     snr_real_interferencia_db = 10 * np.log10(snr_real_interferencia)  
     print(f"SNR real de la señal con interferencia: {snr_real_interferencia_db:.2f} dB")
     else:
     print("No se añadió ruido de interferencia (la potencia del ruido es cero). El SNR es infinito.")   
     SNR real de la señal con interferencia: 12.41 dB  
     
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/71fa1c0a-dafd-4669-b22e-4f897b617e42" />









## Gráficas
     plt.figure(figsize=(12, 4))
     plt.plot(tiempo, senal_con_ruido)
     plt.title('Señal ECG con Ruido Gausseano')
     plt.xlabel('Tiempo (s)')
     plt.ylabel('Amplitud (mV)')
     plt.grid(True)
     plt.xlim(0, 0.1)
     plt.show()
   
   
     plt.figure(figsize=(12, 4))
     plt.plot(tiempo, senal_ruidosa_impulso)
     plt.title('Señal ECG con Ruido de Impulso')
     plt.xlabel('Tiempo (s)')
     plt.ylabel('Amplitud (mV)')
     plt.grid(True)
     plt.xlim(0, 0.1)
     plt.show()
   
     plt.figure(figsize=(12, 4))
     plt.plot(tiempo, senal_con_interferencia)
     plt.title('Señal ECG con Interferencia (60 Hz)')
     plt.xlabel('Tiempo (s)')
     plt.ylabel('Amplitud (mV)')
     plt.grid(True)
     plt.xlim(0, 0.1)
     plt.show()

Cada tipo de ruido se gráfica por separado para visualizar su impacto en la señal ECG. Esto permite comparar cómo cada tipo de ruido afecta la forma de la onda. Por ejemplo, el ruido gaussiano añade fluctuaciones suaves, mientras que el ruido de impulso introduce picos abruptos. La graficación de estas señales es crucial para entender el efecto del ruido y para diseñar técnicas de filtrado adecuadas
