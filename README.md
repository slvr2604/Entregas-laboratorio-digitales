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
data:image/png;base64,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
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
