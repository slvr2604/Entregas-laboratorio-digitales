# Entregas-laboratorio-digitales
Prácticos del laboratorio de procesamiento digital de señales.



## Laboratorio 1: Análisis estadístico de una señal. (Registro b001)
### Parte A:

Nos enfocaremos en la lectura del registro "b001" hasta el cálculo de estadísticas, histogramas y curtosis. 

El estudio de las señales fisiológicas se basa en el análisis de la serie RR para analizar si las pequeñas discrepancias que se detectan usando dos distintas derivaciones se ven influenciadas por la respiración.
Se estudiaron 20 voluntarios presuntamente sanos y el estudio se dividió en tres fases:
1.  **Reposo basal (5 min):** registros b001 a b020.  
2. **Durante música clásica (~50 min):** registros m001 a m020.  
3. **Post-música (5 min):** registros p001 a p020.  
   
Tenemos en cuenta que los sujetos permanecieron acostados boca arriba, quietos, despiertos, en una cama estándar.
 
Se utilizó una frecuencia de muestreo de 5 kHz en todos los canales.  
Aquí trabajamos con el **registro b001** (fase basal).

**Requisitos**
- **Google Colab**.
- **Python 3.x**
- Librerías: `numpy`, `matplotlib`, `wfdb`, `scipy`.  
- En Colab, el script instala `wfdb` con `pip` y `scipy` suele estar preinstalado.

Se comienza con la intalación de wdfb (que permitirá la lectura de los datos fisiológicos) y además haciendo uso de las librerias necesarias tales como:

```
!pip install wfdb  
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import random
```

También usamos utilidades de Colab para subir archivos y montar Google Drive:

```
from google.colab import files
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')
```
Se lee el registro b001 desde Drive:
```
signals, fields = wfdb.rdsamp("/content/drive/MyDrive/Colab Notebooks/b001")
fields  # metadatos del registro
```

Extraemos un segmento de 10000 datos
```
signal = signals[80000:90000]
```
Si tenemos en cuenta que la frecuencia de muestreo es igual a 5000 Hz y que las muestras son 10000, podremos inferir que el tiempo de la muestra es de dos segundos.


Ahora procedemos con la gráfica. Se grafica la columna 0 (típicamente ECG derivación I):
```
plt.figure(figsize=(12, 4))
plt.plot(signal[:, 0])
plt.title('Señal Fisiológica')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.legend()
plt.grid()
plt.show()
```
<img width="1012" height="394" alt="17560010329671912406512794671258" src="https://github.com/user-attachments/assets/3ed1fd3b-c508-42b4-9c1b-a6e794e96cfe" />

Y luego procedemos con el cálculo de los datos estadísticos:

```
datos = signal[:,0]
media = np.mean(datos)
desviacion = np.std(datos, ddof=1)
coef_var = desviacion / media

print("Media:", media)
print("Desviación estándar:", desviacion)
print("Coeficiente de variación:", coef_var)
```
**Media: -0.014106341399538005**


**Desviacion estandar: 0.18098538862422828**


**Coeficiente de variacion: 12.83%**


Luego procedemos a normalizar el histograma y se grafica como barras centradas en cada bin.
Se obtiene el recuento por bins y sus bordes con numpy:
```
valores, bordes = np.histogram(datos, bins=30)
```
Y se grafica el histograma:
```
plt.figure(figsize=(12, 4))
plt.hist(datos, bins=30, edgecolor="black")
plt.xlabel("Amplitud (mV)")
plt.ylabel("Frecuencia (Hz)")
plt.title("Histograma de la señal")
plt.grid(True)
plt.show()
```
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/bbb9fa2c-4bcd-4375-acba-57faf768613d" />

Para calcular la función de probabilidad se normaliza el histograma (density=True) y se grafica como barras centradas en cada bin:
```
valores, bordes = np.histogram(datos, bins=30, density=True)
centros = (bordes[:-1] + bordes[1:]) / 2
plt.figure(figsize=(12, 4))
plt.bar(centros, valores, width=(bordes[1]-bordes[0]), edgecolor="black")
plt.xlabel("Amplitud (mV)")
plt.ylabel("Probabilidad")
plt.title("Función de probabilidad de la señal")
plt.grid(True)
plt.show()
```
<img width="996" height="394" alt="image" src="https://github.com/user-attachments/assets/2db437da-29cb-48c6-a535-a14bcb719be8" />

Ahora calculamos la curtosis (con SciPy)
```
from scipy.stats import kurtosis
curtosis_scipy = kurtosis(datos, fisher=True)  
```
**Curtosis con SciPy: 21.850227609596285**

fisher=True entrega el exceso de curtosis (se resta 3).
Valores > 0 implican colas más pesadas que una normal y < 0, más ligeras.

Tras haber calculado los datos estadísticos con funciones de Python, procedemos a calcular los datos a través de funciones manuales:

Se calcula media, desviación estandar y coeficiente de variación.
```
cont = 0
suma = 0
for valor in signal[:, 0]:
    suma += valor
    cont += 1
media_manual = suma / cont
```
```
sum_a = 0
for valor in signal[:, 0]:
    sum_a += (valor - media_manual)**2
desviacion_manual = (sum_a / cont)**0.5
```
```
coef = (desviacion_manual / abs(media_manual)) * 100 
```
**Media: -0.014106341399537866**
**Desviacion estandar: 0.18098538862422764**
**Coeficiente de variacion: 1283.007290821395** (Revisar)

Se vuelve a calcular el histograma donde definimos el número de bins y límites, contamos ocurrencias por bin y graficamos con plt.bar. Además, se imprimen valor mínimo, valor máximo y ancho de bin.

```
amplitudes_senal = signal[:,0]
num_bins = 30
min_amplitud = min(amplitudes_senal)
max_amplitud = max(amplitudes_senal)
ancho_bin = (max_amplitud - min_amplitud) / num_bins
limites_bins = [min_amplitud + i * ancho_bin for i in range(num_bins + 1)]

frecuencias_manual = [0] * num_bins
for amplitud in amplitudes_senal:
    for i in range(num_bins):
        if limites_bins[i] <= amplitud < limites_bins[i+1]:
            frecuencias_manual[i] += 1
            break
centros_bin = [min_amplitud + (i + 0.5) * ancho_bin for i in range(num_bins)]

plt.figure(figsize=(12, 4))
plt.bar(centros_bin, frecuencias_manual, width=ancho_bin, align='center', edgecolor='black')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Frecuencia (Hz)')
plt.title('Histograma Manual')
plt.grid(True)
plt.show()

print("Valor mínimo:", min_amplitud)
print("Valor máximo:", max_amplitud)
print("Ancho de cada bin:", ancho_bin)
```
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/57a4a461-3f90-4ffb-b237-6eb710ad2d57" />

**Valor mínimo: -0.1434077873293117
Valor máximo: 1.1874437609651838
Ancho de cada bin: 0.044361718276483185**

Para el PDF se normaliza cada bin como frecuencia / (N * ancho_bin) y se grafica.
```
N_muestras = len(datos)
ancho_bin_prob = ancho_bin
frecuencias_prob = frecuencias_manual
pdf_manual = [frecuencia / (N_muestras * ancho_bin_prob) for frecuencia in frecuencias_prob]
centros_bin_prob = centros_bin


plt.figure(figsize=(12, 4))
plt.bar(centros_bin_prob, pdf_manual, width=ancho_bin_prob, align='center', edgecolor='black')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Probabilidad')
plt.title('Función de Probabilidad Manual')
plt.grid(True)
plt.show()
```
<img width="996" height="393" alt="image" src="https://github.com/user-attachments/assets/af73cb4e-8c7a-4216-85f6-27008ca5e22a" />

Finalmente calculamos la curtosis de manera manual
```
datos2 = np.array(datos)
media = np.mean(datos2)
m2 = np.mean((datos2 - media)**2)
m4 = np.mean((datos2 - media)**4)

curtosis_manual = m4 / (m2**2) - 3
print("La curtosis es:", curtosis_manual)
```
Y observamos que corresponde con la curtosis de funciones de Python.


**La curtosis es: 21.850227609596285**.


Finalmente concluimos:


La media refleja el nivel promedio de la señal en milivoltios (mV). En el ECG, suele estar alrededor de cero porque se registran diferencias de potencial eléctrico, lo que indica que el análisis fue correcto. La desviación estandar nos muestra cuánto se dispersan los valores respecto a la media. En este caso nos mostró un valor bajo indicando estabilidad de la señal.
El coeficiente de variación en un ECG limpio, debe ser relativamente bajo (<10–15%), lo que en este caso nos indica que la señal es estable y repetitiva.

El histograma permitió ver la distribución de amplitudes. 

La curtosis al ser positiva, significa que los picos son más agudos y concentrados (lo esperado en un ECG con complejos QRS bien definidos). 

Los resultados sugieren que la señal tiene un comportamiento estable, con baja dispersión, distribución centrada alrededor de cero y picos característicos. Esto confirma que los cálculos reflejan una señal fisiológica regular y sin alteraciones mayores en la calidad del registro.

A su vez, la señal ECG del sujeto b001 en reposo muestra un ritmo cardíaco regular y fisiológicamente normal. El análisis estadístico confirma esta estabilidad al evidenciar baja variabilidad relativa y distribución esperada de la amplitud.

### Parte B:   
Se generó una señal fisiológica del mismo tipo de la usada en la parte A usando el generador de señales biológicas junto con el *NI DAQ* y el osciloscopio para verificar que la señal diera de la manera adecuada. Cuando se visualizo que la gráfica estaba bien, se hizo un código en phyton para poder graficar la señal eb colab por medio de un documento .txt


    !pip install nidaqmx
    import nidaqmx
    import numpy as np
    import pandas as pd
 
    canal_ai = "Dev1/ai0"
    frecuencia = 5000
    muestras = 10000

    with nidaqmx.Task() as adquisicion:
    adquisicion.ai_channels.add_ai_voltage_chan(canal_ai)
    adquisicion.timing.cfg_samp_clk_timing(rate=frecuencia, samps_per_chan=muestras)
    senal = adquisicion.read(number_of_samples_per_channel=muestras)
    senal = np.array(senal)

    t = np.arange(0, muestras) / frecuencia

    datos = pd.DataFrame({"Tiempo (s)": t, "Voltaje (V)": senal})

    datos.to_csv("registro_senal.csv", index=False)
    datos.to_csv("registro_senal.txt", sep="\t", index=False)
    datos.to_feather("registro_senal.feather")

    print("Archivos guardados: registro_senal.csv, registro_senal.txt y registro_senal.feather")

El código carga un archivo de texto que contiene los datos de la señal adquirida con el DAQ. Primero importa las librerías necesarias (NumPy y Matplotlib), después abre el archivo y guarda la primera columna como tiempo y la segunda como la amplitud de la señal. Luego grafica la señal en un intervalo de 0 a 0.1 segundos.   
 
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
 

<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/bc5afd39-b815-4a82-bd27-3b23bf47097f" />

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



 **Datos estadisticos de la parte B adquiridos con el DAQ:**

  Media: -1.2833
  
  Desviación Estándar: 0.6300
  
  Valor Mínimo: -2.3909
  
  Valor Máximo: 2.4551
  
  Rango: 4.8460
  
  Coeficiente de Variación: 49.0944%

  Curtosis de la señal: 19.1574

**Datos estadisticos de la parte A adquiridos por funciones:**

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

<img width="988" height="393" alt="image" src="https://github.com/user-attachments/assets/353d1bae-75ee-419b-927a-bd6546e41b3c" />

<img width="996" height="394" alt="image" src="https://github.com/user-attachments/assets/2ddb1f91-2a77-4853-95fc-b914a7f9aa45" />


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
<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/7c4c90f0-7d7e-4acb-9a8b-5322ff3e12f3" />

<img width="1014" height="393" alt="image" src="https://github.com/user-attachments/assets/8361bea5-93a1-49c4-a234-c905de745556" />


La primera gráfica, que es la del DAQ, muestra la señal más dispersa y cargada hacia valores negativos, porque viene directamente de la adquisición real. En cambio, la segunda, hecha con funciones, se ve más concentrada cerca de cero y con menos ruido. Se posria observar que la señal con el DAQ sale mas desordenada y sacada por funciones se ve un poco mas limpia.


### Parte C:
La **Relación Señal-Ruido (SNR)** es una medida fundamental en el análisis de señales. Indica cuán fuerte es una señal en comparación con el ruido de fondo que la acompaña, y permite evaluar qué tan clara o distinguible resulta frente a las interferencias que pueden distorsionarla.  

**SNR = Potenciadelaseñal / Potenciadelruido**


Pero como esta relación se expresa en decibelios (dB) la expresión queda como:  

**SNR_dB = (10 · log10(Potenciadelaseñal / Potenciadelruido))**


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


     import random 
     porcentaje_ruido_impulso = 0.01 # 1% ruido de impulso
     amplitud_impulso = np.max(np.abs(senal)) * 2 
     
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

Se uso random para tener valores aleatorios de forma que el impulso pueda ser negativo o positivo, desde el inicio se programo para que solo el 1% de la señal tuviera ruido y se hizo una copia de la señal original para no modificarla cuando se le agregará el ruido, aparte se definió que la amplitud del ruido fuera la más alta de la señal y se multiplica por 2 para que sea más claro el ruido. También se calculó cuántas muestras serán afectadas por el ruido y se generaron indices aleatorios para agregar ruido y que no se repitieran, luego se hizo un for para que recorra tos los indices y les agregue un impulso sea negativo o positivo, finalmente se hicieron los cálculos de potencia para la señal original y para la del ruido con el fin de sacar el SNR.

    SNR real de la señal con ruido de impulso: 9.28 dB
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/6538bc69-37b1-4015-86d2-097534f1ddbe" />


**c.Contaminar la señal con ruido artefacto y medir el SNR**  
El ruido artefacto es una distorsión no deseada que aparece en una señal debido a fallos técnicos, interferencias ambientales o procesos biológicos, y que no forma parte de la información original. Puede surgir por errores en sensores, movimientos musculares en señales médicas, interferencias electromagnéticas, o problemas de muestreo digital, y afecta la calidad y precisión del análisis. A diferencia de otros tipos de ruido, los artefactos suelen tener patrones específicos o repentinos que pueden confundirse con datos reales, por lo que se utilizan técnicas de filtrado, separación de fuentes o detección automática para eliminarlos o reducir su impacto. En este caso nosotros tomamos la red eléctrica en Colombia que esta a 60 Hz como ruido artefacto.

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

Inicialmente se definió la amplitud de la interferencia para después generar la señal de interferencia que es una onda senoidal, después se le sumo dicha señal a la origina y finalmente se calcularon potencias junto con el SNR.  
SNR real de la señal con interferencia: 12.41 dB
     
<img width="1015" height="393" alt="image" src="https://github.com/user-attachments/assets/71fa1c0a-dafd-4669-b22e-4f897b617e42" />

La señal ECG mostrada evidencia cómo la interferencia de 60 Hz proveniente de la red eléctrica puede superponerse a los componentes fisiológicos reales, generando oscilaciones que distorsionan la lectura. Este tipo de ruido artefacto, puede comprometer la precisión diagnóstica si no se filtra adecuadamente.







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

