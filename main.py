
#https://github.com/Jacopoo0 

import matplotlib.pyplot as plt           # Matplotlib: libreria per grafici 2D
import matplotlib.image as mpimg           # Modulo di Matplotlib per leggere immagini
import seaborn as sns                      # Seaborn: interfaccia “high-level” per grafici statistici
import pandas as pd                        # Pandas: analisi/struttura dati in DataFrame
import numpy as np                         # NumPy: operazioni numeriche su array
import tensorflow as tf                    # TensorFlow: libreria di deep learning
import os, random                   # os: funzioni di sistema; random: scelte casuali





root_dir= os.path.join(os.getcwd(), "progetto/progetto")
os.chdir(root_dir)
path_dataset = os.path.join(root_dir,'dataset')

counts = {}                                                          # Dizionario vuoto {classe: numero_di_immagini}
for folder in os.listdir(path_dataset):                                  # Itera nelle cartelle di path_dataset
    path = os.path.join(path_dataset, folder)                            # Path completo alla sottocartella
    if os.path.isdir(path):                                          # Controlla che sia una directory
        num_files = len([f for f in os.listdir(path)                 # Conta i file (immagini) presenti
                         if os.path.isfile(os.path.join(path, f))])
        counts[folder] = num_files                                   # Salva il conteggio per la categoria

print(counts)


# Converte il dizionario in DataFrame e visualizza i numeri


df = pd.DataFrame.from_dict(counts, orient='index', columns=['Numero di immagini'])  # Dizionario → DataFrame
df.index.name = 'Categoria'                                         # Imposta il nome dell’indice
df = df.sort_values('Numero di immagini', ascending=False)          # Ordina per numero di immagini (desc)


# Heatmap con Seaborn: numero di immagini per categoria

plt.figure(figsize=(8, len(df) * 0.6))                              # Altezza dinamica: 0.6 pollici per categoria
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5) # Heatmap con annotazioni intere (“d”)
plt.title("Numero di immagini per categoria")                       # Titolo del grafico
plt.tight_layout()                                                  # Riduce margini per non tagliare testo
plt.show()                                                          # Visualizza la heatmap



# Mostra 25 immagini random dal dataset (griglia 5×5)

print(path_dataset)

plt.figure(figsize=(12, 12))                                        # Figura grande 12×12 pollici
for i in range(25):                                                 # Loop per 25 immagini
    random_folder = random.choice(os.listdir(path_dataset))             # Sceglie una categoria a caso
    filename = random.choice(os.listdir(f"{path_dataset}/{random_folder}"))  # File casuale nella categoria
    path = f"{path_dataset}/{random_folder}/{filename}"                # Path completo al file
    img = mpimg.imread(path)                                        # Carica l’immagine (array NumPy)
    plt.subplot(5, 5, i + 1)                                        # Posizione nella griglia 5×5
    plt.imshow(img)                                                 # Mostra l’immagine
    plt.axis("off")                                                 # Rimuove gli assi
    plt.title(random_folder)                                        # Titolo = nome della classe
plt.show()                                                          # Mostra l’intera griglia



# Caricamento delle immagini in un tf.data.Dataset con API Keras

full_data = tf.keras.utils.image_dataset_from_directory(
    path_dataset,                             # Directory sorgente
    image_size=(256, 256),                # Ridimensiona a 256×256
    crop_to_aspect_ratio=True,            # Croppa per rapporto corretto
    seed=42,                               
)



# Split: 70% train, 15% validation, 15% test

train_size = int(0.7 * len(full_data))    # Calcola numero di batch train
val_size   = int(0.15 * len(full_data))   # Numero batch validation
test_size  = int(0.15 * len(full_data))   # Numero batch test

train_data = full_data.take(train_size)   # Primi batch → train
test_data  = full_data.skip(train_size)   # Salta train, resta test+val
val_data   = test_data.skip(test_size)    # Ultimo pezzo → val
test_data  = test_data.take(test_size)    # Primo pezzo di test_data → test

print(f"lunghezza train, test, val,:" ,len(train_data), len(test_data), len(val_data))  # Stampa numero di batch per verifica

print(full_data.class_names)              # Stampa le etichette delle classi




# Ottimizzazione del pipeline con cache + prefetch

train_data_prefetched = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data_prefetched   = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data_prefetched  = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
#train_data_prefetched                          # Visualizza info sul dataset prefetch



# Mostra 32 immagini del training set (prima batch)

plt.figure(figsize=(10, 10))
class_names = full_data.class_names
for images, labels in train_data.take(1):       # Estrae un batch
    for i in range(32):                         # Visualizza 32 esempi
        ax = plt.subplot(6, 6, i + 1)           # Griglia 6×6
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])       # Titolo = classe
        plt.axis("off")                         # Nasconde assi
plt.show()                                      # Mostra griglia



# Definizione di un semplice modello CNN con Keras

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),  # Normalizza pixel
    tf.keras.layers.Conv2D(32, 3, activation="relu"),              # Convoluzione 32 filtri 3×3
    tf.keras.layers.MaxPooling2D(),                                # Max pooling
    tf.keras.layers.Dropout(0.1),                                  # Dropout 10%
    tf.keras.layers.Conv2D(16, 3, activation="relu"),              # Seconda conv
    tf.keras.layers.MaxPooling2D(),                                # Pooling
    tf.keras.layers.Dropout(0.2),                                  # Dropout 20%
    tf.keras.layers.Flatten(),                                     # Flatten → vettore
    tf.keras.layers.Dense(len(full_data.class_names), activation="softmax")  # Layer output
])

model.summary()                            # Stampa l’architettura




# Compilazione del modello

model.compile(
    optimizer="adam",                      # Ottimizzatore Adam
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Loss per etichette intere
    metrics=["accuracy"]                   # Monitoraggio accuracy
)


# Addestramento (10 epoche o numero a preferenza)

training = model.fit(
    train_data_prefetched,                            # Dataset train
    validation_data=val_data_prefetched,              # Dataset validation
    epochs=10                                        # Numero epoche
)

model.save('cibo_classifier.keras')           # Salva il modello in formato Keras


# Grafico accuracy vs val_accuracy

pd.DataFrame(training.history)[["accuracy", "val_accuracy"]].plot()  # Pandas → grafico
plt.title("Accuracy e Val_Accuracy nel tempo")
plt.xlabel("Epoca")
plt.ylabel("Accuratezza")
plt.show()