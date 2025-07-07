
#https://github.com/Jacopoo0 


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import os, random  # per navigare il file system e scegliere file casuali


root_dir= os.path.join(os.getcwd(), "progetto/progetto")
os.chdir(root_dir)
path_dataset = os.path.join(root_dir,'dataset')


# === Caricamento del dataset con TensorFlow ===
full_data = tf.keras.utils.image_dataset_from_directory(
    path_dataset,  # Directory con le immagini
    image_size=(256,256),  # Ridimensiona tutte le immagini a 256x256
    crop_to_aspect_ratio=True,  # Croppa per mantenere il rapporto larghezza/altezza
    seed=42  # Fissa il seme per la riproducibilità
)

# Visualizza il dataset e la sua lunghezza in batch
full_data, len(full_data)
class_names = full_data.class_names  # Salva le etichette delle classi
# === Divisione in training, validation e test set ===
train_size = int(0.7 * len(full_data))  # 70% per il training
val_size = int(0.15 * len(full_data))   # 15% per la validazione
test_size = int(0.15 * len(full_data))  # 15% per il test

train_data = full_data.take(train_size)  # Prende i primi batch per il training
test_data = full_data.skip(train_size)   # Salta quelli usati per il training
val_data = test_data.skip(test_size)     # Prende una parte per la validazione
test_data = test_data.take(test_size)    # Prende la parte finale per il test


# Carica il modello salvato
model_path = 'cibo_classifier.keras'
model = tf.keras.models.load_model(model_path)


# Stampa messaggio
print("\n=== INFERENZA SUL TEST SET ===")

# Prendiamo un batch dal test set
for images, labels in test_data.take(1):  # Solo un batch
    predictions = model.predict(images)  # Predizione: array con probabilità per classe
    predicted_labels = np.argmax(predictions, axis=1)  # Converti in etichette (classe con probabilità massima)

    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):  # Mostra al massimo 25 immagini
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i]]
        predicted_label = class_names[predicted_labels[i]]
        color = "green" if predicted_label == true_label else "red"
        plt.title(f"Pred: {predicted_label}\n(True: {true_label})", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break  # Solo un batch