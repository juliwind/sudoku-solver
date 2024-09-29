# digit_recognition.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

MODEL_PATH = "digit_recognition_model.h5"

def define_custom_model():
    """
    Definiert ein einfaches CNN-Modell für die Ziffernerkennung.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model(model, epochs=20):
    """
    Trainiert das Modell mit den MNIST-Daten, die auf 32x32 Pixel und 3 Kanäle erweitert wurden.
    """
    # Daten laden und normalisieren
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Resize von 28x28 auf 32x32
    x_train = np.array([cv2.resize(img, (32, 32)) for img in x_train])
    x_test = np.array([cv2.resize(img, (32, 32)) for img in x_test])
    
    # Erweiterung auf 3 Kanäle
    x_train = np.stack([x_train] * 3, axis=-1).astype('float32') / 255
    x_test = np.stack([x_test] * 3, axis=-1).astype('float32') / 255
    
    # Datenaugmentation
    datagen = ImageDataGenerator(
        rotation_range=15,                # Rotation um bis zu 15 Grad
        width_shift_range=0.2,            # Horizontale Verschiebung
        height_shift_range=0.2,           # Vertikale Verschiebung
        shear_range=0.2,                  # Scherung
        zoom_range=0.2,                   # Zoom
        brightness_range=[0.8, 1.2],      # Helligkeitsanpassung
        fill_mode='nearest'               # Füllmodus
    )
    
    # Datenaugmentation auf den Trainingssatz anwenden
    datagen.fit(x_train)
    
    # Modell trainieren mit augmentierten Daten
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=epochs, validation_data=(x_test, y_test))
    
    # Modell speichern
    model.save(MODEL_PATH)
    print(f"Modell erfolgreich gespeichert unter {MODEL_PATH}")
    print(f"Endgültige Validierungsgenauigkeit: {history.history['val_accuracy'][-1]}")

def load_trained_model(force_train=False):
    """
    Lädt ein vortrainiertes Modell oder trainiert ein neues Modell, wenn kein Modell vorhanden ist
    oder force_train=True gesetzt ist.
    """
    if force_train or not os.path.exists(MODEL_PATH):
        if force_train:
            print("Force-Train-Flag gesetzt. Starte das Training eines neuen Modells...")
        else:
            print("Kein vortrainiertes Modell gefunden. Starte das Training eines neuen Modells...")
        model = define_custom_model()
        train_model(model)
    else:
        try:
            print(f"Lade vortrainiertes Modell von {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            print("Modell erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            print("Starte das Training eines neuen Modells...")
            model = define_custom_model()
            train_model(model)
    return model

def remove_grid_lines(img):
    """
    Entfernt die Rasterlinien aus dem Sudoku-Bild.
    """
    # Bild in Graustufen konvertieren (falls noch nicht geschehen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Schwellwertmethode anwenden, um ein binäres Bild zu erhalten
    _, binary_img = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Strukturierende Elemente für die Linien
    horizontalsize = int(img_gray.shape[1] / 15)
    verticalsize = int(img_gray.shape[0] / 15)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Entferne horizontale Linien
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)
    # Entferne vertikale Linien
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel)
    
    # Kombinierte Linien
    grid_lines = cv2.add(horizontal_lines, vertical_lines)
    
    # Maske für Linien erstellen
    _, mask = cv2.threshold(grid_lines, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Wende Maske an, um die Linien zu entfernen
    img_no_lines = cv2.bitwise_and(img, img, mask=mask)

    # Randbereiche entfernen (15% an jeder Seite)
    img_no_lines = crop_fixed_percentage(img_no_lines, 0.15)
    
    # Zeige die Kontrollbilder (optional, kann bei Bedarf entfernt werden)
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Binary Image", binary_img)
    # cv2.imshow("Horizontal Lines", horizontal_lines)
    # cv2.imshow("Vertical Lines", vertical_lines)
    # cv2.imshow("Combined Grid Lines", grid_lines)
    # cv2.imshow("Mask for Removing Lines", mask)
    # cv2.imshow("Image without Grid Lines", img_no_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_no_lines

def crop_fixed_percentage(img, percentage=0.15):
    """
    Schneidet einen festen Prozentsatz (15% standardmäßig) von jeder Seite des Bildes ab.
    """
    height, width = img.shape[:2]

    # Berechne die Anzahl der Pixel, die an jeder Seite abgeschnitten werden sollen
    x_crop = int(width * percentage)
    y_crop = int(height * percentage)

    # Stelle sicher, dass das Bild nach dem Zuschneiden noch eine positive Größe hat
    if width - 2 * x_crop > 0 and height - 2 * y_crop > 0:
        img_cropped = img[y_crop:height - y_crop, x_crop:width - x_crop]
        return img_cropped
    else:
        # Wenn das Bild zu klein wird, gib das Originalbild zurück
        return img

def preprocess_cell(img):
    """
    Bereitet eine Sudoku-Zelle für die Modellvorhersage vor.
    """
    # Invertieren des Bildes
    img = cv2.bitwise_not(img)
    
    # Morphologische Operationen
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Rasterlinien entfernen
    img = remove_grid_lines(img)
    
    # Resize auf 32x32 Pixel
    img = cv2.resize(img, (32, 32))
    
    # Weiteres Preprocessing
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    
    # Normalisierung
    img = img.astype('float32') / 255.0
    
    # Erweiterung auf 3 Farbkanäle (von Graustufen zu RGB)
    img = np.stack([img] * 3, axis=-1)  # Von (32,32,1) zu (32,32,3)
    img = np.expand_dims(img, axis=0)   # Von (32,32,3) zu (1,32,32,3)
    
    return img  # Rückgabe des korrekt vorbereiteten Bildes

def recognize_digit(cell_img, model, threshold=0.75):
    """
    Erkennt die Ziffer in einer Sudoku-Zelle mithilfe des Modells.
    """
    # Preprocessing der Zelle
    cell = preprocess_cell(cell_img)
    
    # Vorhersage des Modells
    prediction = model.predict(cell)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Debug-Ausgabe
    print(f"Vorhersage: {digit}, Konfidenz: {confidence}")
    
    # Wenn die Konfidenz unter dem Schwellenwert liegt, Zelle als leer markieren
    if confidence < threshold:
        return 0  # Leere Zelle
    return digit
