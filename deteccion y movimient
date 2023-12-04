import cv2
import numpy as np
import tensorflow as tf
from sort import Sort # Importamos el algoritmo de seguimiento de objetos SORT

# Cargamos el modelo preentrenado de detección de objetos YOLO
model = tf.keras.models.load_model('yolo.h5')

# Definimos las posibles clases de objetos que puede detectar el modelo
classes = ["Jugador", "Pelota", "Portería", "Árbitro", "Otra"]

# Definimos los colores para cada clase de objeto
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Creamos una instancia del algoritmo de seguimiento de objetos SORT
tracker = Sort()

def procesar_frame(frame):
    # Predecimos las clases y las coordenadas de los objetos detectados por el modelo YOLO
    # Convertimos el frame a un array de numpy
    frame = np.array(frame)
    # Añadimos una dimensión extra para el batch
    frame = np.expand_dims(frame, axis=0)
    # Obtenemos las probabilidades y las coordenadas de cada clase
    probs, coords = model.predict(frame)
    # Obtenemos las clases y las coordenadas de los objetos con mayor probabilidad
    preds = []
    for i in range(len(probs)):
        if probs[i] > 0.5: # Umbral de confianza
            preds.append([coords[i][0], coords[i][1], coords[i][2], coords[i][3], probs[i], np.argmax(probs[i])])
    # Convertimos la lista de predicciones a un array de numpy
    preds = np.array(preds)
    # Usamos el algoritmo de seguimiento de objetos SORT para asignar un identificador a cada objeto
    tracks = tracker.update(preds)
    # Aquí podemos agregar más lógica para procesar la información detectada y seguida
    # Por ejemplo, podemos guardar los datos en un archivo o enviarlos a otro módulo
    # En este caso, solo vamos a devolver los datos como un diccionario
    data = {}
    for track in tracks:
        # Obtenemos las coordenadas, el identificador y la clase del objeto
        x1, y1, x2, y2, track_id, class_id = track
        # Creamos una clave con el nombre de la clase y el identificador del objeto
        key = f"{classes[class_id]}_{track_id}"
        # Creamos un valor con las coordenadas del objeto
        value = (x1, y1, x2, y2)
        # Añadimos el par clave-valor al diccionario
        data[key] = value
    return data

def analizar_video(video_path):
    # Cargamos el video
    capture = cv2.VideoCapture(video_path)
    # Creamos una lista para guardar los datos de cada frame
    data_list = []
    while True:
        ret, frame = capture.read()

        if not ret:
            break

        # Procesamos el frame
        data = procesar_frame(frame)
        # Añadimos los datos del frame a la lista
        data_list.append(data)

    # Liberamos recursos
    capture.release()
    # Devolvemos la lista de datos
    return data_list

if _name_ == "_main_":
    # Ruta del video
    video_path = "ruta/del/video.mp4"

    # Iniciamos el análisis del video
    data_list = analizar_video(video_path)
    # Aquí podemos hacer algo con la lista de datos, como guardarla en un archivo o enviarla a otro módulo
    # En este caso, solo vamos a mostrarla por consola
    print(data_list)
