# Importar librerías
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Definir constantes
VIDEO_PATH = "video.mp4" # Ruta del video que el usuario sube
MODEL_PATH = "model.h5" # Ruta del modelo de CNN entrenado
CLASSES = ["jugador", "balón", "portería", "línea"] # Clases de objetos que se detectan
COLORS = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)] # Colores para dibujar los objetos
THRESHOLD = 0.5 # Umbral de confianza para la detección de objetos

# Módulo de carga de video
def load_video(video_path):
    # Leer el video y obtener sus propiedades
    video = cv2.VideoCapture(video_path)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    format = video.get(cv2.CAP_PROP_FORMAT)

    # Crear una lista vacía para almacenar las imágenes
    frames = []

    # Recorrer el video y extraer las imágenes
    while video.isOpened():
        # Leer el siguiente frame
        ret, frame = video.read()
        # Si se ha leído correctamente, añadirlo a la lista
        if ret:
            frames.append(frame)
        # Si no, terminar el bucle
        else:
            break
    
    # Liberar el video
    video.release()

    # Devolver la lista de imágenes y las propiedades del video
    return frames, duration, width, height, fps, format

# Módulo de detección de objetos
def detect_objects(frames, model_path, classes, colors, threshold):
    # Cargar el modelo de CNN
    model = tf.keras.models.load_model(model_path)

    # Crear un diccionario vacío para almacenar los datos de los objetos
    objects = {}

    # Recorrer la lista de imágenes
    for i, frame in enumerate(frames):
        # Preprocesar la imagen para adaptarla al modelo
        image = cv2.resize(frame, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Realizar la predicción con el modelo
        prediction = model.predict(image)

        # Obtener las coordenadas, las etiquetas y los niveles de confianza de los objetos
        boxes = prediction[0]
        labels = prediction[1]
        scores = prediction[2]

        # Filtrar los objetos que superen el umbral de confianza
        indices = np.where(scores > threshold)[0]

        # Añadir los datos de los objetos al diccionario
        objects[i] = {"boxes": boxes[indices], "labels": labels[indices], "scores": scores[indices]}

        # Dibujar los objetos en la imagen original
        for box, label, score in zip(boxes[indices], labels[indices], scores[indices]):
            # Obtener las coordenadas del objeto
            x1, y1, x2, y2 = box
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)

            # Obtener la etiqueta y el color del objeto
            label = classes[label]
            color = colors[label]

            # Dibujar el rectángulo y el texto del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Devolver el diccionario de objetos y la lista de imágenes con los objetos dibujados
    return objects, frames

# Módulo de análisis de datos
def analyze_data(objects, frames, fps):
    # Crear un dataframe vacío para almacenar los datos de los jugadores y el balón
    df = pd.DataFrame(columns=["frame", "time", "team", "player", "ball", "x", "y", "speed", "distance", "passes", "shots", "goals"])

    # Crear variables auxiliares para almacenar los datos anteriores
    prev_x = {}
    prev_y = {}
    prev_time = {}
    speed = {}
    distance = {}
    passes = {}
    shots = {}
    goals = {}
    ball_owner = None

    # Recorrer el diccionario de objetos
    for i in objects.keys():
        # Obtener el tiempo correspondiente al frame
        time = i / fps

        # Obtener los datos de los objetos
        boxes = objects[i]["boxes"]
        labels = objects[i]["labels"]
        scores = objects[i]["scores"]

        # Recorrer los objetos
        for box, label, score in zip(boxes, labels, scores):
            # Si el objeto es un jugador
            if label == 0:
                # Obtener el equipo al que pertenece el jugador
                x1, y1, x2, y2 = box
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                pixel = frames[i][cy, cx]
                if pixel[0] > pixel[2]:
                    team = "blue"
                else:
                    team = "red"

                # Obtener el identificador del jugador
                player = f"{team}_{label}_{score}"

                # Obtener la posición del jugador
                x = cx / width
                y = cy / height

                # Calcular la velocidad, la distancia, los pases, los tiros y los goles del jugador
                if player in prev_x and player in prev_y and player in prev_time:
                    dx = x - prev_x[player]
                    dy = y - prev_y[player]
                    dt = time - prev_time[player]
                    speed[player] = np.sqrt(dx**2 + dy**2) / dt
                    distance[player] = distance.get(player, 0) + speed[player] * dt
                    passes[player] = passes.get(player, 0)
                    shots[player] = shots.get(player, 0)
                    goals[player] = goals.get(player, 0)
