# Este archivo contiene la clase DeepSort que realiza el seguimiento de los jugadores en el video de fútbol

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Esta clase implementa el filtro de Kalman para estimar la posición y la velocidad de los jugadores
class KalmanBoxTracker(object):
    # El número de estados del filtro de Kalman
    # Se usan 8 estados: x, y, w, h, vx, vy, vw, vh
    n_states = 8
    # El número de medidas del filtro de Kalman
    # Se usan 4 medidas: x, y, w, h
    n_measurements = 4
    # El número de entradas del filtro de Kalman
    # Se usan 0 entradas
    n_inputs = 0
    # El contador para asignar los track IDs
    count = 0

    def _init_(self, bbox):
        # El método constructor de la clase, que recibe como parámetro el bounding box inicial
        # Inicializar el filtro de Kalman con los parámetros adecuados
        self.kf = KalmanFilter(dim_x=self.n_states, dim_z=self.n_measurements, dim_u=self.n_inputs)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],  # matriz de transición de estado
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # matriz de observación
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.  # aumentar la incertidumbre de la medida para w y h
        self.kf.P[4:, 4:] *= 1000.  # dar alta incertidumbre al estado inicial de la velocidad
        self.kf.P *= 10.  # dar alta incertidumbre al estado inicial
        self.kf.Q[-1, -1] *= 0.01  # reducir la incertidumbre del proceso para vh
        self.kf.Q[4:, 4:] *= 0.01  # reducir la incertidumbre del proceso para la velocidad

        # Inicializar el estado del filtro de Kalman con el bounding box
        # Se asume que la velocidad inicial es cero
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        # Inicializar el tiempo desde la última actualización
        self.time_since_update = 0

        # Inicializar el identificador del tracker
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # Inicializar el historial del tracker
        self.history = []

        # Inicializar el número de hits del tracker
        self.hits = 0

        # Inicializar el número de veces que el tracker ha sido marcado como perdido
        self.hit_streak = 0

        # Inicializar el estado del tracker
        self.state = "Tentative"

    def update(self, bbox):
        # Este método actualiza el estado del filtro de Kalman con el bounding box observado
        # Convertir el bounding box a un vector de medida
        self.kf.update(self.convert_bbox_to_z(bbox))

        # Incrementar el tiempo desde la última actualización
        self.time_since_update = 0

        # Incrementar el número de hits del tracker
        self.hits += 1

        # Incrementar el número de veces que el tracker ha sido marcado como perdido
        self.hit_streak += 1

    def predict(self):
        # Este método predice el estado del filtro de Kalman para el siguiente cuadro
        # Si el tiempo desde la última actualización es mayor que cero, se hace la predicción
        if self.time_since_update > 0:
            self.kf.predict()

        # Incrementar el tiempo desde la última actualización
        self.time_since_update += 1

        # Añadir el estado actual al historial del tracker
        self.history.append(self.kf.x)

    def get_state(self):
        # Este método devuelve el estado actual del tracker como un bounding box
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        # Esta función convierte un bounding box de la forma (x, y, w, h) a un vector de medida de la forma (x, y, w, h)
        # Se asume que el centro del bounding box es la posición del objeto
        x, y, w, h = bbox
        return np.array([x + w / 2., y + h / 2., w, h]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        # Esta función convierte un vector de estado de la forma (x, y, w, h, vx, vy, vw, vh) a un bounding box
