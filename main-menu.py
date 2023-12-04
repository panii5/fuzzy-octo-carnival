# Importar la librería Tkinter
import tkinter as tk
from tkinter import filedialog

# Crear la ventana principal
window = tk.Tk()
window.title("Programa de análisis de video de fútbol")
window.geometry("800x600")

# Crear una variable para almacenar el nombre del video
video_name = tk.StringVar()

# Crear una función para subir el video
def upload_video():
    # Abrir una ventana para seleccionar el archivo
    file = filedialog.askopenfilename(title="Seleccionar video", filetypes=[("MP4 files", "*.mp4")])
    # Si se ha seleccionado un archivo, actualizar el nombre del video
    if file:
        video_name.set(file)

# Crear una función para iniciar el análisis
def start_analysis():
    # Obtener el nombre del video
    video_path = video_name.get()
    # Si hay un video, llamar a las funciones de los módulos de carga, detección, análisis y visualización
    if video_path:
        frames, duration, width, height, fps, format = load_video(video_path)
        objects, frames = detect_objects(frames, MODEL_PATH, CLASSES, COLORS, THRESHOLD)
        df = analyze_data(objects, frames, fps)
        show_results(df, frames)
    # Si no hay un video, mostrar un mensaje de error
    else:
        tk.messagebox.showerror("Error", "No se ha seleccionado ningún video")

# Crear un botón para subir el video
upload_button = tk.Button(window, text="Subir video", command=upload_video)
upload_button.pack()

# Crear una etiqueta para mostrar el nombre del video
video_label = tk.Label(window, textvariable=video_name)
video_label.pack()

# Crear un botón para iniciar el análisis
start_button = tk.Button(window, text="Iniciar análisis", command=start_analysis)
start_button.pack()

# Crear un área para mostrar los gráficos
graph_area = tk.Canvas(window)
graph_area.pack()

# Iniciar el bucle principal de la ventana
window.mainloop()
