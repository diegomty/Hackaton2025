import cv2
import numpy as np
import time

# Cargar el modelo pre-entrenado
prototxt = "models/MobileNetSSD_deploy.prototxt"
model = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Clases que detecta MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Iniciar webcam
cap = cv2.VideoCapture("Videos/trafic3.mp4")

# Variables para medir tiempo entre puntos
puntos = []  # Almacenará los dos puntos seleccionados por el usuario
tiempos = {}  # Almacenará {id_auto: (t_inicio, punto_inicio)}

# Función para manejar clics del mouse
def seleccionar_puntos(event, x, y, flags, param):
    global puntos
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos) < 2:
            puntos.append((x, y))
            print(f"Punto {len(puntos)} seleccionado: ({x}, {y})")

cv2.namedWindow("Deteccion")
cv2.setMouseCallback("Deteccion", seleccionar_puntos)

# Almacenar centroides previos para seguimiento
centroides_previos = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar puntos seleccionados
    for punto in puntos:
        cv2.circle(frame, punto, 5, (0, 255, 0), -1)

    # Preprocesar imagen para el modelo
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Detectar y rastrear autos
    car_count = 0
    centroides_actuales = {}

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if CLASSES[class_id] == "car":
                # Obtener coordenadas del auto
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")
                centroide = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Dibujar cuadro y centroide
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, centroide, 3, (0, 0, 255), -1)
                
                # Asignar ID basado en proximidad con centroides previos
                min_dist = 50  # Máxima distancia para considerar mismo auto
                id_auto = None
                for id_previo, centroide_previo in centroides_previos.items():
                    dist = np.linalg.norm(np.array(centroide) - np.array(centroide_previo))
                    if dist < min_dist:
                        id_auto = id_previo
                        min_dist = dist
                
                # Si no se encuentra, asignar nuevo ID
                if id_auto is None:
                    id_auto = len(centroides_previos) + 1
                
                centroides_actuales[id_auto] = centroide
                car_count += 1

                # Verificar si el auto cruza un punto
                if len(puntos) == 2:
                    for idx, punto in enumerate(puntos):
                        dist = np.linalg.norm(np.array(centroide) - np.array(punto))
                        if dist < 20:  # Umbral de proximidad al punto
                            if id_auto not in tiempos:
                                tiempos[id_auto] = (time.time(), idx)
                            else:
                                t_inicio, punto_inicio = tiempos[id_auto]
                                if punto_inicio != idx:
                                    t_total = time.time() - t_inicio
                                    print(f"Auto {id_auto}: Tiempo entre puntos = {t_total:.2f}s")
                                    del tiempos[id_auto]

    # Actualizar centroides previos
    centroides_previos = centroides_actuales.copy()

    # Mostrar conteo
    cv2.putText(frame, f"Autos: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Deteccion", frame)

    # Salir si la ventana está cerrada o se presiona 'q'
    if cv2.getWindowProperty("Deteccion", cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()