import cv2
import numpy as np

# Configuración inicial
video_path = "Videos/traficoo.mp4"  # Reemplaza con tu video #Frecuencia 97, 500
cap = cv2.VideoCapture(video_path)

# Verificar apertura del video
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Configurar sustracción de fondo
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, 
                                        varThreshold=50, 
                                        detectShadows=False)

# Crear ventanas de control
cv2.namedWindow("Deteccion")
cv2.createTrackbar("VarThreshold", "Deteccion", 50, 200, lambda x: None)
cv2.createTrackbar("AreaMin", "Deteccion", 500, 2000, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video")
        break

    # Obtener parámetros ajustables
    var_threshold = cv2.getTrackbarPos("VarThreshold", "Deteccion")
    area_min = cv2.getTrackbarPos("AreaMin", "Deteccion")
    
    # Actualizar detector
    fgbg.setVarThreshold(var_threshold)
    
    # Aplicar sustracción de fondo
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    # Procesamiento morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Detectar contornos
    contours, _ = cv2.findContours(fgmask, 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar detecciones
    vehicle_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > area_min:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vehicle_count += 1
    
    # Mostrar información
    cv2.putText(frame, f"Vehiculos: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Mostrar resultados
    cv2.imshow("Deteccion", frame)
    cv2.imshow("Mascara", fgmask)
    
    # Control de salida
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()