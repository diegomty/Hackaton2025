import cv2
import numpy as np

# Iniciar webcam
cap = cv2.VideoCapture("Videos/trafic3.mp4")

# Configurar sustracción de fondo con umbral ajustable
var_threshold = 50  # Valor inicial (ajústalo según necesidad)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=False)

# Trackbars para ajustar umbrales en tiempo real
def nothing(x):
    pass

cv2.namedWindow("Ajustes")
cv2.createTrackbar("VarThreshold", "Ajustes", var_threshold, 200, nothing)
cv2.createTrackbar("AreaMin", "Ajustes", 500, 2000, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener valores de los trackbars
    var_threshold = cv2.getTrackbarPos("VarThreshold", "Ajustes")
    area_min = cv2.getTrackbarPos("AreaMin", "Ajustes")

    # Actualizar sustracción de fondo con el nuevo umbral
    fgbg.setVarThreshold(var_threshold)

    # Aplicar sustracción de fondo
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Detectar contornos y filtrar por área mínima
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > area_min:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar resultados
    cv2.imshow("Deteccion", frame)
    cv2.imshow("Mascara", fgmask)

    # Salir con 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()