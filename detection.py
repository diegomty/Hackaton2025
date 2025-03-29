import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Configuración inicial
video_path = "Videos/traficoo.mp4"  # Reemplaza con tu video
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

# --- Configuración para la lectura de datos de SUMO ---
net_file = "red.net.xml"  # Reemplaza con tu archivo .net.xml
edge_ids = ["E1", "E2", "E3", "E4", "E5", "E6"]  # IDs de todas las aristas
data_file = "edge_data.xml" #Nombre del archivo de salida
# Función para leer datos de tráfico de SUMO (ejemplo con edge_data.xml)
def read_sumo_traffic_data(edge_id, data_file):
    try:
        tree = ET.parse(data_file)
        root = tree.getroot()
        
        # Buscar el edge específico
        for interval in root.findall(".//interval"):
            edge = interval.find(f".//edge[@id='{edge_id}']")
            if edge is not None:
                # Extraer datos relevantes (ejemplo: número de vehículos)
                nVehSeen = int(edge.get("nVehSeen", 0))
                return nVehSeen
        return 0 # Si no se encuentra, retornar 0

    except FileNotFoundError:
        print(f"Error: El archivo {data_file} no se encuentra.")
        return None
    except ET.ParseError:
        print(f"Error: No se pudo parsear el archivo XML {data_file}. Asegúrate de que sea un XML válido.")
        return None
    except Exception as e:
        print(f"Error inesperado al leer los datos de SUMO: {e}")
        return None

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

    # --- Lectura e impresión de datos de SUMO ---
    y_offset = 60  # Posición vertical inicial para el texto
    for edge_id in edge_ids:
        sumo_vehicle_count = read_sumo_traffic_data(edge_id, data_file)
        if sumo_vehicle_count is not None:
            cv2.putText(frame, f"SUMO {edge_id}: {sumo_vehicle_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 25  # Ajustar posición para la siguiente línea
        else:
            cv2.putText(frame, f"SUMO {edge_id}: No disponible", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_offset += 25

    # Mostrar información
    cv2.putText(frame, f"Vehiculos Detectados: {vehicle_count}", (10, 30),
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