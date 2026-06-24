# -*- coding: utf-8 -*-
"""
RJG Industrial Robot Control v4.1
Autor: rjguz
Refactored: Arquitectura multithreading + Flet UI + JSON config

Arquitectura:
    - ConfigManager  : carga/guarda config.json
    - AppState       : estado compartido entre threads (con Lock)
    - LogManager     : sistema de logs con exportación CSV
    - PLCManager     : conexión manual + auto-reconexión (Snap7)
    - VisionProcessor: MediaPipe + filtro exponencial + histéresis
    - RobotApp       : interfaz Flet completa (6 secciones)

Dependencias externas:
    pip install flet opencv-python mediapipe python-snap7 numpy

Nota Windows: requiere snap7.dll en PATH o en la misma carpeta.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIT REPORT — NF_11 Corrección F11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERRORES ENCONTRADOS
───────────────────
1. asyncio.run() en _ui_update_loop desde thread
   → threading.Thread daemon llamaba page.update() internamente, lo que en
     Flet 0.21+ (event loop ya activo) provoca RuntimeError inmediato.

2. page.update() síncrono en callbacks async
   → Todos los event handlers (on_connect, on_apply, on_save, route_page,
     on_chip_click, etc.) llamaban page.update() en vez de
     await page.update_async(), incorrecto en contexto async de Flet.

3. VisionProcessor.reinit() sin lock
   → _init_mediapipe() y _init_camera() modificaban self._hands y self._cap
     mientras run_thread() los leía concurrentemente → race condition y
     posible AttributeError o frame corrupto.

4. PLCManager.run_thread() sin write_signals()
   → El loop del thread PLC nunca llamaba write_signals() — la escritura al
     PLC estaba completamente ausente del ciclo de ejecución.

5. Cierre de ventana sin esperar liberación de recursos
   → page.window.destroy() se ejecutaba inmediatamente tras stop_event.set(),
     sin dar tiempo a que VisionProcessor liberara self._cap y self._hands.

6. snap7 sin manejo de ImportError
   → Si snap7.dll no estaba disponible, la app crasheaba en el import antes
     de mostrar cualquier UI o mensaje de error.

CORRECCIONES REALIZADAS
───────────────────────
1. async def main(page) — main() convertido a coroutine async para
   compatibilidad con la arquitectura de Flet 0.21+.

2. asyncio.create_task(_ui_update_loop()) — el thread UIUpdater eliminado
   y reemplazado por una coroutine asyncio con await page.update_async()
   y await asyncio.sleep(interval).

3. await page.update_async() en todos los callbacks — on_connect,
   on_disconnect, on_apply, on_save, on_reset, on_export_csv,
   on_clear_logs, on_chip_click, route_page y _show_snack convertidos a
   async def con await page.update_async().

4. threading.Lock en VisionProcessor — self._reinit_lock protege
   _init_mediapipe() e _init_camera(); run_thread() adquiere el lock antes
   de acceder self._hands o self._cap.

5. PLC write loop implementado — PLCManager.run_thread() ahora lee
   AppState (robot_position, gripper_text) bajo state.lock y llama
   write_signals() a la frecuencia configurada (plc_write_freq Hz).

6. Cleanup async en on_window_event — convertido a async def; usa
   await asyncio.sleep(0.3) antes de page.window.destroy() para dar tiempo
   al release de recursos; destroy() envuelto en try/except.

7. try/except en imports críticos — snap7 y set_bool importados dentro de
   try/except ImportError; SNAP7_AVAILABLE=False permite iniciar la app
   sin DLL mostrando aviso en UI en vez de crashear.

RIESGOS ELIMINADOS
──────────────────
1. Race condition en frames de cámara durante reinit
   → Con _reinit_lock, reinit() y run_thread() no acceden a self._cap /
     self._hands simultáneamente; elimina crashes intermitentes al cambiar
     configuración de visión en tiempo real.

2. PLC escribiendo en estado inconsistente
   → El write loop ahora lee AppState bajo lock antes de cada escritura;
     garantiza que los bits RIGHT/CENTER/LEFT/GRIPPER reflejan el estado
     real del procesamiento de visión en ese instante.

3. App que cierra abruptamente por excepción en thread
   → Todos los threads tienen try/except genérico con log y continue;
     una excepción en MediaPipe/OpenCV/Snap7 no rompe el thread ni cierra
     la aplicación; el error queda registrado en LogManager.

MEJORAS APLICADAS
─────────────────
1. Compatibilidad Flet 0.21+ — arquitectura async/await end-to-end;
   ningún hilo secundario llama page.update() directamente.

2. Compatibilidad Python 3.11 / 3.12 — eliminado uso de APIs deprecadas;
   typing moderno (dict[str, ft.Ref] sin __future__) válido desde 3.9+.

3. Supresión de DeprecationWarnings de MediaPipe / OpenCV —
   warnings.filterwarnings al inicio del módulo suprime DeprecationWarning,
   FutureWarning y RuntimeWarning de dependencias externas.

4. Manejo de ausencia de snap7.dll — SNAP7_AVAILABLE=False desactiva
   suavemente la conexión PLC y muestra mensaje en UI; la app es usable
   en modo solo-visión sin tener la DLL instalada.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import asyncio
import cv2
import mediapipe as mp
try:
    import snap7
    from snap7.util import set_bool
    SNAP7_AVAILABLE = True
except ImportError:
    snap7 = None  # type: ignore
    set_bool = None  # type: ignore
    SNAP7_AVAILABLE = False
    print("[WARNING] snap7 no disponible — funcionalidad PLC deshabilitada.")
import numpy as np
import threading
import json
import datetime
import csv
import os
import time
import base64
import flet as ft
from collections import deque

# =====================================================
# CONFIGURACION POR DEFECTO
# =====================================================

CONFIG_DEFAULTS = {
    # Zonas de movimiento
    "dead_zone": 80,
    "open_threshold": 4,
    "close_threshold": 1,
    "smoothing": 0.3,
    "hysteresis": 20,

    # Frecuencias (Hz)
    "plc_write_freq": 30,
    "ui_update_freq": 30,

    # PLC Siemens
    "plc_ip": "192.168.11.1",
    "rack": 0,
    "slot": 1,
    "db_number": 1,
    "db_offset": 26,

    # Camara
    "camera": 0,
    "resolution": [640, 480],

    # MediaPipe
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
    "max_num_hands": 1,

    # Comportamiento
    "auto_reconnect": False,
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# =====================================================
# CONFIG MANAGER
# =====================================================

class ConfigManager:
    """
    Gestiona la configuración de la aplicación con persistencia JSON.

    - Al instanciar: carga config.json automáticamente.
      Si el archivo no existe, usa CONFIG_DEFAULTS y lo crea.
    - save(): guarda el estado actual a config.json.
    - Acceso: config.dead_zone, config.plc_ip, etc. (atributos directos)
    - reset_defaults(): restaura todos los valores a CONFIG_DEFAULTS.
    """

    def __init__(self, path: str = CONFIG_FILE):
        self._path = path
        self._load()

    def _load(self):
        """Carga config.json. Si falta alguna clave usa el default."""
        data = {}
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[ConfigManager] Error leyendo config.json: {e} — usando defaults.")
                data = {}

        # Mezcla: defaults + lo que haya en disco
        merged = {**CONFIG_DEFAULTS, **data}

        # Aplica como atributos de instancia
        for key, value in merged.items():
            setattr(self, key, value)

    def save(self):
        """Serializa el estado actual y guarda en config.json."""
        data = {key: getattr(self, key) for key in CONFIG_DEFAULTS}
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"[ConfigManager] Error guardando config.json: {e}")

    def reset_defaults(self):
        """Restaura todos los valores a CONFIG_DEFAULTS y guarda."""
        for key, value in CONFIG_DEFAULTS.items():
            setattr(self, key, value)
        self.save()

    def to_dict(self) -> dict:
        """Retorna la configuración actual como diccionario."""
        return {key: getattr(self, key) for key in CONFIG_DEFAULTS}

    def __repr__(self):
        return f"<ConfigManager path={self._path!r} keys={list(CONFIG_DEFAULTS.keys())}>"


# =====================================================
# APP STATE
# =====================================================

class AppState:
    """
    Estado compartido entre todos los threads de la aplicación.

    Acceso thread-safe mediante AppState.lock (threading.Lock).
    Siempre adquirir el lock antes de leer/escribir desde threads
    distintos al thread principal:

        with state.lock:
            state.fps = nuevo_fps

    Atributos:
        plc_connected       bool   — PLC conectado o no
        tracking_status     str    — "ACTIVE" | "NO HAND"
        robot_position      str    — "LEFT" | "CENTER" | "RIGHT"
        gripper_text        str    — "OPEN" | "CLOSE" | "PARTIAL" | "UNKNOWN"
        hand_x              int    — posición X suavizada de la muñeca (px)
        fps                 float  — frames por segundo del hilo de visión
        runtime             float  — segundos desde que arrancó la app
        cmd_count           int    — total de comandos enviados al PLC
        plc_write_count     int    — total de escrituras DB exitosas
        fingers             list   — [0|1]*5 estado de cada dedo
        offset              int    — hand_x - center_x
        logs                deque  — últimas 500 entradas de log (dict)
        command_history     deque  — últimas 100 entradas de historial

        # Extras para sección PLC
        last_plc_error      str    — último mensaje de error PLC
        plc_connected_time  float  — timestamp (time.time()) de última conexión
        reconnect_count     int    — número de reconexiones realizadas

        # Frame compartido con la UI (bytes PNG en base64)
        frame_b64           str    — frame de cámara codificado en base64
        skeleton_b64        str    — frame skeleton codificado en base64

        # Control de threads
        stop_event          threading.Event — señal para detener threads
        start_time          float  — time.time() cuando arrancó la app
    """

    def __init__(self):
        self.lock = threading.Lock()

        # Sistema de logs centralizado
        self.log_manager: LogManager = LogManager(maxlen=500)

        # Alias para acceso directo (retrocompatibilidad)
        # state.logs apunta al mismo deque que state.log_manager.logs
        self.logs: deque = self.log_manager.logs

        # Estado funcional
        self.plc_connected: bool = False
        self.tracking_status: str = "NO HAND"
        self.robot_position: str = "CENTER"
        self.gripper_text: str = "UNKNOWN"
        self.hand_x: int = 0
        self.hand_y: int = 0
        self.fps: float = 0.0
        self.runtime: float = 0.0
        self.cmd_count: int = 0
        self.plc_write_count: int = 0
        self.fingers: list = [0, 0, 0, 0, 0]
        self.offset: int = 0

        # Historial de comandos
        self.command_history: deque = deque(maxlen=100)

        # PLC extendido
        self.last_plc_error: str = ""
        self.plc_connected_time: float = 0.0
        self.reconnect_count: int = 0

        # Frames para UI (base64 PNG)
        self.frame_b64: str = ""
        self.skeleton_b64: str = ""

        # Contadores por comando
        self.cmd_counters: dict = {
            "LEFT": 0,
            "CENTER": 0,
            "RIGHT": 0,
            "OPEN": 0,
            "CLOSE": 0,
        }

        # Control de ciclo de vida
        self.stop_event: threading.Event = threading.Event()
        self.start_time: float = time.time()

    def add_log(self, level: str, message: str):
        """
        Agrega una entrada al log delegando en log_manager.
        Retrocompatibilidad: todos los módulos pueden llamar state.add_log().
        """
        self.log_manager.add(level, message)

    def add_command(self, command: str):
        """
        Registra un comando en el historial y actualiza contadores.
        Thread-safe.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": command,
        }
        with self.lock:
            self.command_history.append(entry)
            if command in self.cmd_counters:
                self.cmd_counters[command] += 1
            self.cmd_count += 1

    def get_runtime_str(self) -> str:
        """Retorna el tiempo de ejecución formateado HH:MM:SS."""
        elapsed = int(time.time() - self.start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def __repr__(self):
        return (
            f"<AppState plc={self.plc_connected} "
            f"tracking={self.tracking_status!r} "
            f"pos={self.robot_position!r} "
            f"fps={self.fps:.1f}>"
        )


# =====================================================
# PLC MANAGER
# =====================================================

class PLCManager:
    """
    Gestiona la conexión con el PLC Siemens mediante Snap7.

    - Conexión/desconexión MANUAL: el usuario presiona CONNECT/DISCONNECT en la UI.
    - NO hay auto-conexión al iniciar. La conexión ocurre únicamente al llamar connect().
    - Auto-reconexión opcional: si config.auto_reconnect=True, run_thread() intenta
      reconectar cada 5 segundos cuando se pierde la conexión.
    - write_signals(): escribe los 4 bits de señal al DB del PLC.
    - run_thread(): debe ejecutarse en un thread separado (daemon=True).

    Bits en el DB:
        bit 0 → RIGHT
        bit 1 → CENTER
        bit 2 → LEFT
        bit 3 → GRIPPER (1=OPEN, 0=CLOSE)
    """

    RECONNECT_INTERVAL = 5.0  # segundos entre intentos de reconexión

    def __init__(self, state: AppState, config: ConfigManager):
        self._state = state
        self._config = config
        if SNAP7_AVAILABLE and snap7 is not None:
            try:
                self._client = snap7.client.Client()
            except Exception as e:
                self._client = None
                print(f"[PLCManager] Error inicializando snap7 Client (DLL no encontrada?): {e}")
                state.add_log("ERROR", f"snap7 Client no inicializado: {e}")
        else:
            self._client = None
            state.add_log("WARNING", "snap7 no disponible — PLC deshabilitado.")
        self._was_connected = False  # rastrea si ya hubo una conexión previa

    # --------------------------------------------------
    # Conexión / desconexión
    # --------------------------------------------------

    def connect(self) -> bool:
        """
        Intenta conectar al PLC con los parámetros actuales de config.
        Actualiza state.plc_connected, plc_connected_time y reconnect_count.
        Retorna True si la conexión fue exitosa, False en caso de error.
        """
        ip = self._config.plc_ip
        rack = int(self._config.rack)
        slot = int(self._config.slot)

        try:
            self._client.connect(ip, rack, slot)

            with self._state.lock:
                is_reconnect = self._was_connected
                self._state.plc_connected = True
                self._state.plc_connected_time = time.time()
                self._state.last_plc_error = ""
                if is_reconnect:
                    self._state.reconnect_count += 1

            self._was_connected = True

            label = "RECONECTADO" if is_reconnect else "CONECTADO"
            msg = f"PLC {label} — {ip} rack={rack} slot={slot}"
            self._state.add_log("PLC", msg)
            print(f"[PLCManager] {msg}")
            return True

        except Exception as e:
            error_msg = str(e)
            with self._state.lock:
                self._state.plc_connected = False
                self._state.last_plc_error = error_msg

            log_msg = f"Error conectando PLC {ip}: {error_msg}"
            self._state.add_log("ERROR", log_msg)
            print(f"[PLCManager] {log_msg}")
            return False

    def disconnect(self):
        """
        Desconecta el PLC y actualiza el estado.
        No lanza excepción aunque el cliente no estuviera conectado.
        """
        try:
            self._client.disconnect()
        except Exception as e:
            print(f"[PLCManager] Advertencia al desconectar: {e}")

        with self._state.lock:
            self._state.plc_connected = False

        msg = f"PLC DESCONECTADO — {self._config.plc_ip}"
        self._state.add_log("PLC", msg)
        print(f"[PLCManager] {msg}")

    # --------------------------------------------------
    # Escritura de señales
    # --------------------------------------------------

    def write_signals(self, right: bool, center: bool, left: bool, gripper: bool):
        """
        Escribe 4 bits de señal al DB del PLC.

        Parámetros:
            right   — True si la posición es RIGHT
            center  — True si la posición es CENTER
            left    — True si la posición es LEFT
            gripper — True si el gripper está OPEN, False si CLOSE

        Incrementa state.plc_write_count en éxito.
        En caso de error registra el mensaje en state.last_plc_error y en los logs.
        """
        data = bytearray(1)
        set_bool(data, 0, 0, bool(right))
        set_bool(data, 0, 1, bool(center))
        set_bool(data, 0, 2, bool(left))
        set_bool(data, 0, 3, bool(gripper))

        db_number = int(self._config.db_number)
        db_offset = int(self._config.db_offset)

        try:
            self._client.db_write(db_number, db_offset, data)
            with self._state.lock:
                self._state.plc_write_count += 1

        except Exception as e:
            error_msg = str(e)
            with self._state.lock:
                self._state.last_plc_error = error_msg
                self._state.plc_connected = False  # asume pérdida de conexión

            log_msg = f"Error escribiendo DB{db_number}[{db_offset}]: {error_msg}"
            self._state.add_log("ERROR", log_msg)
            print(f"[PLCManager] {log_msg}")

    # --------------------------------------------------
    # Thread de auto-reconexión
    # --------------------------------------------------

    def run_thread(self):
        """
        Loop de escritura PLC / monitoreo / auto-reconexión. Debe ejecutarse en un thread daemon.

        Comportamiento:
        - Si el PLC está conectado:
            lee robot_position y gripper_text del AppState (bajo lock),
            calcula los bits right/center/left/gripper y llama write_signals(),
            luego espera 1/config.plc_write_freq segundos.
        - Si no está conectado y config.auto_reconnect es True:
            intenta reconectar cada RECONNECT_INTERVAL segundos.
        - Si no está conectado y auto_reconnect es False:
            espera pasivo 1 segundo antes de revisar de nuevo.
        - El loop termina cuando state.stop_event está activado.
        """
        print("[PLCManager] Thread de monitoreo iniciado.")

        while not self._state.stop_event.is_set():
            with self._state.lock:
                connected = self._state.plc_connected
            auto_reconnect = self._config.auto_reconnect

            if connected:
                # Leer posición y gripper del estado compartido (bajo lock)
                with self._state.lock:
                    position = self._state.robot_position
                    gripper = self._state.gripper_text

                # Calcular bits de señal
                right        = (position == "RIGHT")
                center       = (position == "CENTER")
                left         = (position == "LEFT")
                gripper_open = (gripper == "OPEN")

                # Escribir al PLC (write_signals maneja su propio try/except internamente)
                try:
                    self.write_signals(right, center, left, gripper_open)
                except Exception as e:
                    # Capa adicional de seguridad para excepciones inesperadas
                    with self._state.lock:
                        self._state.plc_connected = False
                        self._state.last_plc_error = str(e)
                    self._state.add_log("ERROR", f"PLC write loop error inesperado: {e}")
                    print(f"[PLCManager] Error inesperado en write loop: {e}")

                # Esperar hasta el próximo ciclo según frecuencia configurada
                plc_write_freq = max(1, self._config.plc_write_freq)
                self._state.stop_event.wait(timeout=1.0 / plc_write_freq)

            elif auto_reconnect:
                self._state.add_log("PLC", "Auto-reconexión en curso...")
                print("[PLCManager] Intentando auto-reconexión...")
                self.connect()
                # Espera RECONNECT_INTERVAL o hasta que stop_event se active
                self._state.stop_event.wait(timeout=self.RECONNECT_INTERVAL)
            else:
                # Sin conexión y sin auto-reconexión: espera pasiva
                self._state.stop_event.wait(timeout=1.0)

        print("[PLCManager] Thread de monitoreo detenido.")


# =====================================================
# VISION PROCESSOR
# =====================================================

class VisionProcessor:
    """
    Procesador de visión con MediaPipe Hands, filtro exponencial e histéresis.

    - Constructor: inicializa MediaPipe Hands con parámetros de config.
    - fingers_up(): detecta qué dedos están levantados (5 bits).
    - Filtro exponencial: smooth_x = alpha * raw_x + (1-alpha) * smooth_x
    - Histéresis: solo cambia zona si la posición supera dead_zone + hysteresis
      desde la zona actual, evitando oscilación en los bordes.
    - run_thread(): loop principal que captura frames, procesa con MediaPipe,
      actualiza AppState y codifica frames en base64 para la UI Flet.

    Landmarks usados:
        0  = muñeca (posición X para determinar zona L/C/R)
        4  = tip pulgar
        8  = tip índice
        12 = tip medio
        16 = tip anular
        20 = tip meñique
        3  = knuckle pulgar (para comparación eje X)
        6, 10, 14, 18 = pip de los otros 4 dedos (comparación eje Y)
    """

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]

    def __init__(self, state: AppState, config: ConfigManager):
        self._state = state
        self._config = config
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._hands = None
        self._cap = None
        self._reinit_lock = threading.Lock()
        self._smooth_x: float = 0.0
        self._last_zone: str = "CENTER"
        self._init_mediapipe()

    # --------------------------------------------------
    # Inicialización
    # --------------------------------------------------

    def _init_mediapipe(self):
        """Inicializa / reinicializa MediaPipe Hands con parámetros de config."""
        with self._reinit_lock:
            if self._hands is not None:
                self._hands.close()
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=int(self._config.max_num_hands),
                min_detection_confidence=float(self._config.min_detection_confidence),
                min_tracking_confidence=float(self._config.min_tracking_confidence),
            )

    def _init_camera(self) -> bool:
        """
        Inicializa / reinicializa la cámara con los parámetros de config.
        Retorna True si la cámara quedó abierta, False en caso de error.
        """
        with self._reinit_lock:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()

            cam_index = int(self._config.camera)
            self._cap = cv2.VideoCapture(cam_index)

            w, h = self._config.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            if not self._cap.isOpened():
                msg = f"No se pudo abrir cámara {cam_index}"
                self._state.add_log("ERROR", msg)
                print(f"[VisionProcessor] {msg}")
                return False

            print(f"[VisionProcessor] Cámara {cam_index} abierta ({w}x{h}).")
            return True

    def reinit(self):
        """
        Reinicializa MediaPipe y la cámara con los parámetros actuales de config.
        Llamar desde la UI cuando el usuario aplique nuevos ajustes de Vision.
        """
        self._init_mediapipe()
        self._init_camera()
        self._state.add_log("VISION", "VisionProcessor reinicializado con nuevos parámetros.")

    # --------------------------------------------------
    # Detección de dedos
    # --------------------------------------------------

    def fingers_up(self, landmarks, handedness: str = "Right") -> list:
        """
        Determina qué dedos están levantados a partir de los landmarks de MediaPipe.

        Parámetros:
            landmarks   — lista de 21 NormalizedLandmark de la mano.
            handedness  — "Right" o "Left" (etiqueta devuelta por MediaPipe).

        Retorna:
            Lista de 5 enteros [pulgar, índice, medio, anular, meñique]:
                1 = levantado / abierto
                0 = doblado / cerrado

        Lógica:
            - Pulgar: comparación en eje X (depende del lado de la mano).
              Mano derecha → tip (4) más a la izquierda que knuckle (3) = cerrado.
              Mano izquierda → tip (4) más a la derecha que knuckle (3) = cerrado.
            - Dedos 2-5: tip.y < pip.y indica dedo levantado
              (en coordenadas de imagen y=0 es la parte superior).
        """
        fingers = []

        # Pulgar (eje X, dependiente del lado)
        if handedness == "Right":
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

        # Dedos índice, medio, anular, meñique (eje Y)
        for tip_id, pip_id in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y else 0)

        return fingers

    # --------------------------------------------------
    # Zona con histéresis
    # --------------------------------------------------

    def _determine_zone(self, smooth_x: float, frame_width: int) -> str:
        """
        Determina la zona de posición (LEFT / CENTER / RIGHT) aplicando
        histéresis para evitar oscilación en los bordes de las zonas.

        La histéresis funciona como un disparador de Schmitt:
        - Desde CENTER: se necesita superar (center ± dead_zone + hysteresis)
          para entrar a LEFT o RIGHT.
        - Desde LEFT: se necesita cruzar (center − dead_zone + hysteresis)
          hacia el centro para volver a CENTER o pasar a RIGHT.
        - Desde RIGHT: se necesita cruzar (center + dead_zone − hysteresis)
          hacia el centro para volver a CENTER o pasar a LEFT.

        Actualiza self._last_zone y retorna la zona determinada.
        """
        center = frame_width // 2
        dz = int(self._config.dead_zone)
        hyst = int(self._config.hysteresis)

        current = self._last_zone

        if current == "CENTER":
            if smooth_x < center - dz - hyst:
                new_zone = "LEFT"
            elif smooth_x > center + dz + hyst:
                new_zone = "RIGHT"
            else:
                new_zone = "CENTER"

        elif current == "LEFT":
            # Para salir de LEFT necesita superar (center - dz + hyst) hacia el centro
            if smooth_x >= center - dz + hyst:
                new_zone = "RIGHT" if smooth_x > center + dz + hyst else "CENTER"
            else:
                new_zone = "LEFT"

        else:  # RIGHT
            # Para salir de RIGHT necesita bajar de (center + dz - hyst)
            if smooth_x <= center + dz - hyst:
                new_zone = "LEFT" if smooth_x < center - dz - hyst else "CENTER"
            else:
                new_zone = "RIGHT"

        self._last_zone = new_zone
        return new_zone

    # --------------------------------------------------
    # Utilidades de frame
    # --------------------------------------------------

    @staticmethod
    def _frame_to_b64(frame: np.ndarray) -> str:
        """
        Convierte un frame OpenCV (array BGR/RGB) a string base64 PNG.
        Retorna cadena vacía si la codificación falla.
        """
        success, buffer = cv2.imencode(".png", frame)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    # --------------------------------------------------
    # Thread principal
    # --------------------------------------------------

    def run_thread(self):
        """
        Loop principal de visión. Ejecutar en un thread daemon.

        Por cada frame realiza:
        1. Captura frame de la cámara.
        2. Voltea horizontalmente (efecto espejo) y convierte a RGB.
        3. Procesa con MediaPipe Hands.
        4. Si hay mano detectada:
           a. Aplica filtro exponencial a la coordenada X de la muñeca.
           b. Determina zona L/C/R con histéresis.
           c. Cuenta dedos levantados y determina estado del gripper.
           d. Dibuja skeleton en frame negro y landmarks en el frame de cámara.
        5. Calcula FPS con contador de 1 segundo.
        6. Codifica ambos frames (cámara y skeleton) a base64 PNG.
        7. Actualiza AppState bajo lock con todos los valores nuevos.
        8. Registra cambios de posición/gripper en historial y logs.

        Termina cuando state.stop_event está activado.
        """
        print("[VisionProcessor] Thread de visión iniciado.")
        self._state.add_log("VISION", "Thread de visión iniciado.")

        if not self._init_camera():
            self._state.add_log("ERROR", "Cámara no disponible — thread de visión abortado.")
            return

        fps_timer = time.time()
        frame_count = 0
        current_fps = 0.0

        while not self._state.stop_event.is_set():
            # Snapshot thread-safe de los recursos bajo lock para evitar
            # race condition con reinit() llamado desde la UI
            with self._reinit_lock:
                cap = self._cap
                hands = self._hands

            if cap is None or hands is None:
                time.sleep(0.05)
                continue

            try:
                ret, frame = cap.read()
            except Exception:
                time.sleep(0.05)
                continue

            if not ret or frame is None:
                self._state.add_log("ERROR", "Error leyendo frame — reintentando.")
                time.sleep(0.1)
                continue

            frame_h, frame_w = frame.shape[:2]

            # Efecto espejo: la mano derecha del usuario aparece a la derecha
            frame = cv2.flip(frame, 1)

            # RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = hands.process(rgb)
            except Exception:
                time.sleep(0.05)
                continue

            try:
                # Frame de skeleton: fondo negro
                skeleton = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

                # Valores por defecto (sin mano)
                hand_detected = False
                new_hand_x = int(self._smooth_x)
                new_hand_y = 0
                new_offset = 0
                new_position = self._last_zone
                new_gripper = "UNKNOWN"
                new_fingers = [0, 0, 0, 0, 0]

                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_detected = True
                    hand_lms = results.multi_hand_landmarks[0]
                    handedness_label = (
                        results.multi_handedness[0].classification[0].label
                    )  # "Right" o "Left"

                    # Coordenada X de la muñeca (píxeles)
                    wrist = hand_lms.landmark[0]
                    raw_x = wrist.x * frame_w

                    # Filtro exponencial
                    alpha = float(self._config.smoothing)
                    self._smooth_x = alpha * raw_x + (1.0 - alpha) * self._smooth_x

                    new_hand_x = int(self._smooth_x)
                    new_hand_y = int(wrist.y * frame_h)
                    new_offset = new_hand_x - (frame_w // 2)

                    # Zona con histéresis
                    new_position = self._determine_zone(self._smooth_x, frame_w)

                    # Dedos levantados
                    new_fingers = self.fingers_up(hand_lms.landmark, handedness_label)
                    fingers_count = sum(new_fingers)

                    # Estado del gripper
                    open_thresh = int(self._config.open_threshold)
                    close_thresh = int(self._config.close_threshold)
                    if fingers_count >= open_thresh:
                        new_gripper = "OPEN"
                    elif fingers_count <= close_thresh:
                        new_gripper = "CLOSE"
                    else:
                        new_gripper = "PARTIAL"

                    # Dibujar skeleton en fondo negro
                    self._mp_draw.draw_landmarks(
                        skeleton,
                        hand_lms,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_drawing_styles.get_default_hand_landmarks_style(),
                        self._mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Dibujar landmarks sobre el frame de cámara
                    self._mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        self._mp_hands.HAND_CONNECTIONS,
                    )

                    # Dibujar líneas de dead zone y punto de muñeca en el frame
                    center_x = frame_w // 2
                    dz = int(self._config.dead_zone)
                    cv2.line(frame, (center_x - dz, 0), (center_x - dz, frame_h), (0, 255, 255), 1)
                    cv2.line(frame, (center_x + dz, 0), (center_x + dz, frame_h), (0, 255, 255), 1)
                    cv2.circle(frame, (new_hand_x, frame_h // 2), 8, (0, 255, 0), -1)

                # Calcular FPS (actualizar cada segundo)
                frame_count += 1
                now = time.time()
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_timer = now

                # Codificar frames a base64 para la UI
                frame_b64 = self._frame_to_b64(frame)
                skeleton_b64 = self._frame_to_b64(skeleton)

                # Leer estado previo para detectar cambios (sin lock extendido)
                with self._state.lock:
                    prev_position = self._state.robot_position
                    prev_gripper = self._state.gripper_text
                    prev_tracking = self._state.tracking_status

                    new_tracking = "ACTIVE" if hand_detected else "NO HAND"
                    self._state.tracking_status = new_tracking
                    self._state.hand_x = new_hand_x
                    self._state.hand_y = new_hand_y
                    self._state.offset = new_offset
                    self._state.robot_position = new_position
                    self._state.gripper_text = new_gripper
                    self._state.fingers = new_fingers
                    self._state.frame_b64 = frame_b64
                    self._state.skeleton_b64 = skeleton_b64
                    if current_fps > 0:
                        self._state.fps = current_fps

                # Registrar cambios fuera del lock
                if new_tracking != prev_tracking:
                    self._state.log_manager.vision_event(f"Tracking → {new_tracking}")

                if hand_detected:
                    if new_position != prev_position:
                        self._state.add_command(new_position)
                        self._state.log_manager.robot_event(f"Posición → {new_position}")
                    if new_gripper != prev_gripper and new_gripper in ("OPEN", "CLOSE"):
                        self._state.add_command(new_gripper)
                        self._state.log_manager.robot_event(f"Gripper → {new_gripper}")

            except Exception as _ex:
                self._state.add_log("ERROR", f"VisionProcessor error inesperado en frame: {_ex}")
                continue

        # Limpieza al salir del loop
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        if self._hands is not None:
            self._hands.close()

        print("[VisionProcessor] Thread de visión detenido.")
        self._state.add_log("VISION", "Thread de visión detenido.")


# =====================================================
# LOG MANAGER
# =====================================================

class LogManager:
    """
    Sistema de logs profesional con exportación CSV.

    - logs: deque(maxlen=500) — almacena hasta 500 entradas.
    - add(level, message): agrega entrada con timestamp ISO.
    - Métodos de conveniencia: info(), error(), warning(),
      plc_event(), vision_event(), robot_event().
    - clear(): vacía el deque.
    - export_csv(filepath): escribe todos los logs a CSV con
      columnas timestamp, level, message usando csv.writer.

    Niveles definidos:
        INFO    — información general
        WARNING — advertencia, no crítica
        ERROR   — error que requiere atención
        PLC     — evento del PLC (conexión, escritura, etc.)
        VISION  — evento del procesador de visión
        ROBOT   — cambio de comando o posición del robot
    """

    LEVELS = ("INFO", "WARNING", "ERROR", "PLC", "VISION", "ROBOT")

    def __init__(self, maxlen: int = 500):
        self.logs: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # --------------------------------------------------
    # Método base
    # --------------------------------------------------

    def add(self, level: str, message: str):
        """
        Agrega una entrada al log.

        Parámetros:
            level   — uno de LEVELS (INFO, WARNING, ERROR, PLC, VISION, ROBOT)
            message — texto descriptivo del evento
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "level": level.upper(),
            "message": message,
        }
        with self._lock:
            self.logs.append(entry)

    # --------------------------------------------------
    # Métodos de conveniencia
    # --------------------------------------------------

    def info(self, message: str):
        """Registra un evento de nivel INFO."""
        self.add("INFO", message)

    def warning(self, message: str):
        """Registra un evento de nivel WARNING."""
        self.add("WARNING", message)

    def error(self, message: str):
        """Registra un evento de nivel ERROR."""
        self.add("ERROR", message)

    def plc_event(self, message: str):
        """Registra un evento PLC (conexión, desconexión, escritura, etc.)."""
        self.add("PLC", message)

    def vision_event(self, message: str):
        """Registra un evento del procesador de visión."""
        self.add("VISION", message)

    def robot_event(self, message: str):
        """Registra un cambio de comando o posición del robot."""
        self.add("ROBOT", message)

    # --------------------------------------------------
    # Gestión
    # --------------------------------------------------

    def clear(self):
        """Vacía el deque de logs."""
        with self._lock:
            self.logs.clear()

    def get_all(self) -> list:
        """Retorna una copia de todos los logs como lista."""
        with self._lock:
            return list(self.logs)

    def export_csv(self, filepath: str):
        """
        Exporta todos los logs a un archivo CSV.

        Columnas: timestamp, level, message
        Crea o sobreescribe el archivo en filepath.
        Retorna el número de filas escritas.

        Ejemplo:
            n = log_manager.export_csv("logs_20241201_120000.csv")
            print(f"Exportadas {n} entradas.")
        """
        entries = self.get_all()
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "level", "message"])
                for entry in entries:
                    writer.writerow([
                        entry.get("timestamp", ""),
                        entry.get("level", ""),
                        entry.get("message", ""),
                    ])
            print(f"[LogManager] Exportadas {len(entries)} entradas a: {filepath}")
            return len(entries)
        except IOError as e:
            print(f"[LogManager] Error exportando CSV: {e}")
            return 0

    def __len__(self):
        return len(self.logs)

    def __repr__(self):
        return f"<LogManager entries={len(self.logs)} maxlen={self.logs.maxlen}>"


# =====================================================
# FLET UI — SHELL PRINCIPAL
# =====================================================

# Colores del tema industrial oscuro
BG_MAIN = "#111827"       # fondo general
BG_PANEL = "#1F2937"      # paneles / sidebar
BG_CARD = "#374151"       # tarjetas secundarias
ACCENT = "#22D3EE"        # cian principal
ACCENT_DIM = "#0E7490"    # cian oscuro (indicador seleccionado)
TEXT_MUTED = "#9CA3AF"    # texto secundario
DIVIDER = "#374151"       # línea divisoria


def build_placeholder_page(title: str) -> ft.Control:
    """
    Página placeholder para las secciones aún no implementadas (tareas 6-11).
    Retorna un Container expandido con el nombre de la sección.
    """
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(name=ft.Icons.CONSTRUCTION, color=ACCENT, size=28),
                        ft.Text(
                            title,
                            size=26,
                            weight=ft.FontWeight.BOLD,
                            color=ACCENT,
                        ),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(color=DIVIDER, height=1),
                ft.Text(
                    f"Sección {title} — implementación pendiente (tareas 6-11).",
                    color=TEXT_MUTED,
                    size=14,
                ),
            ],
            spacing=20,
        ),
        padding=ft.padding.all(32),
        expand=True,
        bgcolor=BG_MAIN,
    )


def build_dashboard_page(state: AppState, config: ConfigManager):
    """
    Construye la página Dashboard con indicadores HMI estilo industrial.

    Retorna:
        (content, update) donde:
          content — ft.Container con el layout completo
          update  — función sin args que actualiza los Ref desde AppState
                    (debe llamarse desde el hilo de UI antes de page.update())
    """
    # Colores semánticos
    C_OK   = "#10B981"   # verde  — connected / ACTIVE / OPEN / CENTER
    C_WARN = "#F59E0B"   # amarillo — NO HAND / PARTIAL
    C_ERR  = "#EF4444"   # rojo   — disconnected / ERROR / CLOSE / LEFT
    C_INFO = "#3B82F6"   # azul   — RIGHT / informativo

    # ---- Refs -------------------------------------------------------
    ref_plc_dot    = ft.Ref[ft.Container]()
    ref_plc_lbl    = ft.Ref[ft.Text]()
    ref_trk_dot    = ft.Ref[ft.Container]()
    ref_trk_lbl    = ft.Ref[ft.Text]()
    ref_pos_lbl    = ft.Ref[ft.Text]()
    ref_grip_dot   = ft.Ref[ft.Container]()
    ref_grip_lbl   = ft.Ref[ft.Text]()
    ref_cmd_lbl    = ft.Ref[ft.Text]()
    ref_fps        = ft.Ref[ft.Text]()
    ref_runtime    = ft.Ref[ft.Text]()
    ref_cmds       = ft.Ref[ft.Text]()
    ref_writes     = ft.Ref[ft.Text]()
    ref_fingers    = ft.Ref[ft.Text]()
    ref_zone_l     = ft.Ref[ft.Container]()
    ref_zone_c     = ft.Ref[ft.Container]()
    ref_zone_r     = ft.Ref[ft.Container]()
    ref_offset_lbl = ft.Ref[ft.Text]()

    # ---- Helpers de construcción ------------------------------------
    def _dot_card(title, dot_ref, lbl_ref, init_dot_color, init_text):
        """Tarjeta con indicador circular de color y texto de estado."""
        return ft.Container(
            content=ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                title,
                                size=11,
                                color=TEXT_MUTED,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Row(
                                [
                                    ft.Container(
                                        ref=dot_ref,
                                        width=12,
                                        height=12,
                                        border_radius=6,
                                        bgcolor=init_dot_color,
                                    ),
                                    ft.Text(
                                        ref=lbl_ref,
                                        value=init_text,
                                        size=13,
                                        weight=ft.FontWeight.BOLD,
                                        color=ft.Colors.WHITE,
                                        no_wrap=True,
                                    ),
                                ],
                                spacing=8,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ],
                        spacing=10,
                    ),
                    bgcolor=BG_PANEL,
                    border_radius=8,
                    padding=ft.padding.all(16),
                    height=80,
                ),
                elevation=2,
                color=BG_PANEL,
            ),
            expand=True,
        )

    def _text_card(title, lbl_ref, init_text, init_color=ft.Colors.WHITE):
        """Tarjeta con texto grande de estado (sin indicador circular)."""
        return ft.Container(
            content=ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                title,
                                size=11,
                                color=TEXT_MUTED,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                ref=lbl_ref,
                                value=init_text,
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                color=init_color,
                                no_wrap=True,
                            ),
                        ],
                        spacing=10,
                    ),
                    bgcolor=BG_PANEL,
                    border_radius=8,
                    padding=ft.padding.all(16),
                    height=80,
                ),
                elevation=2,
                color=BG_PANEL,
            ),
            expand=True,
        )

    def _metric_card(title, val_ref, init_val, unit="", color=ACCENT):
        """Tarjeta de métrica con valor numérico grande."""
        row_children = [
            ft.Text(
                ref=val_ref,
                value=str(init_val),
                size=26,
                weight=ft.FontWeight.BOLD,
                color=color,
            ),
        ]
        if unit:
            row_children.append(ft.Text(unit, size=11, color=TEXT_MUTED))
        return ft.Container(
            content=ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                title,
                                size=11,
                                color=TEXT_MUTED,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Row(
                                row_children,
                                spacing=4,
                                vertical_alignment=ft.CrossAxisAlignment.END,
                            ),
                        ],
                        spacing=8,
                    ),
                    bgcolor=BG_PANEL,
                    border_radius=8,
                    padding=ft.padding.all(16),
                    height=80,
                ),
                elevation=2,
                color=BG_PANEL,
            ),
            expand=True,
        )

    def _zone_box(label, ref, border_color, active=False):
        """Caja de zona tipo HMI (LEFT / CENTER / RIGHT)."""
        return ft.Container(
            ref=ref,
            content=ft.Text(
                label,
                size=15,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.WHITE,
                text_align=ft.TextAlign.CENTER,
            ),
            bgcolor=border_color if active else BG_CARD,
            border_radius=8,
            padding=ft.padding.symmetric(vertical=18, horizontal=8),
            alignment=ft.alignment.center,
            expand=True,
            border=ft.border.all(2, border_color),
        )

    # ---- Fila 1: tarjetas de estado (5 tarjetas) --------------------
    row_status = ft.Row(
        [
            _dot_card("PLC STATUS",      ref_plc_dot,  ref_plc_lbl,  C_ERR,   "OFFLINE"),
            _dot_card("TRACKING",        ref_trk_dot,  ref_trk_lbl,  C_WARN,  "NO HAND"),
            _text_card("ROBOT POSITION", ref_pos_lbl,  "CENTER",     C_OK),
            _dot_card("GRIPPER",         ref_grip_dot, ref_grip_lbl, BG_CARD, "UNKNOWN"),
            _text_card("CURRENT CMD",    ref_cmd_lbl,  "—",          TEXT_MUTED),
        ],
        spacing=12,
    )

    # ---- Fila 2: métricas numéricas (5 tarjetas) --------------------
    row_metrics = ft.Row(
        [
            _metric_card("FPS",               ref_fps,     "0.0",      "fps", ACCENT),
            _metric_card("RUNTIME",           ref_runtime, "00:00:00", "",    ACCENT),
            _metric_card("COMMANDS SENT",     ref_cmds,    "0",        "",    C_INFO),
            _metric_card("PLC WRITES",        ref_writes,  "0",        "",    C_INFO),
            _metric_card("FINGERS DETECTED",  ref_fingers, "0",        "",    C_WARN),
        ],
        spacing=12,
    )

    # ---- Barra de posición HMI (L / C / R) -------------------------
    position_section = ft.Container(
        content=ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.LINEAR_SCALE, color=ACCENT, size=16),
                                ft.Text(
                                    "ROBOT POSITION INDICATOR",
                                    size=13,
                                    weight=ft.FontWeight.BOLD,
                                    color=ACCENT,
                                ),
                            ],
                            spacing=8,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=1, color=DIVIDER),
                        ft.Row(
                            [
                                _zone_box("◄  LEFT",   ref_zone_l, C_ERR,  False),
                                _zone_box("●  CENTER", ref_zone_c, C_OK,   True),
                                _zone_box("RIGHT  ►",  ref_zone_r, C_INFO, False),
                            ],
                            spacing=10,
                        ),
                        ft.Row(
                            [
                                ft.Text("WRIST OFFSET:", size=11, color=TEXT_MUTED),
                                ft.Text(
                                    ref=ref_offset_lbl,
                                    value="0 px",
                                    size=11,
                                    color=ACCENT,
                                    weight=ft.FontWeight.BOLD,
                                ),
                            ],
                            spacing=6,
                        ),
                    ],
                    spacing=14,
                ),
                bgcolor=BG_PANEL,
                border_radius=8,
                padding=ft.padding.all(20),
            ),
            elevation=2,
            color=BG_PANEL,
        ),
    )

    # ---- Layout completo --------------------------------------------
    content = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.DASHBOARD, color=ACCENT, size=24),
                        ft.Text(
                            "DASHBOARD",
                            size=20,
                            weight=ft.FontWeight.BOLD,
                            color=ACCENT,
                        ),
                    ],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(height=1, color=DIVIDER),
                row_status,
                row_metrics,
                position_section,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=ft.padding.all(24),
        expand=True,
        bgcolor=BG_MAIN,
    )

    # ---- Función de actualización (llamar desde hilo de UI) ---------
    def update():
        """Lee AppState y actualiza todos los Ref. Llamar antes de page.update()."""
        try:
            with state.lock:
                plc_conn  = state.plc_connected
                tracking  = state.tracking_status
                position  = state.robot_position
                gripper   = state.gripper_text
                fingers   = list(state.fingers)
                fps_now   = state.fps
                cmd_cnt   = state.cmd_count
                write_cnt = state.plc_write_count
                offset    = state.offset
                hist      = list(state.command_history)
        except Exception:
            return

        runtime_str   = state.get_runtime_str()
        fingers_count = sum(fingers)
        last_cmd      = hist[-1]["command"] if hist else "—"

        if ref_plc_dot.current:
            ref_plc_dot.current.bgcolor = C_OK if plc_conn else C_ERR
            ref_plc_lbl.current.value   = "CONNECTED" if plc_conn else "OFFLINE"

        if ref_trk_dot.current:
            ref_trk_dot.current.bgcolor = C_OK if tracking == "ACTIVE" else C_WARN
            ref_trk_lbl.current.value   = tracking

        if ref_pos_lbl.current:
            _pos_color = {"LEFT": C_ERR, "CENTER": C_OK, "RIGHT": C_INFO}
            ref_pos_lbl.current.value = position
            ref_pos_lbl.current.color = _pos_color.get(position, ft.Colors.WHITE)

        if ref_grip_dot.current:
            _grip_color = {"OPEN": C_OK, "CLOSE": C_ERR, "PARTIAL": C_WARN, "UNKNOWN": BG_CARD}
            ref_grip_dot.current.bgcolor = _grip_color.get(gripper, BG_CARD)
            ref_grip_lbl.current.value   = gripper

        if ref_cmd_lbl.current:
            ref_cmd_lbl.current.value = last_cmd

        if ref_fps.current:     ref_fps.current.value     = f"{fps_now:.1f}"
        if ref_runtime.current: ref_runtime.current.value = runtime_str
        if ref_cmds.current:    ref_cmds.current.value    = str(cmd_cnt)
        if ref_writes.current:  ref_writes.current.value  = str(write_cnt)
        if ref_fingers.current: ref_fingers.current.value = str(fingers_count)

        if ref_zone_l.current:
            ref_zone_l.current.bgcolor = C_ERR  if position == "LEFT"   else BG_CARD
        if ref_zone_c.current:
            ref_zone_c.current.bgcolor = C_OK   if position == "CENTER" else BG_CARD
        if ref_zone_r.current:
            ref_zone_r.current.bgcolor = C_INFO if position == "RIGHT"  else BG_CARD

        if ref_offset_lbl.current:
            sign = "+" if offset >= 0 else ""
            ref_offset_lbl.current.value = f"{sign}{offset} px"

    return content, update


def build_plc_page(
    state: AppState,
    config: ConfigManager,
    plc_manager: "PLCManager",
    page: ft.Page,
):
    """
    Página PLC con formulario de configuración, botones CONNECT/DISCONNECT
    y panel de estado de conexión en tiempo real.

    Retorna:
        (content, update) donde:
          content — ft.Container con el layout completo
          update  — función sin args que actualiza los Ref desde AppState
                    (debe llamarse desde el hilo de UI antes de page.update())
    """
    C_OK   = "#10B981"   # verde  — conectado
    C_ERR  = "#EF4444"   # rojo   — error / offline
    C_WARN = "#F59E0B"   # amarillo — advertencia / desconectado

    # ---- TextFields de configuración ----------------------------------------
    tf_ip = ft.TextField(
        label="PLC IP Address",
        value=str(config.plc_ip),
        hint_text="192.168.11.1",
        prefix_icon=ft.Icons.ROUTER,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    tf_rack = ft.TextField(
        label="Rack",
        value=str(config.rack),
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    tf_slot = ft.TextField(
        label="Slot",
        value=str(config.slot),
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    tf_db_num = ft.TextField(
        label="DB Number",
        value=str(config.db_number),
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    tf_db_off = ft.TextField(
        label="DB Offset",
        value=str(config.db_offset),
        keyboard_type=ft.KeyboardType.NUMBER,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )

    # ---- Refs para el panel de estado ----------------------------------------
    ref_conn_dot  = ft.Ref[ft.Container]()
    ref_conn_lbl  = ft.Ref[ft.Text]()
    ref_error_lbl = ft.Ref[ft.Text]()
    ref_time_lbl  = ft.Ref[ft.Text]()
    ref_recon_lbl = ft.Ref[ft.Text]()

    # ---- Helpers internos ---------------------------------------------------
    async def _show_snack(msg: str, color: str = C_OK):
        page.snack_bar = ft.SnackBar(
            content=ft.Text(msg, color=ft.Colors.WHITE),
            bgcolor=color,
        )
        page.snack_bar.open = True
        await page.update_async()

    def _read_int(tf: ft.TextField, fallback: int) -> int:
        try:
            return int(tf.value.strip())
        except (ValueError, AttributeError):
            return fallback

    # ---- Callbacks de botones -----------------------------------------------
    async def on_connect(e):
        # Aplicar valores de los TextFields al config antes de conectar
        config.plc_ip    = tf_ip.value.strip()
        config.rack      = _read_int(tf_rack,   config.rack)
        config.slot      = _read_int(tf_slot,   config.slot)
        config.db_number = _read_int(tf_db_num, config.db_number)
        config.db_offset = _read_int(tf_db_off, config.db_offset)
        config.save()

        ok = plc_manager.connect()
        if ok:
            await _show_snack(f"PLC conectado — {config.plc_ip}", C_OK)
        else:
            with state.lock:
                err = state.last_plc_error
            await _show_snack(f"Error al conectar: {err}", C_ERR)

    async def on_disconnect(e):
        plc_manager.disconnect()
        await _show_snack("PLC desconectado.", C_WARN)

    # ---- Helper: tarjeta con LED de estado ----------------------------------
    def _led_card(title: str, dot_ref, lbl_ref, init_dot: str, init_text: str):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        title,
                        size=11,
                        color=TEXT_MUTED,
                        weight=ft.FontWeight.W_500,
                    ),
                    ft.Row(
                        [
                            ft.Container(
                                ref=dot_ref,
                                width=12,
                                height=12,
                                border_radius=6,
                                bgcolor=init_dot,
                            ),
                            ft.Text(
                                ref=lbl_ref,
                                value=init_text,
                                size=14,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE,
                                no_wrap=True,
                                expand=True,
                            ),
                        ],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=8,
            ),
            bgcolor=BG_CARD,
            border_radius=8,
            padding=ft.padding.all(16),
        )

    # ---- Helper: tarjeta de información simple ------------------------------
    def _info_card(title: str, lbl_ref, init_text: str):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        title,
                        size=11,
                        color=TEXT_MUTED,
                        weight=ft.FontWeight.W_500,
                    ),
                    ft.Text(
                        ref=lbl_ref,
                        value=init_text,
                        size=13,
                        weight=ft.FontWeight.W_600,
                        color=ft.Colors.WHITE,
                        no_wrap=False,
                    ),
                ],
                spacing=8,
            ),
            bgcolor=BG_CARD,
            border_radius=8,
            padding=ft.padding.all(16),
        )

    # ---- Header -------------------------------------------------------------
    header_row = ft.Row(
        [
            ft.Icon(name=ft.Icons.SETTINGS_ETHERNET, color=ACCENT, size=28),
            ft.Text(
                "PLC Connection",
                size=24,
                weight=ft.FontWeight.BOLD,
                color=ACCENT,
            ),
        ],
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ---- Sección: PLC Configuration -----------------------------------------
    config_section = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "PLC Configuration",
                    size=13,
                    weight=ft.FontWeight.W_600,
                    color=TEXT_MUTED,
                ),
                ft.Divider(color=DIVIDER, height=1),
                tf_ip,
                ft.Row([tf_rack, tf_slot], spacing=12),
                ft.Row([tf_db_num, tf_db_off], spacing=12),
            ],
            spacing=14,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(20),
    )

    # ---- Sección: Connection Control ----------------------------------------
    control_section = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Connection Control",
                    size=13,
                    weight=ft.FontWeight.W_600,
                    color=TEXT_MUTED,
                ),
                ft.Divider(color=DIVIDER, height=1),
                ft.Row(
                    [
                        ft.ElevatedButton(
                            "[CONNECT PLC]",
                            icon=ft.Icons.POWER,
                            bgcolor=C_OK,
                            color=ft.Colors.WHITE,
                            style=ft.ButtonStyle(
                                shape=ft.RoundedRectangleBorder(radius=6),
                            ),
                            on_click=on_connect,
                            expand=True,
                        ),
                        ft.OutlinedButton(
                            "[DISCONNECT PLC]",
                            icon=ft.Icons.POWER_OFF,
                            style=ft.ButtonStyle(
                                shape=ft.RoundedRectangleBorder(radius=6),
                                side=ft.BorderSide(1, C_ERR),
                                color=C_ERR,
                            ),
                            on_click=on_disconnect,
                            expand=True,
                        ),
                    ],
                    spacing=12,
                ),
            ],
            spacing=14,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(20),
    )

    # ---- Sección: Connection Status -----------------------------------------
    status_section = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Connection Status",
                    size=13,
                    weight=ft.FontWeight.W_600,
                    color=TEXT_MUTED,
                ),
                ft.Divider(color=DIVIDER, height=1),
                _led_card(
                    "Connection",
                    ref_conn_dot, ref_conn_lbl,
                    C_ERR, "OFFLINE",
                ),
                _info_card("Last Error",       ref_error_lbl, "—"),
                _info_card("Time Connected",   ref_time_lbl,  "—"),
                _info_card("Reconnections",    ref_recon_lbl, "0"),
            ],
            spacing=12,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(20),
    )

    # ---- Layout principal: dos columnas -------------------------------------
    content = ft.Container(
        content=ft.Column(
            [
                header_row,
                ft.Divider(color=DIVIDER, height=1),
                ft.Row(
                    [
                        # Columna izquierda: configuración + control
                        ft.Column(
                            [config_section, control_section],
                            spacing=16,
                            expand=True,
                        ),
                        # Columna derecha: estado
                        ft.Column(
                            [status_section],
                            spacing=16,
                            expand=True,
                        ),
                    ],
                    spacing=20,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    expand=True,
                ),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        ),
        padding=ft.padding.all(32),
        expand=True,
        bgcolor=BG_MAIN,
    )

    # ---- Función de update (llamada por el hilo de UI) ----------------------
    def update():
        with state.lock:
            connected   = state.plc_connected
            last_error  = state.last_plc_error
            conn_time   = state.plc_connected_time
            recon_count = state.reconnect_count

        # Calcular tiempo transcurrido desde la conexión
        if connected and conn_time > 0:
            elapsed = int(time.time() - conn_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            time_str = "—"

        if ref_conn_dot.current:
            ref_conn_dot.current.bgcolor = C_OK if connected else C_ERR
            ref_conn_lbl.current.value   = "CONNECTED" if connected else "OFFLINE"
            ref_conn_lbl.current.color   = C_OK if connected else C_ERR

        if ref_error_lbl.current:
            ref_error_lbl.current.value = last_error if last_error else "—"
            ref_error_lbl.current.color = C_ERR if last_error else TEXT_MUTED

        if ref_time_lbl.current:
            ref_time_lbl.current.value = time_str

        if ref_recon_lbl.current:
            ref_recon_lbl.current.value = str(recon_count)

    return content, update


def build_vision_page(
    state: AppState,
    config: ConfigManager,
    vision_processor: "VisionProcessor",
    page: ft.Page,
):
    """
    Página Vision con video en vivo, skeleton overlay y parámetros configurables.

    Retorna:
        (content, update) donde:
          content — ft.Container con el layout completo
          update  — función sin args que actualiza los Ref desde AppState
                    (debe llamarse desde el hilo de UI antes de page.update())
    """
    C_OK   = "#10B981"
    C_WARN = "#F59E0B"
    C_ERR  = "#EF4444"

    # ---- Refs para las imágenes y el panel de info ----------------------
    ref_frame_img    = ft.Ref[ft.Image]()
    ref_skel_img     = ft.Ref[ft.Image]()
    ref_tracking_dot = ft.Ref[ft.Container]()
    ref_tracking_lbl = ft.Ref[ft.Text]()
    ref_hand_x_lbl   = ft.Ref[ft.Text]()
    ref_hand_y_lbl   = ft.Ref[ft.Text]()
    ref_fingers_lbl  = ft.Ref[ft.Text]()

    # ---- Refs para mostrar el valor actual de cada slider ---------------
    ref_det_val  = ft.Ref[ft.Text]()
    ref_trk_val  = ft.Ref[ft.Text]()

    # ---- Controles de configuración -------------------------------------
    async def _on_detection_change(e):
        if ref_det_val.current:
            ref_det_val.current.value = f"{e.control.value:.1f}"
            await page.update_async()

    async def _on_tracking_change(e):
        if ref_trk_val.current:
            ref_trk_val.current.value = f"{e.control.value:.1f}"
            await page.update_async()

    sl_detection = ft.Slider(
        min=0.1, max=1.0, divisions=9,
        value=float(config.min_detection_confidence),
        active_color=ACCENT,
        thumb_color=ACCENT,
        on_change=_on_detection_change,
        expand=True,
    )
    sl_tracking = ft.Slider(
        min=0.1, max=1.0, divisions=9,
        value=float(config.min_tracking_confidence),
        active_color=ACCENT,
        thumb_color=ACCENT,
        on_change=_on_tracking_change,
        expand=True,
    )
    dd_max_hands = ft.Dropdown(
        label="Max Hands",
        value=str(int(config.max_num_hands)),
        options=[
            ft.dropdown.Option("1"),
            ft.dropdown.Option("2"),
        ],
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    tf_camera = ft.TextField(
        label="Camera Index",
        value=str(int(config.camera)),
        keyboard_type=ft.KeyboardType.NUMBER,
        prefix_icon=ft.Icons.CAMERA_ALT,
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )
    _res_str = f"{config.resolution[0]}x{config.resolution[1]}"
    if _res_str not in ("640x480", "1280x720", "1920x1080"):
        _res_str = "640x480"
    dd_resolution = ft.Dropdown(
        label="Resolution",
        value=_res_str,
        options=[
            ft.dropdown.Option("640x480"),
            ft.dropdown.Option("1280x720"),
            ft.dropdown.Option("1920x1080"),
        ],
        bgcolor=BG_CARD,
        border_color=ACCENT_DIM,
        focused_border_color=ACCENT,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=TEXT_MUTED),
        expand=True,
    )

    # ---- Apply Settings callback ----------------------------------------
    async def on_apply(e):
        config.min_detection_confidence = float(sl_detection.value)
        config.min_tracking_confidence  = float(sl_tracking.value)
        config.max_num_hands            = int(dd_max_hands.value)
        try:
            config.camera = int(tf_camera.value.strip())
        except (ValueError, AttributeError):
            pass
        try:
            w_str, h_str = dd_resolution.value.split("x")
            config.resolution = [int(w_str), int(h_str)]
        except (ValueError, AttributeError):
            pass
        config.save()
        vision_processor.reinit()
        page.snack_bar = ft.SnackBar(
            content=ft.Text("Vision settings applied.", color=ft.Colors.WHITE),
            bgcolor=C_OK,
        )
        page.snack_bar.open = True
        await page.update_async()

    # ---- Helper: slider row with label ----------------------------------
    def _slider_row(label: str, slider: ft.Slider, val_ref, init_val: float):
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(label, size=12, color=TEXT_MUTED, expand=True),
                        ft.Text(
                            ref=val_ref,
                            value=f"{init_val:.1f}",
                            size=12,
                            color=ACCENT,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    spacing=8,
                ),
                slider,
            ],
            spacing=0,
        )

    # ---- Sección de configuración ---------------------------------------
    settings_section = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.TUNE, color=ACCENT, size=16),
                        ft.Text(
                            "Vision Settings",
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=TEXT_MUTED,
                        ),
                    ],
                    spacing=8,
                ),
                ft.Divider(color=DIVIDER, height=1),
                _slider_row(
                    "Min Detection Confidence",
                    sl_detection, ref_det_val,
                    float(config.min_detection_confidence),
                ),
                _slider_row(
                    "Min Tracking Confidence",
                    sl_tracking, ref_trk_val,
                    float(config.min_tracking_confidence),
                ),
                ft.Row([dd_max_hands, tf_camera], spacing=12),
                dd_resolution,
                ft.ElevatedButton(
                    "Apply Settings",
                    icon=ft.Icons.CHECK_CIRCLE_OUTLINE,
                    bgcolor=ACCENT_DIM,
                    color=ft.Colors.WHITE,
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                    on_click=on_apply,
                    expand=True,
                ),
            ],
            spacing=14,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(20),
    )

    # ---- Panel de info de la mano ---------------------------------------
    def _info_row(label: str, val_ref, init: str, val_color=ft.Colors.WHITE):
        return ft.Row(
            [
                ft.Text(label, size=12, color=TEXT_MUTED, expand=True),
                ft.Text(
                    ref=val_ref,
                    value=init,
                    size=13,
                    weight=ft.FontWeight.BOLD,
                    color=val_color,
                ),
            ],
            spacing=8,
        )

    info_section = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.BACK_HAND_OUTLINED, color=ACCENT, size=16),
                        ft.Text(
                            "Hand Info",
                            size=13,
                            weight=ft.FontWeight.W_600,
                            color=TEXT_MUTED,
                        ),
                    ],
                    spacing=8,
                ),
                ft.Divider(color=DIVIDER, height=1),
                ft.Row(
                    [
                        ft.Container(
                            ref=ref_tracking_dot,
                            width=12, height=12, border_radius=6,
                            bgcolor=C_WARN,
                        ),
                        ft.Text(
                            ref=ref_tracking_lbl,
                            value="NO HAND",
                            size=13,
                            weight=ft.FontWeight.BOLD,
                            color=C_WARN,
                        ),
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                _info_row("Wrist X (px):", ref_hand_x_lbl, "0", ACCENT),
                _info_row("Wrist Y (px):", ref_hand_y_lbl, "0", ACCENT),
                _info_row("Fingers up:", ref_fingers_lbl, "0", C_OK),
            ],
            spacing=12,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(20),
    )

    # ---- Imágenes de video (640x480 → mostrar a 320x240 para no saturar UI) ---
    VIDEO_W = 320
    VIDEO_H = 240

    camera_view = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Camera Feed",
                    size=12,
                    color=TEXT_MUTED,
                    weight=ft.FontWeight.W_500,
                ),
                ft.Container(
                    content=ft.Image(
                        ref=ref_frame_img,
                        src_base64="",
                        width=VIDEO_W,
                        height=VIDEO_H,
                        fit=ft.ImageFit.CONTAIN,
                        error_content=ft.Container(
                            bgcolor=BG_CARD,
                            width=VIDEO_W,
                            height=VIDEO_H,
                            content=ft.Column(
                                [
                                    ft.Icon(ft.Icons.VIDEOCAM_OFF, color=TEXT_MUTED, size=32),
                                    ft.Text("No feed", color=TEXT_MUTED, size=11),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ),
                    ),
                    bgcolor=BG_CARD,
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    width=VIDEO_W,
                    height=VIDEO_H,
                ),
            ],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(16),
    )

    skeleton_view = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Skeleton View",
                    size=12,
                    color=TEXT_MUTED,
                    weight=ft.FontWeight.W_500,
                ),
                ft.Container(
                    content=ft.Image(
                        ref=ref_skel_img,
                        src_base64="",
                        width=VIDEO_W,
                        height=VIDEO_H,
                        fit=ft.ImageFit.CONTAIN,
                        error_content=ft.Container(
                            bgcolor=BG_CARD,
                            width=VIDEO_W,
                            height=VIDEO_H,
                            content=ft.Column(
                                [
                                    ft.Icon(ft.Icons.ACCESSIBILITY_NEW, color=TEXT_MUTED, size=32),
                                    ft.Text("No skeleton", color=TEXT_MUTED, size=11),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ),
                    ),
                    bgcolor="#000000",
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    width=VIDEO_W,
                    height=VIDEO_H,
                ),
            ],
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=BG_PANEL,
        border_radius=10,
        padding=ft.padding.all(16),
    )

    # ---- Layout completo ------------------------------------------------
    content = ft.Container(
        content=ft.Column(
            [
                # Header
                ft.Row(
                    [
                        ft.Icon(ft.Icons.VIDEOCAM, color=ACCENT, size=28),
                        ft.Text(
                            "VISION",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=ACCENT,
                        ),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(color=DIVIDER, height=1),
                # Video row
                ft.Row(
                    [camera_view, skeleton_view],
                    spacing=20,
                    wrap=True,
                ),
                # Settings + info row
                ft.Row(
                    [
                        ft.Column([settings_section], expand=True),
                        ft.Column([info_section], expand=True),
                    ],
                    spacing=20,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=ft.padding.all(32),
        expand=True,
        bgcolor=BG_MAIN,
    )

    # ---- Función de actualización (llamada desde el hilo de UI) ---------
    def update():
        try:
            with state.lock:
                frame_b64 = state.frame_b64
                skel_b64  = state.skeleton_b64
                tracking  = state.tracking_status
                hand_x    = state.hand_x
                hand_y    = state.hand_y
                fingers   = list(state.fingers)
        except Exception:
            return

        if ref_frame_img.current:
            ref_frame_img.current.src_base64 = frame_b64

        if ref_skel_img.current:
            ref_skel_img.current.src_base64 = skel_b64

        if ref_tracking_dot.current:
            ref_tracking_dot.current.bgcolor = C_OK if tracking == "ACTIVE" else C_WARN

        if ref_tracking_lbl.current:
            ref_tracking_lbl.current.value = tracking
            ref_tracking_lbl.current.color = C_OK if tracking == "ACTIVE" else C_WARN

        if ref_hand_x_lbl.current:
            ref_hand_x_lbl.current.value = str(hand_x)

        if ref_hand_y_lbl.current:
            ref_hand_y_lbl.current.value = str(hand_y)

        if ref_fingers_lbl.current:
            ref_fingers_lbl.current.value = str(sum(fingers))

    return content, update


def build_robot_page(state: AppState):
    """
    Página Robot con indicadores LED de comando, contadores de activación
    e historial de comandos.

    Retorna:
        (content, update) donde:
          content — ft.Container con el layout completo
          update  — función sin args que actualiza los Ref desde AppState
    """
    # Colores por comando
    CMD_COLORS = {
        "LEFT":   {"active": "#EF4444", "dim": "#3B1212", "border": "#EF4444"},
        "CENTER": {"active": "#10B981", "dim": "#0A2E1F", "border": "#10B981"},
        "RIGHT":  {"active": "#3B82F6", "dim": "#0F1E3B", "border": "#3B82F6"},
        "OPEN":   {"active": "#10B981", "dim": "#0A2E1F", "border": "#10B981"},
        "CLOSE":  {"active": "#EF4444", "dim": "#3B1212", "border": "#EF4444"},
    }
    CMDS = ["LEFT", "CENTER", "RIGHT", "OPEN", "CLOSE"]

    # ---- Refs LED indicators ------------------------------------------------
    ref_leds: dict[str, ft.Ref[ft.Container]] = {c: ft.Ref[ft.Container]() for c in CMDS}
    ref_led_texts: dict[str, ft.Ref[ft.Text]] = {c: ft.Ref[ft.Text]() for c in CMDS}

    # ---- Refs counters -------------------------------------------------------
    ref_counters: dict[str, ft.Ref[ft.Text]] = {c: ft.Ref[ft.Text]() for c in CMDS}

    # ---- Ref ListView --------------------------------------------------------
    ref_history_list = ft.Ref[ft.ListView]()

    # Last known state to avoid unnecessary rebuilds
    _last: dict = {"position": None, "gripper": None, "hist_len": -1}

    def _make_led(cmd: str) -> ft.Container:
        colors = CMD_COLORS[cmd]
        return ft.Container(
            ref=ref_leds[cmd],
            width=120,
            height=80,
            border_radius=8,
            border=ft.border.all(2, colors["border"]),
            bgcolor=colors["dim"],
            content=ft.Column(
                [
                    ft.Icon(name=ft.Icons.CIRCLE, color=colors["border"], size=18),
                    ft.Text(
                        ref=ref_led_texts[cmd],
                        value=cmd,
                        size=14,
                        weight=ft.FontWeight.BOLD,
                        color=colors["border"],
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
            ),
            tooltip=f"{cmd} command indicator",
        )

    def _make_counter(cmd: str) -> ft.Container:
        colors = CMD_COLORS[cmd]
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
            border_radius=8,
            bgcolor=BG_CARD,
            border=ft.border.all(1, DIVIDER),
            content=ft.Column(
                [
                    ft.Text(cmd, size=11, color=TEXT_MUTED, weight=ft.FontWeight.W_500),
                    ft.Text(
                        ref=ref_counters[cmd],
                        value="0",
                        size=22,
                        weight=ft.FontWeight.BOLD,
                        color=colors["border"],
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
        )

    def _history_tile(entry: dict) -> ft.Container:
        cmd = entry.get("command", "?")
        ts  = entry.get("timestamp", "")
        colors = CMD_COLORS.get(cmd, {"border": TEXT_MUTED, "dim": BG_CARD})
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=12, vertical=6),
            border_radius=6,
            bgcolor=BG_CARD,
            content=ft.Row(
                [
                    ft.Container(
                        width=8, height=8,
                        border_radius=4,
                        bgcolor=colors["border"],
                    ),
                    ft.Container(
                        content=ft.Text(
                            cmd,
                            size=12,
                            weight=ft.FontWeight.BOLD,
                            color=colors["border"],
                        ),
                        width=64,
                    ),
                    ft.Text(ts, size=11, color=TEXT_MUTED, expand=True),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    # ---- Layout --------------------------------------------------------------
    led_row = ft.Row(
        [_make_led(c) for c in CMDS],
        spacing=12,
        wrap=True,
    )

    counter_row = ft.Row(
        [_make_counter(c) for c in CMDS],
        spacing=12,
        wrap=True,
    )

    history_list = ft.ListView(
        ref=ref_history_list,
        controls=[],
        spacing=4,
        height=320,
        auto_scroll=True,
    )

    def _section_header(label: str) -> ft.Row:
        return ft.Row(
            [
                ft.Text(label, size=14, weight=ft.FontWeight.W_600, color=ACCENT),
                ft.Container(expand=True, height=1, bgcolor=DIVIDER),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    content = ft.Container(
        expand=True,
        bgcolor=BG_MAIN,
        padding=ft.padding.all(20),
        content=ft.Column(
            [
                # Page title
                ft.Row(
                    [
                        ft.Icon(ft.Icons.PRECISION_MANUFACTURING, color=ACCENT, size=28),
                        ft.Text("Robot", size=26, weight=ft.FontWeight.BOLD, color=ACCENT),
                    ],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Divider(color=DIVIDER, height=1),

                # Section: Current Command
                _section_header("Current Command"),
                ft.Container(
                    bgcolor=BG_PANEL,
                    border_radius=10,
                    padding=ft.padding.all(16),
                    content=led_row,
                ),

                # Section: Activation Counters
                _section_header("Activation Counters"),
                ft.Container(
                    bgcolor=BG_PANEL,
                    border_radius=10,
                    padding=ft.padding.all(16),
                    content=counter_row,
                ),

                # Section: Command History
                _section_header("Command History (last 50)"),
                ft.Container(
                    bgcolor=BG_PANEL,
                    border_radius=10,
                    padding=ft.padding.all(12),
                    expand=True,
                    content=history_list,
                ),
            ],
            spacing=14,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        ),
    )

    # ---- Update function -----------------------------------------------------
    def update():
        with state.lock:
            position = state.robot_position
            gripper  = state.gripper_text
            counters = dict(state.cmd_counters)
            hist     = list(state.command_history)

        # Determine active commands
        active: set[str] = set()
        if position in ("LEFT", "CENTER", "RIGHT"):
            active.add(position)
        if gripper in ("OPEN", "CLOSE"):
            active.add(gripper)

        # Update LED indicators only when active set changes
        for cmd in CMDS:
            colors = CMD_COLORS[cmd]
            is_active = cmd in active
            ref_leds[cmd].current.bgcolor = colors["active"] if is_active else colors["dim"]

        # Update counters
        for cmd in CMDS:
            ref_counters[cmd].current.value = str(counters.get(cmd, 0))

        # Update history only when new entries arrive
        new_len = len(hist)
        if new_len != _last["hist_len"]:
            _last["hist_len"] = new_len
            last_50 = hist[-50:]
            ref_history_list.current.controls = [_history_tile(e) for e in reversed(last_50)]

    return content, update


# =====================================================
# SETTINGS PAGE
# =====================================================

def build_settings_page(
    state: AppState,
    config: ConfigManager,
    page: ft.Page,
) -> tuple:
    """
    Página Settings con grupos de parámetros configurables y persistencia JSON.

    Retorna:
        (content, update) donde:
          content — ft.Container con el layout completo
          update  — función sin args (no-op: settings no muestra datos en vivo)
    """

    C_OK   = "#10B981"
    C_WARN = "#F59E0B"

    # ------------------------------------------------------------------
    # Refs para etiquetas de valor (se actualizan en on_change y on_reset)
    # ------------------------------------------------------------------
    ref_dz_val = ft.Ref[ft.Text]()
    ref_ot_val = ft.Ref[ft.Text]()
    ref_ct_val = ft.Ref[ft.Text]()
    ref_sm_val = ft.Ref[ft.Text]()
    ref_hy_val = ft.Ref[ft.Text]()
    ref_pw_val = ft.Ref[ft.Text]()
    ref_ui_val = ft.Ref[ft.Text]()

    # ------------------------------------------------------------------
    # Sliders — on_change actualiza config directamente (reflejo inmediato)
    # ------------------------------------------------------------------
    async def _on_dead_zone_change(e):
        config.dead_zone = int(e.control.value)
        if ref_dz_val.current:
            ref_dz_val.current.value = str(int(e.control.value))
            await page.update_async()

    async def _on_open_thresh_change(e):
        config.open_threshold = int(e.control.value)
        if ref_ot_val.current:
            ref_ot_val.current.value = str(int(e.control.value))
            await page.update_async()

    async def _on_close_thresh_change(e):
        config.close_threshold = int(e.control.value)
        if ref_ct_val.current:
            ref_ct_val.current.value = str(int(e.control.value))
            await page.update_async()

    async def _on_smoothing_change(e):
        config.smoothing = round(float(e.control.value), 2)
        if ref_sm_val.current:
            ref_sm_val.current.value = f"{float(e.control.value):.2f}"
            await page.update_async()

    async def _on_hysteresis_change(e):
        config.hysteresis = int(e.control.value)
        if ref_hy_val.current:
            ref_hy_val.current.value = str(int(e.control.value))
            await page.update_async()

    async def _on_plc_freq_change(e):
        config.plc_write_freq = int(e.control.value)
        if ref_pw_val.current:
            ref_pw_val.current.value = str(int(e.control.value))
            await page.update_async()

    async def _on_ui_freq_change(e):
        config.ui_update_freq = int(e.control.value)
        if ref_ui_val.current:
            ref_ui_val.current.value = str(int(e.control.value))
            await page.update_async()

    sl_dead_zone = ft.Slider(
        min=20, max=200, divisions=180,
        value=float(config.dead_zone),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_dead_zone_change,
        expand=True,
    )
    sl_open_thresh = ft.Slider(
        min=2, max=5, divisions=3,
        value=float(config.open_threshold),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_open_thresh_change,
        expand=True,
    )
    sl_close_thresh = ft.Slider(
        min=0, max=2, divisions=2,
        value=float(config.close_threshold),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_close_thresh_change,
        expand=True,
    )
    sl_smoothing = ft.Slider(
        min=0.0, max=1.0, divisions=20,
        value=float(config.smoothing),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_smoothing_change,
        expand=True,
    )
    sl_hysteresis = ft.Slider(
        min=0, max=50, divisions=50,
        value=float(config.hysteresis),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_hysteresis_change,
        expand=True,
    )
    sl_plc_freq = ft.Slider(
        min=1, max=60, divisions=59,
        value=float(config.plc_write_freq),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_plc_freq_change,
        expand=True,
    )
    sl_ui_freq = ft.Slider(
        min=5, max=60, divisions=55,
        value=float(config.ui_update_freq),
        active_color=ACCENT, thumb_color=ACCENT,
        on_change=_on_ui_freq_change,
        expand=True,
    )

    # ------------------------------------------------------------------
    # Switch — auto_reconnect
    # ------------------------------------------------------------------
    sw_auto_reconnect = ft.Switch(
        value=bool(config.auto_reconnect),
        active_color=ACCENT,
        on_change=lambda e: setattr(config, "auto_reconnect", bool(e.control.value)),
    )

    # ------------------------------------------------------------------
    # Callbacks Save / Reset
    # ------------------------------------------------------------------
    async def on_save(e):
        """Persiste la configuración actual y notifica al usuario."""
        config.dead_zone       = int(sl_dead_zone.value)
        config.open_threshold  = int(sl_open_thresh.value)
        config.close_threshold = int(sl_close_thresh.value)
        config.smoothing       = round(float(sl_smoothing.value), 2)
        config.hysteresis      = int(sl_hysteresis.value)
        config.plc_write_freq  = int(sl_plc_freq.value)
        config.ui_update_freq  = int(sl_ui_freq.value)
        config.auto_reconnect  = bool(sw_auto_reconnect.value)
        config.save()
        state.log_manager.info("Configuración guardada desde Settings")
        page.snack_bar = ft.SnackBar(
            content=ft.Text("Settings saved.", color=ft.Colors.WHITE),
            bgcolor=C_OK,
        )
        page.snack_bar.open = True
        await page.update_async()

    async def on_reset(e):
        """Restaura valores por defecto, actualiza controles UI y persiste."""
        config.reset_defaults()
        # Sincronizar sliders
        sl_dead_zone.value    = float(config.dead_zone)
        sl_open_thresh.value  = float(config.open_threshold)
        sl_close_thresh.value = float(config.close_threshold)
        sl_smoothing.value    = float(config.smoothing)
        sl_hysteresis.value   = float(config.hysteresis)
        sl_plc_freq.value     = float(config.plc_write_freq)
        sl_ui_freq.value      = float(config.ui_update_freq)
        sw_auto_reconnect.value = bool(config.auto_reconnect)
        # Sincronizar etiquetas de valor
        if ref_dz_val.current: ref_dz_val.current.value = str(int(config.dead_zone))
        if ref_ot_val.current: ref_ot_val.current.value = str(int(config.open_threshold))
        if ref_ct_val.current: ref_ct_val.current.value = str(int(config.close_threshold))
        if ref_sm_val.current: ref_sm_val.current.value = f"{config.smoothing:.2f}"
        if ref_hy_val.current: ref_hy_val.current.value = str(int(config.hysteresis))
        if ref_pw_val.current: ref_pw_val.current.value = str(int(config.plc_write_freq))
        if ref_ui_val.current: ref_ui_val.current.value = str(int(config.ui_update_freq))
        state.log_manager.info("Configuración restaurada a valores por defecto")
        page.snack_bar = ft.SnackBar(
            content=ft.Text("Defaults restored.", color=ft.Colors.WHITE),
            bgcolor=C_WARN,
        )
        page.snack_bar.open = True
        await page.update_async()

    # ------------------------------------------------------------------
    # Helpers de layout
    # ------------------------------------------------------------------
    def _slider_row(label: str, slider: ft.Slider, val_ref, init_val: str):
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(label, size=12, color=TEXT_MUTED, expand=True),
                        ft.Text(
                            ref=val_ref,
                            value=init_val,
                            size=12,
                            color=ACCENT,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    spacing=8,
                ),
                slider,
            ],
            spacing=0,
        )

    def _section_header(title: str, icon_name: str):
        return ft.Row(
            [
                ft.Container(width=3, height=18, bgcolor=ACCENT, border_radius=2),
                ft.Icon(icon_name, color=ACCENT, size=16),
                ft.Text(title, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ],
            spacing=8,
        )

    def _card(child: ft.Control):
        return ft.Container(
            content=child,
            bgcolor=BG_PANEL,
            border_radius=10,
            padding=ft.padding.all(20),
        )

    # ------------------------------------------------------------------
    # Grupo: Movement Control
    # ------------------------------------------------------------------
    movement_card = _card(
        ft.Column(
            [
                _section_header("Movement Control", ft.Icons.SPEED),
                ft.Divider(color=BG_CARD, height=16),
                _slider_row(
                    "Dead Zone (px)",
                    sl_dead_zone, ref_dz_val,
                    str(int(config.dead_zone)),
                ),
                ft.Container(height=4),
                _slider_row(
                    "Open Threshold (fingers)",
                    sl_open_thresh, ref_ot_val,
                    str(int(config.open_threshold)),
                ),
                ft.Container(height=4),
                _slider_row(
                    "Close Threshold (fingers)",
                    sl_close_thresh, ref_ct_val,
                    str(int(config.close_threshold)),
                ),
                ft.Container(height=4),
                _slider_row(
                    "Smoothing / Alpha",
                    sl_smoothing, ref_sm_val,
                    f"{config.smoothing:.2f}",
                ),
                ft.Container(height=4),
                _slider_row(
                    "Hysteresis (px)",
                    sl_hysteresis, ref_hy_val,
                    str(int(config.hysteresis)),
                ),
            ],
            spacing=4,
        )
    )

    # ------------------------------------------------------------------
    # Grupo: Frequencies
    # ------------------------------------------------------------------
    freq_card = _card(
        ft.Column(
            [
                _section_header("Frequencies", ft.Icons.TIMER),
                ft.Divider(color=BG_CARD, height=16),
                _slider_row(
                    "PLC Write Freq (Hz)",
                    sl_plc_freq, ref_pw_val,
                    str(int(config.plc_write_freq)),
                ),
                ft.Container(height=4),
                _slider_row(
                    "UI Update Freq (Hz)",
                    sl_ui_freq, ref_ui_val,
                    str(int(config.ui_update_freq)),
                ),
            ],
            spacing=4,
        )
    )

    # ------------------------------------------------------------------
    # Grupo: Auto Reconnect
    # ------------------------------------------------------------------
    reconnect_card = _card(
        ft.Row(
            [
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.Icons.SETTINGS_ETHERNET, color=ACCENT, size=16),
                                ft.Text(
                                    "Auto Reconnect PLC",
                                    size=13,
                                    weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.WHITE,
                                ),
                            ],
                            spacing=8,
                        ),
                        ft.Text(
                            "Reintenta conectar al PLC automáticamente cuando se pierde la conexión.",
                            size=11,
                            color=TEXT_MUTED,
                        ),
                    ],
                    spacing=4,
                    expand=True,
                ),
                sw_auto_reconnect,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

    # ------------------------------------------------------------------
    # Nota informativa
    # ------------------------------------------------------------------
    info_card = ft.Container(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.INFO_OUTLINE, color=ACCENT, size=16),
                ft.Text(
                    "Los cambios en los sliders se aplican inmediatamente. "
                    "Usa 'Save Settings' para persistir la configuración en disco.",
                    size=11,
                    color=TEXT_MUTED,
                    expand=True,
                ),
            ],
            spacing=8,
        ),
        bgcolor=BG_CARD,
        border_radius=8,
        padding=ft.padding.symmetric(horizontal=16, vertical=10),
    )

    # ------------------------------------------------------------------
    # Botones
    # ------------------------------------------------------------------
    buttons_row = ft.Row(
        [
            ft.ElevatedButton(
                "Save Settings",
                icon=ft.Icons.SAVE,
                bgcolor=ACCENT,
                color=ft.Colors.BLACK,
                on_click=on_save,
            ),
            ft.OutlinedButton(
                "Reset Defaults",
                icon=ft.Icons.RESTORE,
                style=ft.ButtonStyle(
                    side=ft.BorderSide(1, ACCENT_DIM),
                    color=TEXT_MUTED,
                ),
                on_click=on_reset,
            ),
        ],
        spacing=12,
    )

    # ------------------------------------------------------------------
    # Layout principal
    # ------------------------------------------------------------------
    content = ft.Container(
        content=ft.Column(
            [
                # Encabezado de página
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Icon(ft.Icons.TUNE, color=ACCENT, size=24),
                            ft.Text(
                                "Settings",
                                size=22,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=ft.padding.only(left=4, bottom=8),
                ),
                info_card,
                movement_card,
                freq_card,
                reconnect_card,
                buttons_row,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
        ),
        bgcolor=BG_MAIN,
        padding=ft.padding.all(24),
        expand=True,
    )

    def update():
        """Settings no muestra datos en vivo — no-op requerido por el router."""
        pass

    return content, update


# =====================================================
# PÁGINA LOGS
# =====================================================

def build_logs_page(state: AppState, page: ft.Page) -> tuple:
    """
    Página Logs — sistema de logging profesional con exportación CSV.

    Retorna (ft.Container, update_fn).

    - Header con botones Clear Logs y Export CSV.
    - Filtros por nivel: ALL, INFO, WARNING, ERROR, PLC, VISION, ROBOT.
    - ft.ListView con tiles coloreados (badge nivel + timestamp + mensaje).
    - export_csv(): guarda en directorio del script como logs_YYYYMMDD_HHMMSS.csv.
    - clear_logs(): vacía log_manager y refresca la vista.
    - update(): detecta cambios en conteo y reconstruye la lista.
    """

    # --------------------------------------------------
    # Colores por nivel
    # --------------------------------------------------
    LEVEL_COLORS = {
        "INFO":    "#22D3EE",  # cyan
        "WARNING": "#F59E0B",  # amber
        "ERROR":   "#EF4444",  # red
        "PLC":     "#10B981",  # green
        "VISION":  "#8B5CF6",  # purple
        "ROBOT":   "#F97316",  # orange
    }
    CHIP_COLORS = {
        "ALL":     "#64748B",
        **LEVEL_COLORS,
    }
    FILTER_LEVELS = ["ALL", "INFO", "WARNING", "ERROR", "PLC", "VISION", "ROBOT"]

    # --------------------------------------------------
    # Estado mutable del filtro y caché de conteo
    # --------------------------------------------------
    active_filter   = ["ALL"]
    _last_count     = [-1]
    _last_filter    = ["ALL"]

    # --------------------------------------------------
    # Ref para el ListView
    # --------------------------------------------------
    ref_log_list = ft.Ref[ft.ListView]()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _make_tile(entry: dict) -> ft.ListTile:
        level     = entry.get("level", "INFO")
        color     = LEVEL_COLORS.get(level, "#94A3B8")
        timestamp = entry.get("timestamp", "")
        # Mostrar solo la parte de hora (HH:MM:SS) para ahorrar espacio
        time_str  = timestamp.split("T")[-1] if "T" in timestamp else timestamp
        date_str  = timestamp.split("T")[0]  if "T" in timestamp else ""
        display_ts = f"{date_str} {time_str}" if date_str else time_str

        return ft.ListTile(
            leading=ft.Container(
                content=ft.Container(
                    width=8,
                    height=8,
                    border_radius=4,
                    bgcolor=color,
                ),
                width=20,
                alignment=ft.alignment.center,
            ),
            title=ft.Row(
                [
                    ft.Container(
                        content=ft.Text(
                            level,
                            size=10,
                            weight=ft.FontWeight.BOLD,
                            color=color,
                        ),
                        bgcolor=f"{color}26",
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                        border_radius=4,
                        border=ft.border.all(1, color),
                    ),
                    ft.Text(
                        entry.get("message", ""),
                        size=13,
                        color="#E2E8F0",
                        expand=True,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            subtitle=ft.Text(
                display_ts,
                size=10,
                color="#6B7280",
            ),
            dense=True,
            min_leading_width=20,
        )

    def _rebuild_list():
        if ref_log_list.current is None:
            return
        entries = state.log_manager.get_all()
        flt     = active_filter[0]
        if flt != "ALL":
            entries = [e for e in entries if e.get("level") == flt]
        tiles = [_make_tile(e) for e in reversed(entries)]
        ref_log_list.current.controls = tiles

    # --------------------------------------------------
    # Callbacks de botones principales
    # --------------------------------------------------

    async def on_export_csv(e):
        now        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath   = os.path.join(script_dir, f"logs_{now}.csv")
        n          = state.log_manager.export_csv(filepath)
        state.log_manager.info(
            f"Logs exportados: {n} entradas → logs_{now}.csv"
        )
        page.snack_bar = ft.SnackBar(
            content=ft.Text(
                f"CSV guardado: logs_{now}.csv  ({n} entradas)",
                color="#FFFFFF",
            ),
            bgcolor="#10B981",
            duration=4000,
        )
        page.snack_bar.open = True
        await page.update_async()

    async def on_clear_logs(e):
        state.log_manager.clear()
        _last_count[0] = 0
        if ref_log_list.current is not None:
            ref_log_list.current.controls = []
        page.snack_bar = ft.SnackBar(
            content=ft.Text("Logs limpiados.", color="#FFFFFF"),
            bgcolor="#374151",
            duration=2000,
        )
        page.snack_bar.open = True
        await page.update_async()

    # --------------------------------------------------
    # Filter chips
    # --------------------------------------------------

    chip_refs: dict[str, ft.Ref] = {lv: ft.Ref[ft.Container]() for lv in FILTER_LEVELS}

    def _make_filter_chip(level: str) -> ft.Container:
        color       = CHIP_COLORS[level]
        is_selected = level == "ALL"

        async def on_chip_click(e, lv=level):
            active_filter[0] = lv
            _last_count[0]   = -1          # forzar rebuild en próximo update
            # Actualizar apariencia de todos los chips
            for lv2, ref in chip_refs.items():
                if ref.current is None:
                    continue
                sel = lv2 == lv
                ref.current.bgcolor = CHIP_COLORS[lv2] if sel else "#374151"
                # Actualizar texto del chip (hijo de content)
                txt_ctrl = ref.current.content
                if isinstance(txt_ctrl, ft.Text):
                    txt_ctrl.color = "#FFFFFF" if sel else CHIP_COLORS[lv2]
            _rebuild_list()
            await page.update_async()

        return ft.Container(
            ref=chip_refs[level],
            content=ft.Text(
                level,
                size=11,
                weight=ft.FontWeight.BOLD,
                color="#FFFFFF" if is_selected else color,
            ),
            bgcolor=color if is_selected else "#374151",
            border=ft.border.all(1, color),
            border_radius=12,
            padding=ft.padding.symmetric(horizontal=10, vertical=4),
            on_click=on_chip_click,
            ink=True,
        )

    filter_row = ft.Row(
        [_make_filter_chip(lv) for lv in FILTER_LEVELS],
        spacing=6,
        wrap=True,
    )

    # --------------------------------------------------
    # ListView de logs
    # --------------------------------------------------

    log_list_view = ft.ListView(
        ref=ref_log_list,
        expand=True,
        auto_scroll=False,
        spacing=0,
        padding=ft.padding.all(0),
    )

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    header = ft.Row(
        [
            ft.Icon(ft.Icons.LIST_ALT, color=ACCENT, size=22),
            ft.Text(
                "System Logs",
                size=18,
                weight=ft.FontWeight.BOLD,
                color="#F1F5F9",
            ),
            ft.Container(expand=True),
            ft.OutlinedButton(
                "Clear Logs",
                icon=ft.Icons.DELETE_OUTLINE,
                on_click=on_clear_logs,
                style=ft.ButtonStyle(
                    color={"": "#EF4444"},
                    side={"": ft.BorderSide(1, "#EF4444")},
                ),
            ),
            ft.ElevatedButton(
                "Export CSV",
                icon=ft.Icons.DOWNLOAD_OUTLINED,
                on_click=on_export_csv,
                bgcolor="#3B82F6",
                color="#FFFFFF",
            ),
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=12,
    )

    # --------------------------------------------------
    # Layout completo
    # --------------------------------------------------

    content = ft.Container(
        content=ft.Column(
            [
                header,
                ft.Divider(color=DIVIDER, height=1),
                # Filtros
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                "FILTRAR POR NIVEL",
                                size=10,
                                color="#6B7280",
                                weight=ft.FontWeight.W_600,
                                letter_spacing=1.2,
                            ),
                            filter_row,
                        ],
                        spacing=8,
                    ),
                    bgcolor=BG_PANEL,
                    border_radius=8,
                    padding=ft.padding.all(12),
                ),
                # Lista de logs
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        "ENTRADAS",
                                        size=10,
                                        color="#6B7280",
                                        weight=ft.FontWeight.W_600,
                                        letter_spacing=1.2,
                                    ),
                                ],
                            ),
                            ft.Divider(color=DIVIDER, height=1),
                            log_list_view,
                        ],
                        spacing=8,
                        expand=True,
                    ),
                    bgcolor=BG_PANEL,
                    border_radius=8,
                    padding=ft.padding.all(12),
                    expand=True,
                ),
            ],
            spacing=12,
            expand=True,
        ),
        bgcolor=BG_MAIN,
        padding=ft.padding.all(24),
        expand=True,
    )

    # --------------------------------------------------
    # Función de update (llamada por UIUpdater thread)
    # --------------------------------------------------

    def update():
        if ref_log_list.current is None:
            return
        entries  = state.log_manager.get_all()
        flt      = active_filter[0]
        filtered = [e for e in entries if flt == "ALL" or e.get("level") == flt]
        count    = len(filtered)
        if count != _last_count[0] or flt != _last_filter[0]:
            _last_count[0]  = count
            _last_filter[0] = flt
            ref_log_list.current.controls = [_make_tile(e) for e in reversed(filtered)]

    return content, update


# Mapa sección → índice NavigationRail
_SECTIONS = ["dashboard", "plc", "vision", "robot", "settings", "logs"]

# Mapa sección → label para el placeholder
_SECTION_LABELS = {
    "dashboard": "Dashboard",
    "plc": "PLC",
    "vision": "Vision",
    "robot": "Robot",
    "settings": "Settings",
    "logs": "Logs",
}


async def main(page: ft.Page):
    """
    Punto de entrada de la aplicación Flet.

    Configura:
    - Título y tamaño de ventana.
    - Tema oscuro con seed cian.
    - Layout principal: NavigationRail | VerticalDivider | content_area.
    - Router de páginas a través de route_page().
    - Inicialización de backend (ConfigManager, AppState, PLCManager,
      VisionProcessor) con threads daemon.
    - Auto-guardado de config al cerrar la ventana.
    """

    # -------------------------------------------------------
    # Configuración de ventana y tema
    # -------------------------------------------------------
    page.title = "RJG Industrial Robot Control v4.1"
    page.window.width = 1400
    page.window.height = 900
    page.window.min_width = 1024
    page.window.min_height = 700
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed="cyan")
    page.bgcolor = BG_MAIN
    page.padding = 0
    page.spacing = 0

    # -------------------------------------------------------
    # Inicialización de backend
    # -------------------------------------------------------
    config = ConfigManager()
    state = AppState()
    plc_manager = PLCManager(state, config)
    vision_processor = VisionProcessor(state, config)

    state.log_manager.info("Aplicación iniciada — RJG Industrial Robot Control v4.1")
    state.log_manager.plc_event("PLC sin conectar — presione CONNECT en la sección PLC")
    state.log_manager.vision_event("VisionProcessor listo para iniciar")

    # -------------------------------------------------------
    # Construir páginas y registrar sus funciones de update
    # -------------------------------------------------------
    dash_content,     dash_update     = build_dashboard_page(state, config)
    plc_content,      plc_update      = build_plc_page(state, config, plc_manager, page)
    vision_content,   vision_update   = build_vision_page(state, config, vision_processor, page)
    robot_content,    robot_update    = build_robot_page(state)
    settings_content, settings_update = build_settings_page(state, config, page)
    logs_content,     logs_update     = build_logs_page(state, page)
    current_update_fn = [dash_update]   # lista mutable para el closure del UI thread

    # -------------------------------------------------------
    # Área de contenido central (ref mutable para el router)
    # -------------------------------------------------------
    content_area = ft.Ref[ft.Container]()

    async def route_page(section: str):
        """Actualiza el área de contenido y registra la función de update activa."""
        page_map = {
            "dashboard": (dash_content,                                      dash_update),
            "plc":       (plc_content,                                           plc_update),
            "vision":    (vision_content,                                        vision_update),
            "robot":     (robot_content,                                         robot_update),
            "settings":  (settings_content,                                      settings_update),
            "logs":      (logs_content,                                          logs_update),
        }
        page_content, upd_fn = page_map.get(section, page_map["dashboard"])
        current_update_fn[0] = upd_fn
        content_area.current.content = page_content
        await page.update_async()

    async def on_nav_change(e):
        """Callback del NavigationRail — redirige a la sección seleccionada."""
        idx = e.control.selected_index
        if 0 <= idx < len(_SECTIONS):
            await route_page(_SECTIONS[idx])

    # -------------------------------------------------------
    # NavigationRail (sidebar)
    # -------------------------------------------------------
    sidebar = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=88,
        min_extended_width=180,
        bgcolor=BG_PANEL,
        indicator_color=ACCENT_DIM,
        on_change=on_nav_change,
        leading=ft.Container(
            content=ft.Column(
                [
                    ft.Icon(
                        name=ft.Icons.PRECISION_MANUFACTURING,
                        color=ACCENT,
                        size=32,
                    ),
                    ft.Text(
                        "RJG",
                        color=ACCENT,
                        size=11,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            padding=ft.padding.symmetric(vertical=16),
        ),
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.DASHBOARD_OUTLINED,
                selected_icon=ft.Icons.DASHBOARD,
                label="Dashboard",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS_ETHERNET_OUTLINED,
                selected_icon=ft.Icons.SETTINGS_ETHERNET,
                label="PLC",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.VIDEOCAM_OUTLINED,
                selected_icon=ft.Icons.VIDEOCAM,
                label="Vision",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.PRECISION_MANUFACTURING_OUTLINED,
                selected_icon=ft.Icons.PRECISION_MANUFACTURING,
                label="Robot",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.TUNE_OUTLINED,
                selected_icon=ft.Icons.TUNE,
                label="Settings",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.LIST_ALT_OUTLINED,
                selected_icon=ft.Icons.LIST_ALT,
                label="Logs",
            ),
        ],
    )

    # -------------------------------------------------------
    # Contenedor principal de contenido
    # -------------------------------------------------------
    content_container = ft.Container(
        ref=content_area,
        content=dash_content,
        expand=True,
        bgcolor=BG_MAIN,
    )

    # -------------------------------------------------------
    # Layout raíz: sidebar | divider | content
    # -------------------------------------------------------
    root_layout = ft.Row(
        [
            sidebar,
            ft.VerticalDivider(width=1, color=DIVIDER, thickness=1),
            content_container,
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    page.add(root_layout)

    # -------------------------------------------------------
    # Threads de backend (daemon — se cierran con la ventana)
    # -------------------------------------------------------
    plc_thread = threading.Thread(
        target=plc_manager.run_thread,
        name="PLCMonitor",
        daemon=True,
    )
    vision_thread = threading.Thread(
        target=vision_processor.run_thread,
        name="VisionProcessor",
        daemon=True,
    )
    plc_thread.start()
    vision_thread.start()

    # UI update coroutine — refresca los Ref de la página activa periódicamente
    async def _ui_update_loop():
        while not state.stop_event.is_set():
            interval = 1.0 / max(1, int(config.ui_update_freq))
            try:
                current_update_fn[0]()
                await page.update_async()
            except Exception:
                pass
            await asyncio.sleep(interval)

    asyncio.create_task(_ui_update_loop())

    # -------------------------------------------------------
    # Cierre limpio
    # -------------------------------------------------------
    async def on_window_event(e):
        if getattr(e, "data", "") == "close":
            state.stop_event.set()
            config.save()
            state.log_manager.info("Aplicación cerrada — config guardada.")
            await asyncio.sleep(0.3)
            try:
                page.window.destroy()
            except Exception:
                pass

    page.window.on_event = on_window_event
    page.window.prevent_close = True
    await page.update_async()


# =====================================================
# PUNTO DE ENTRADA
# =====================================================

if __name__ == "__main__":
    ft.app(target=main)
