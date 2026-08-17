# Eva Holobox en máquina virtual (versión web / browser)

Cómo correr Eva para el holobox cuando el motor vive en una **máquina virtual con GPU**
y el holobox solo abre **un link en un navegador**. El navegador del holobox pone el
micrófono, la cámara y la bocina; la VM pone la GPU y la conexión con Gemini.

---

## 1. Arquitectura

```
        MÁQUINA VIRTUAL (GPU)                      HOLOBOX (navegador)
 ┌─────────────────────────────────┐         ┌──────────────────────────────┐
 │  src/eva_web.py                 │   WS    │  web/index.html              │
 │  ├─ EvaRenderer (FLP/TensorRT)  │◄───────►│  ├─ mic 16 kHz PCM16  ──►    │
 │  ├─ Gemini Live (voz + visión)  │         │  ├─ cámara JPEG ~1fps ──►    │
 │  └─ lip-sync / gestos           │         │  ├─ ◄── voz Eva 24 kHz PCM16 │
 └─────────────────────────────────┘         │  └─ ◄── video Eva JPEG @15fps│
                                             └──────────────────────────────┘
```

- Todo viaja por **un solo WebSocket** (`/ws`) en el puerto 8080.
- El servidor solo atiende **una conversación a la vez** (una sola GPU). Si se abre
  el link en otra pantalla, la nueva sesión "patea" a la anterior (mensaje
  *"Eva se abrió en otra pantalla"*).
- El navegador reconecta solo si se cae el servidor o la red (kiosk-resilient).

## 2. Requisitos en la VM

| Qué | Detalle |
|---|---|
| GPU | NVIDIA con drivers + los engines TensorRT ya generados en `third_party/FLP-win/` |
| Python | El del paquete FLP: `third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe` (no el python del sistema) |
| API key | `.env` en la raíz del proyecto con `GEMINI_API_KEY=...` |
| Red | Puerto **8080** abierto entrante (firewall de Windows + reglas de red de la VM) |
| Internet | Salida a internet (Gemini Live) y, si se usa túnel, a Cloudflare |

> La VM debe tener GPU real (passthrough / instancia GPU). El renderer no corre en CPU.

## 3. Arrancar el servidor (variante holobox 9:16)

Desde la raíz del proyecto en la VM:

```bat
third_party\FLP-win\FasterLivePortrait-windows\venv\python.exe src\eva_web.py ^
    --port 8080 ^
    --source assets\eva_body_blanco_916.png ^
    --source-max-dim 1600 ^
    --height 1280
```

- `--source assets\eva_body_blanco_916.png` — el cuerpo 9:16 fondo blanco del holobox
  (mismo que usa `run_eva_holobox.bat` en local).
- `--source-max-dim 1600` — más resolución para salida vertical (cuesta ~0.7 fps).
- `--height 1280` — alto del video JPEG que se manda al navegador. Se puede subir a
  1536 si la red LAN aguanta; bajar a 720 si el video se traba por ancho de banda.
- `--no-vision` — agregar si NO se quiere mandar la cámara a Gemini (solo audio).

Tarda **~30 s** en cargar los engines de GPU. Cuando imprime
`[web] Eva server on http://localhost:8080` ya está listo.

Otros parámetros útiles: `--fps 15` (default), `--voice Leda` (default),
`--jpeg-quality 80`, `--sway`, `--body-sway 1.6`, `--body-breath`.

## 4. El link para el holobox — OJO con HTTPS

**Este es el gotcha #1:** el navegador solo entrega micrófono y cámara en un
*origen seguro* (`https://...` o `http://localhost`). Un link `http://IP-de-la-VM:8080`
muestra a Eva pero **el permiso de micrófono/cámara ni siquiera aparece**.

### Opción A — Túnel Cloudflare (recomendado, da HTTPS gratis)

En la VM, con el servidor ya corriendo:

```bat
cloudflared tunnel --url http://localhost:8080
```

Imprime un link tipo `https://algo-algo.trycloudflare.com`. **Ese es el link que se
abre en el holobox.** (Es lo que hace `run_eva_web.bat`: servidor + túnel juntos.)

- Ventaja: funciona desde cualquier red, HTTPS sin certificados.
- Contras: el link **cambia en cada arranque** del túnel (y al ser origen nuevo,
  Chrome vuelve a pedir permiso de mic/cámara), y el audio/video dan un rodeo por
  internet (+latencia). Para un link fijo: túnel Cloudflare con dominio propio.

### Opción B — LAN directa (menor latencia, requiere flag en Chrome)

Si el holobox y la VM están en la misma red, abrir `http://<IP-de-la-VM>:8080`
en un Chrome lanzado con el origen marcado como seguro:

```bat
chrome.exe --unsafely-treat-insecure-origin-as-secure=http://<IP-de-la-VM>:8080 ^
           --kiosk http://<IP-de-la-VM>:8080
```

(`--kiosk` = pantalla completa sin barra; para salir Alt+F4.)

### Autoarranque sugerido en el holobox

```bat
chrome.exe --kiosk --autoplay-policy=no-user-gesture-required "https://TU-LINK"
```

Aun así hay que **tocar el botón "▶ Hablar con Eva"** una vez — el navegador exige
un gesto del usuario para activar el audio. En un kiosko táctil basta un tap al arrancar.

## 5. Dispositivos: los del holobox por default, externos cuando se quiera

Al tocar "Hablar con Eva", la página pide mic + cámara y usa **los dispositivos
default del navegador/SO del holobox**. La bocina es la salida de audio default.

### Micrófono 🎤
- **En pantalla**: abajo a la izquierda hay medidor de nivel + un **dropdown para
  elegir micrófono** entre todos los detectados (interno del holobox, USB, etc.).
  El cambio es en caliente, sin recargar.
- **Auto-recuperación**: si el mic elegido da 12 s de silencio absoluto, la página
  lo declara muerto y **cicla sola al siguiente** hasta oír algo. Si el track se cae,
  lo reabre. Es decir: conectar un mic USB externo y elegirlo en el dropdown, y listo.
- El medidor dice "te escucho ✓" cuando hay señal — es la verificación rápida en sitio.

### Cámara 📷
- La página pide la cámara frontal default; **no tiene selector en pantalla** (hoy).
- Para usar una cámara externa: en Chrome del holobox ir a
  `chrome://settings/content/camera` y poner la externa como default, luego recargar
  la página. (Alternativa: deshabilitar la interna en el Administrador de dispositivos.)
- Sin cámara no pasa nada: Eva funciona solo con audio (el kiosko sin cámara es un
  caso soportado). El preview propio aparece abajo a la derecha cuando sí hay cámara.

### Bocina 🔊
- Eva se reproduce por la **salida de audio default del sistema** del holobox
  (la página no tiene selector de salida).
- Para cambiar a bocinas externas: cambiar el dispositivo de salida predeterminado
  en el SO del holobox (Windows: icono de volumen → elegir dispositivo; Android:
  se va solo al último conectado por USB/Bluetooth). No hace falta recargar.
- **Anti-eco integrado**: el servidor silencia el mic mientras Eva habla y ~1.5 s
  después, así la voz de Eva saliendo por la bocina no se le regresa a Gemini.
  Aun así, si bocina y mic quedan pegados a todo volumen, conviene separar el mic
  o bajar un poco el volumen.

## 6. Operación y diagnóstico

- **Estado en pantalla**: arriba al centro ("Conectado — habla con Eva 🎙️",
  errores de JS, caídas de mic, etc.).
- **Gestos manuales**: botones abajo (Asentir / Negar / Ladear / Acercarse / Encoger).
- **Test de micrófono**: abrir `https://TU-LINK/mictest` en el holobox — prueba cada
  dispositivo de entrada y reporta resultados a la consola del servidor (`[mictest]`).
- **Consola del servidor (VM)**: cada 5 s imprime un heartbeat:
  `[web] mic: N pkts/5s (peak 0.xxx, gated N) | gemini audio N kB, turns N`
  - `pkts/5s = 0` → el navegador no está mandando audio (permiso/origen inseguro).
  - `peak ~0.000` → mic muerto o silenciado.
  - `gated` alto → el mic se está muteando porque Eva habla (normal durante su turno).
  - También llega telemetría del navegador: `[web] diag: {actx, mic, label, rms}`.
- **Una sesión a la vez**: si alguien abre el link en su teléfono, se roba la sesión
  del holobox. No compartir el link del túnel más allá del equipo.

## 7. Problemas típicos

| Síntoma | Causa / arreglo |
|---|---|
| Eva se ve pero no escucha, y nunca pidió permisos | Link `http://` con IP → origen inseguro. Usar túnel HTTPS (opción A) o el flag de Chrome (opción B). |
| "⚠️ Sin micrófono" | Permiso denegado — candado en la barra de dirección → permitir micrófono → recargar. |
| Pidió permisos otra vez | El link del túnel cambió (origen nuevo). Normal con trycloudflare. |
| Audio entrecortado / video trabado | Ancho de banda: bajar `--height` a 720 y/o `--jpeg-quality` a 60, o pasar a LAN directa. |
| "Eva se abrió en otra pantalla" | Otro dispositivo abrió el link y tomó la sesión. Recargar en el holobox para retomarla. |
| Se oye eco / Eva se interrumpe sola | Mic demasiado cerca de la bocina — separarlos o bajar volumen (el gate de 1.5 s cubre lo normal). |
| El servidor tarda en arrancar | Normal: ~30 s cargando engines TensorRT. |
