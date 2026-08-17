"""Genera los clips de relleno de Eva (assets/fillers/) con su misma voz.

Cada clip es una frase corta tipo "déjame checarlo" que eva_platica.py
reproduce del lado del cliente mientras corre un query de BigQuery, para que
Eva no se quede callada. Se generan una sola vez vía Gemini Live (voz Leda,
la misma de Eva) y quedan como wav 24 kHz mono s16 + un .txt con el texto
(el texto alimenta el phoneme scheduler para que la boca cuadre).

Uso:  third_party/.../venv/python.exe scripts/gen_fillers.py
"""
import asyncio
import os
import sys
import wave

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

FILLERS = [
    "Déjame checarlo, dame un segundito.",
    "¡Buena pregunta! Deja consulto los números, no me tardo.",
    "Va, déjame revisar el sistema rapidito.",
    "Un momentito, estoy jalando la información.",
    "Deja lo busco en los datos, ahorita te digo.",
]
OUT_DIR = os.path.join(PROJECT_ROOT, "assets", "fillers")
RATE = 24000  # Gemini Live output rate (igual que RECV_RATE en eva_platica)


async def main():
    from eva_platica import GeminiVoice, MODEL_CANDIDATES

    class FakeSpeaker:
        def feed(self, d):
            pass

        def clear(self):
            pass

        def speaking(self):
            return False

    from google.genai import types

    gv = GeminiVoice(FakeSpeaker(), voice="Leda")
    client = gv._client()
    config = gv._config()
    os.makedirs(OUT_DIR, exist_ok=True)

    async with client.aio.live.connect(model=MODEL_CANDIDATES[0], config=config) as session:
        for i, phrase in enumerate(FILLERS, 1):
            await session.send_client_content(turns=types.Content(
                role="user", parts=[types.Part(
                    text="Repite EXACTAMENTE esta frase, sin agregar ni quitar "
                         f"nada, con tono casual y amable: «{phrase}»")]))
            buf = bytearray()
            async for message in session.receive():
                if message.data:
                    buf.extend(message.data)
                sc = getattr(message, "server_content", None)
                if sc is not None and getattr(sc, "turn_complete", False):
                    break
            if len(buf) < RATE // 2:  # < 0.25 s: algo salió mal
                print(f"[{i}] FALLÓ ({len(buf)} bytes): {phrase}")
                continue
            base = os.path.join(OUT_DIR, f"filler_{i:02d}")
            with wave.open(base + ".wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(bytes(buf))
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(phrase)
            print(f"[{i}] {len(buf) / 2 / RATE:.1f}s  {phrase}")


if __name__ == "__main__":
    asyncio.run(main())
