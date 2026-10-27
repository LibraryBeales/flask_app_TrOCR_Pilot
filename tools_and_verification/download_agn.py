"""
Descarga en paralelo las imágenes de un expediente del Archivo General de la Nación (México).

Las imágenes siguen un patrón de ID consecutivo:
    https://repositorio.agn.gob.mx/api/images/image/<ID>
    página 1 -> 8653604, página 2 -> 8653605, ..., página N -> 8653604 + N - 1

Uso:
    python descargar_agn.py
    python descargar_agn.py --inicio 8653604 --total 1090 --hilos 6 --salida agn_expediente

Características:
    - Descargas concurrentes (ThreadPoolExecutor)
    - Reintentos con espera exponencial
    - Reanudable: omite archivos ya descargados
    - Escritura atómica (.part -> archivo final), sin archivos corruptos a medias
    - Detecta la extensión real (jpg/png/tif) por Content-Type o bytes mágicos
    - Guarda un registro de fallos para reintentarlos después
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests
from requests.adapters import HTTPAdapter

URL_BASE = "https://repositorio.agn.gob.mx/api/images/image/{id}"
REFERER = "https://repositorio.agn.gob.mx/"

EXT_POR_TIPO = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/webp": ".webp",
}


def extension_por_bytes(datos: bytes) -> str:
    if datos.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if datos.startswith(b"\x89PNG"):
        return ".png"
    if datos[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if datos[:4] == b"RIFF" and datos[8:12] == b"WEBP":
        return ".webp"
    return ""


def crear_sesion(hilos: int) -> requests.Session:
    sesion = requests.Session()
    adaptador = HTTPAdapter(pool_connections=hilos, pool_maxsize=hilos)
    sesion.mount("https://", adaptador)
    sesion.headers.update({
        "User-Agent": "Mozilla/5.0 (investigacion academica; descarga de expediente)",
        "Referer": REFERER,
        "Accept": "image/*,*/*;q=0.8",
    })
    return sesion


def ya_existe(carpeta: str, nombre_base: str) -> bool:
    for ext in (".jpg", ".png", ".tif", ".webp"):
        if os.path.exists(os.path.join(carpeta, nombre_base + ext)):
            return True
    return False


def descargar(sesion, image_id, pagina, carpeta, reintentos, pausa):
    nombre_base = f"pagina_{pagina:04d}"
    if ya_existe(carpeta, nombre_base):
        return pagina, "omitida", None

    url = URL_BASE.format(id=image_id)
    ultimo_error = None

    for intento in range(1, reintentos + 1):
        try:
            r = sesion.get(url, timeout=60)
            if r.status_code == 200:
                datos = r.content
                tipo = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
                ext = EXT_POR_TIPO.get(tipo) or extension_por_bytes(datos)
                if not ext:
                    # El servidor devolvió algo que no es imagen (HTML de error, JSON, etc.)
                    raise ValueError(f"Respuesta no es imagen (Content-Type: {tipo or 'desconocido'})")

                destino = os.path.join(carpeta, nombre_base + ext)
                temporal = destino + ".part"
                with open(temporal, "wb") as f:
                    f.write(datos)
                os.replace(temporal, destino)

                if pausa:
                    time.sleep(pausa)
                return pagina, "ok", None

            if r.status_code == 404:
                return pagina, "error", f"HTTP 404 (ID {image_id} no existe)"
            ultimo_error = f"HTTP {r.status_code}"
        except Exception as e:
            ultimo_error = str(e)

        time.sleep(2 ** intento)  # espera exponencial: 2, 4, 8, ...

    return pagina, "error", ultimo_error


def main():
    p = argparse.ArgumentParser(description="Descarga imágenes del repositorio del AGN")
    p.add_argument("--inicio", type=int, default=8653604, help="ID de la imagen de la página 1")
    p.add_argument("--total", type=int, default=1090, help="Número de páginas")
    p.add_argument("--hilos", type=int, default=6, help="Descargas simultáneas (recomendado 4-8)")
    p.add_argument("--salida", default="agn_imagenes", help="Carpeta de destino")
    p.add_argument("--reintentos", type=int, default=4)
    p.add_argument("--pausa", type=float, default=0.2, help="Pausa por hilo tras cada descarga (s)")
    args = p.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    sesion = crear_sesion(args.hilos)

    tareas = [(args.inicio + i, i + 1) for i in range(args.total)]
    print(f"Descargando {args.total} páginas (IDs {args.inicio}–{args.inicio + args.total - 1}) "
          f"con {args.hilos} hilos en '{args.salida}'...")

    ok = omitidas = 0
    fallos = []
    inicio_t = time.time()

    ex = ThreadPoolExecutor(max_workers=args.hilos)
    pendientes = {
        ex.submit(descargar, sesion, image_id, pagina, args.salida, args.reintentos, args.pausa)
        for image_id, pagina in tareas
    }
    n = 0
    try:
        while pendientes:
            # timeout corto para que Ctrl+C responda de inmediato (también en Windows)
            hechos, pendientes = wait(pendientes, timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in hechos:
                n += 1
                pagina, estado, error = fut.result()
                if estado == "ok":
                    ok += 1
                elif estado == "omitida":
                    omitidas += 1
                else:
                    fallos.append((pagina, args.inicio + pagina - 1, error))
                print(f"\r[{n}/{args.total}] ok={ok} omitidas={omitidas} fallos={len(fallos)}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nPausado. Terminando las descargas en curso (unos segundos)...")
        ex.shutdown(wait=True, cancel_futures=True)
        print(f"Descargadas en esta sesión: {ok}. Ejecuta el mismo comando para continuar.")
        sys.exit(130)
    ex.shutdown()

    print(f"\nTerminado en {time.time() - inicio_t:.1f} s")

    if fallos:
        ruta_log = os.path.join(args.salida, "fallos.txt")
        with open(ruta_log, "w", encoding="utf-8") as f:
            for pagina, image_id, error in sorted(fallos):
                f.write(f"pagina={pagina}\tid={image_id}\t{error}\n")
        print(f"{len(fallos)} páginas fallaron. Detalle en {ruta_log}")
        print("Vuelve a ejecutar el mismo comando para reintentar solo las faltantes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
