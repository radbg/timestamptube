"""
TimestampTube — Generador de timestamps para videos largos de YouTube.

Descarga el audio con yt-dlp, transcribe con Whisper local y usa Claude
para identificar temas/secciones del podcast o live.
Genera timestamps listos para pegar en la descripción de YouTube.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

# ── Cargar .env si existe ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

MODELO_CLAUDE = "claude-sonnet-4-5-20250514"
CHUNK_DURACION = 600   # 10 minutos por chunk
CHUNK_OVERLAP = 5      # 5 segundos de overlap para no perder bordes

MAPA_CALIDAD = {
    "Rápida": "small",
    "Balanceada": "medium",
    "Alta calidad": "large",
}

MAPA_IDIOMAS = {
    "Español": "es",
    "Inglés": "en",
    "Portugués": "pt",
    "Francés": "fr",
}

SYSTEM_PROMPT = (
    "Eres un experto en contenido de YouTube, especializado en podcasts, lives y conversaciones "
    "largas. Tu trabajo es analizar transcripciones y segmentar el contenido por temas, "
    "identificando exactamente cuándo cambia el tema de conversación y asignando títulos "
    "claros y atractivos a cada sección."
)

PROMPT_TIMESTAMPS = """Analiza esta transcripción de un video largo (podcast, live, o conversación) \
e identifica TODOS los temas o secciones del contenido.

Tu objetivo es generar los timestamps para la descripción de YouTube, donde cada tema tiene:
- El momento exacto de inicio (timestamp)
- Un título corto y descriptivo del tema

TRANSCRIPCIÓN CON TIMESTAMPS:
{transcript}

INSTRUCCIONES:
- Identifica cada cambio de tema o sección en la conversación
- El primer timestamp SIEMPRE debe ser 00:00 (Introducción o el tema inicial)
- Los títulos deben ser concisos (3-8 palabras), descriptivos y atractivos
- No uses emojis en los títulos
- Incluye entre 8 y 30 temas dependiendo de la duración del video
- Para un video de 1 hora, unos 12-20 temas es ideal
- Para un video de 2+ horas, 20-30 temas
- Agrupa subtemas menores bajo un tema principal si duran menos de 2 minutos
- Los timestamps deben estar en orden cronológico

FORMATO DE RESPUESTA:
IMPORTANTE: Tu respuesta debe contener ÚNICAMENTE un objeto JSON válido. Sin explicaciones, texto, ni markdown.

{{
  "titulo_video": "Título sugerido para el video (opcional)",
  "temas": [
    {{
      "timestamp_seconds": 0,
      "titulo": "Introducción"
    }},
    {{
      "timestamp_seconds": 185,
      "titulo": "Cómo empezó todo"
    }}
  ]
}}"""


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE AUDIO Y TRANSCRIPCIÓN (Whisper local)
# ══════════════════════════════════════════════════════════════════════════════

def _obtener_duracion_audio(audio_path: str) -> float:
    """Obtiene la duración de un archivo de audio con ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path,
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode == 0 and resultado.stdout.strip():
        try:
            return float(resultado.stdout.strip())
        except ValueError:
            pass
    return 0.0


def _extraer_chunk(audio_path: str, inicio: float, duracion: float, chunk_path: str) -> bool:
    """Extrae un fragmento de audio con FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(inicio),
        "-t", str(duracion),
        "-i", audio_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        chunk_path,
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    return resultado.returncode == 0


def _extraer_segmentos_whisper(resultado: dict) -> list:
    """Extrae y filtra segmentos del resultado de Whisper."""
    if not resultado:
        return []
    segmentos = []
    for seg in resultado.get("segments", []):
        texto = seg.get("text", "").strip()
        if not texto:
            continue
        duracion = float(seg["end"]) - float(seg["start"])
        # Filtrar posibles alucinaciones: segmento largo con muy poco texto
        if duracion > 30 and len(texto) < 10:
            continue
        segmentos.append({
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": texto,
        })
    return segmentos


def _deduplicar_segmentos(segmentos: list) -> list:
    """Elimina duplicados causados por el overlap entre chunks (>50% solapamiento)."""
    if len(segmentos) <= 1:
        return segmentos
    segmentos.sort(key=lambda s: s["start"])
    resultado = [segmentos[0]]
    for seg in segmentos[1:]:
        prev = resultado[-1]
        overlap_inicio = max(prev["start"], seg["start"])
        overlap_fin = min(prev["end"], seg["end"])
        overlap_dur = max(0, overlap_fin - overlap_inicio)
        dur_seg = seg["end"] - seg["start"]
        dur_prev = prev["end"] - prev["start"]
        if dur_seg > 0 and dur_prev > 0:
            ratio = overlap_dur / min(dur_seg, dur_prev)
            if ratio > 0.5:
                # Quedarse con el segmento de texto más largo (más completo)
                if len(seg["text"]) > len(prev["text"]):
                    resultado[-1] = seg
                continue
        resultado.append(seg)
    return resultado


def transcribir_audio(
    audio_path: str,
    modelo: str = "medium",
    idioma: str = "es",
    on_progreso: Optional[Callable] = None,
) -> list:
    """
    Transcribe un archivo de audio con Whisper local.
    Para audios largos (>15 min), divide en chunks de 10 min con 5 seg de overlap.

    Returns:
        Lista de segmentos: [{"start": float, "end": float, "text": str}, ...]
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper no está instalado.\n"
            "Instálalo con: pip install openai-whisper"
        )

    def _p(pct: float, msg: str):
        if on_progreso:
            on_progreso(pct, msg)

    duracion_total = _obtener_duracion_audio(audio_path)
    if duracion_total <= 0:
        duracion_total = 3600  # fallback 1h

    _p(0.05, f"Cargando modelo Whisper '{modelo}'...")
    try:
        modelo_whisper = whisper.load_model(modelo)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo Whisper '{modelo}'.\n"
            f"Prueba con un modelo más pequeño (small o base).\nError: {e}"
        )

    # ── Video corto: procesar de una sola vez ─────────────────────────────────
    if duracion_total <= CHUNK_DURACION * 1.5:
        _p(0.15, f"Transcribiendo audio ({duracion_total/60:.1f} min)...")
        resultado = modelo_whisper.transcribe(
            audio_path,
            language=idioma,
            verbose=False,
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        _p(0.92, "Transcripción completada")
        return _extraer_segmentos_whisper(resultado)

    # ── Video largo: dividir en chunks de 10 min ─────────────────────────────
    chunks = []
    pos = 0.0
    while pos < duracion_total:
        dur_chunk = min(CHUNK_DURACION + CHUNK_OVERLAP, duracion_total - pos)
        if dur_chunk < 5:
            break
        chunks.append((pos, dur_chunk))
        pos += CHUNK_DURACION

    n_chunks = len(chunks)
    _p(0.10, f"Video largo ({duracion_total/60:.1f} min) → {n_chunks} partes de ~10 min")

    todos_segmentos = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (inicio_chunk, dur_chunk) in enumerate(chunks):
            pct = 0.10 + (0.80 * i / n_chunks)
            min_ini = inicio_chunk / 60
            min_fin = (inicio_chunk + dur_chunk) / 60
            _p(pct, f"Transcribiendo parte {i+1}/{n_chunks} ({min_ini:.0f}–{min_fin:.0f} min)...")

            chunk_path = os.path.join(tmpdir, f"chunk_{i+1:03d}.wav")
            if not _extraer_chunk(audio_path, inicio_chunk, dur_chunk, chunk_path):
                continue

            try:
                res = modelo_whisper.transcribe(
                    chunk_path,
                    language=idioma,
                    verbose=False,
                    task="transcribe",
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                )
                segs = _extraer_segmentos_whisper(res)
                # Ajustar timestamps al offset real en el video completo
                for seg in segs:
                    seg["start"] += inicio_chunk
                    seg["end"] += inicio_chunk
                todos_segmentos.extend(segs)
            except Exception:
                continue
            finally:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

    if not todos_segmentos:
        raise RuntimeError("La transcripción está vacía. Verifica que el video tenga audio audible.")

    segmentos = _deduplicar_segmentos(todos_segmentos)
    segmentos.sort(key=lambda s: s["start"])
    _p(0.92, f"{len(segmentos)} segmentos transcritos")
    return segmentos


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

def obtener_info_youtube(url: str) -> dict:
    """Obtiene título, duración y canal sin descargar el video."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-playlist", url]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        return {"title": "Video de YouTube", "duration": 0, "channel": ""}
    try:
        info = json.loads(resultado.stdout)
        return {
            "title": info.get("title", "Video de YouTube"),
            "duration": info.get("duration", 0),
            "channel": info.get("channel", ""),
        }
    except json.JSONDecodeError:
        return {"title": "Video de YouTube", "duration": 0, "channel": ""}


def descargar_audio_youtube(url: str, output_dir: str, on_progreso=None) -> str:
    """Descarga solo el audio de YouTube como WAV 16kHz mono."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--extractor-args", "youtube:player_client=ios,web",
        "--no-playlist",
        "-o", output_path,
        url,
    ]

    if on_progreso:
        on_progreso(0.05, "Descargando audio de YouTube...")

    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"No se pudo descargar el audio.\n"
            f"Verifica que el enlace sea válido y que yt-dlp esté actualizado.\n"
            f"Error: {resultado.stderr[-500:]}"
        )

    # Buscar el archivo descargado (yt-dlp puede variar el ext)
    for ext in ["wav", "m4a", "mp3", "webm", "opus"]:
        candidate = os.path.join(output_dir, f"audio.{ext}")
        if os.path.exists(candidate):
            size_mb = os.path.getsize(candidate) / (1024 * 1024)
            if on_progreso:
                on_progreso(0.15, f"Audio descargado ({size_mb:.0f} MB)")
            return candidate

    raise RuntimeError("No se encontró el archivo de audio descargado.")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

def _obtener_api_key() -> str:
    """Obtiene la API key desde st.secrets (Streamlit Cloud) o variables de entorno."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def generar_timestamps_con_claude(segmentos: list, status_placeholder=None) -> dict:
    """
    Envía la transcripción a Claude con streaming para identificar temas.

    Returns:
        Dict con "titulo_video" (str) y "temas" (list de {timestamp_seconds, titulo})
    """
    from anthropic import Anthropic

    api_key = _obtener_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY no configurada.\n"
            "- Local: crea un archivo .env con ANTHROPIC_API_KEY=tu_clave\n"
            "- Streamlit Cloud: agrega la clave en Settings → Secrets"
        )

    # Construir texto de transcripción con timestamps
    lineas = []
    for seg in segmentos:
        m = int(seg["start"] // 60)
        s = int(seg["start"] % 60)
        lineas.append(f"[{m:02d}:{s:02d}] {seg['text'].strip()}")
    transcript_text = "\n".join(lineas)

    prompt = PROMPT_TIMESTAMPS.format(transcript=transcript_text)
    client = Anthropic(api_key=api_key)

    texto_respuesta = ""
    with client.messages.stream(
        model=MODELO_CLAUDE,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if hasattr(event, "type") and event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    texto_respuesta += event.delta.text
                    if status_placeholder:
                        chars = len(texto_respuesta)
                        status_placeholder.caption(f"🤖 Generando respuesta... ({chars} caracteres)")

    if status_placeholder:
        status_placeholder.empty()

    # Parsear JSON — limpiar posible markdown
    try:
        texto_limpio = texto_respuesta.strip()
        if texto_limpio.startswith("```"):
            texto_limpio = re.sub(r"^```[a-z]*\n?", "", texto_limpio)
            texto_limpio = re.sub(r"\n?```$", "", texto_limpio)
        return json.loads(texto_limpio)
    except json.JSONDecodeError:
        # Intentar extraer JSON del texto
        match = re.search(r'\{[\s\S]*"temas"[\s\S]*\}', texto_respuesta)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            f"Claude no devolvió JSON válido.\nRespuesta: {texto_respuesta[:500]}"
        )


def formatear_timestamps(temas: list) -> str:
    """Formatea los temas como timestamps listos para YouTube."""
    lineas = []
    for tema in temas:
        seg = tema["timestamp_seconds"]
        h = int(seg // 3600)
        m = int((seg % 3600) // 60)
        s = int(seg % 60)
        ts = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        lineas.append(f"{ts} {tema['titulo']}")
    return "\n".join(lineas)


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ EN DISCO
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path.home() / ".timestamptube"


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()


def _cache_ruta(url: str) -> Path:
    return CACHE_DIR / _cache_key(url) / "transcripcion.json"


def guardar_cache(url: str, titulo: str, segmentos: list, modelo: str) -> None:
    ruta = _cache_ruta(url)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "1.0",
        "url": url,
        "titulo": titulo,
        "modelo_whisper": modelo,
        "total_segmentos": len(segmentos),
        "duracion_video": segmentos[-1]["end"] if segmentos else 0,
        "segmentos": segmentos,
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cargar_cache(url: str) -> Optional[dict]:
    ruta = _cache_ruta(url)
    if not ruta.exists():
        return None
    try:
        with open(ruta, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("segmentos"):
            return data
    except Exception:
        pass
    return None


def borrar_cache(url: str) -> None:
    ruta = _cache_ruta(url)
    if ruta.exists():
        ruta.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# UI — STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

def _verificar_password() -> bool:
    """Muestra pantalla de login y devuelve True si la contraseña es correcta."""
    if st.session_state.get("autenticado"):
        return True

    # Obtener contraseña configurada
    try:
        password_correcta = st.secrets["APP_PASSWORD"]
    except Exception:
        password_correcta = os.environ.get("APP_PASSWORD", "")

    if not password_correcta:
        # Si no hay contraseña configurada, dejar pasar (modo dev local)
        return True

    # Pantalla de login
    st.markdown("""
    <div style="max-width: 380px; margin: 6rem auto 0; text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">⏱️</div>
        <h2 style="margin-bottom: 0.2rem;">TimestampTube</h2>
        <p style="color: #888; margin-bottom: 2rem;">Ingresa la contraseña para continuar</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        password_ingresada = st.text_input(
            "Contraseña",
            type="password",
            placeholder="••••••••",
            label_visibility="collapsed",
        )
        if st.button("Entrar", type="primary", use_container_width=True):
            if password_ingresada == password_correcta:
                st.session_state["autenticado"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")

    return False


def main():
    st.set_page_config(
        page_title="TimestampTube",
        page_icon="⏱️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    _inyectar_css()

    # ── Login ─────────────────────────────────────────────────────────────────
    if not _verificar_password():
        st.stop()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="tt-header">
        <div class="tt-logo">⏱️</div>
        <h1>TimestampTube</h1>
        <p>Genera timestamps para podcasts y lives de YouTube con IA</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Verificar API key ─────────────────────────────────────────────────────
    if not _obtener_api_key():
        st.error(
            "**ANTHROPIC_API_KEY no configurada.**\n\n"
            "- **Local:** crea un archivo `.env` con `ANTHROPIC_API_KEY=sk-ant-...`\n"
            "- **Streamlit Cloud:** ve a *Settings → Secrets* y agrega la clave"
        )
        st.stop()

    # ── Router de fases ───────────────────────────────────────────────────────
    fase = st.session_state.get("fase", "inicio")

    if fase == "inicio":
        _fase_inicio()
    elif fase == "transcrito":
        _fase_transcrito()
    elif fase == "completado":
        _fase_completado()
    else:
        st.session_state["fase"] = "inicio"
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1: INICIO
# ──────────────────────────────────────────────────────────────────────────────

def _fase_inicio():
    import shutil

    with st.container():
        url = st.text_input(
            "Enlace de YouTube",
            placeholder="https://www.youtube.com/watch?v=...",
            key="url_input",
            help="Pega el enlace completo del podcast, live o video largo",
        )

        col1, col2 = st.columns(2)
        with col1:
            calidad = st.radio(
                "Calidad de transcripción",
                options=list(MAPA_CALIDAD.keys()),
                index=1,
                key="calidad",
                help="Rápida: small (veloz) · Balanceada: medium · Alta: large (lento pero preciso)",
            )
        with col2:
            idioma = st.selectbox(
                "Idioma del audio",
                options=list(MAPA_IDIOMAS.keys()),
                index=0,
                key="idioma",
            )

    if not url:
        st.markdown("---")
        _mostrar_instrucciones()
        return

    # Validar dominio de YouTube
    if not re.match(r'^https?://(www\.)?(youtube\.com|youtu\.be)/', url):
        st.warning("El enlace no parece ser de YouTube. Verifica que sea correcto.")
        return

    # Verificar yt-dlp instalado
    if not shutil.which("yt-dlp"):
        st.error("**yt-dlp** no está instalado. Ejecuta en terminal: `pip install yt-dlp`")
        return

    # ── Verificar caché ───────────────────────────────────────────────────────
    cache = cargar_cache(url)
    if cache:
        dur_min = cache.get("duracion_video", 0) / 60
        n_segs = cache.get("total_segmentos", 0)
        titulo_cache = cache.get("titulo", "Video guardado")
        modelo_cache = cache.get("modelo_whisper", "?")

        st.markdown(f"""
        <div class="tt-cache-box">
            <div class="tt-cache-icon">💾</div>
            <div>
                <div class="tt-cache-titulo">Transcripción guardada encontrada</div>
                <div class="tt-cache-info">{titulo_cache}</div>
                <div class="tt-cache-meta">{n_segs} segmentos · {dur_min:.1f} min · modelo: {modelo_cache}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Usar transcripción guardada", type="primary", use_container_width=True):
                st.session_state.update({
                    "segmentos": cache["segmentos"],
                    "url": url,
                    "titulo_video": cache.get("titulo", ""),
                    "whisper_modelo": cache.get("modelo_whisper", "medium"),
                    "fase": "transcrito",
                })
                st.rerun()
        with c2:
            if st.button("🔄 Transcribir de nuevo", use_container_width=True):
                borrar_cache(url)
                st.rerun()
        return

    # ── Botón iniciar ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        iniciar = st.button(
            "🚀 GENERAR TIMESTAMPS",
            type="primary",
            use_container_width=True,
            key="btn_iniciar",
        )

    if not iniciar:
        return

    # ── Pipeline completo: descargar → transcribir → analizar ─────────────────
    whisper_modelo = MAPA_CALIDAD[calidad]
    idioma_code = MAPA_IDIOMAS[idioma]

    progreso = st.progress(0, text="Preparando...")
    status = st.empty()

    try:
        # Paso 1: Info del video
        progreso.progress(2, text="Obteniendo información del video...")
        info = obtener_info_youtube(url)
        titulo = info.get("title", "Video de YouTube")
        duracion = info.get("duration", 0)
        canal = info.get("channel", "")

        info_str = f"📺 **{titulo}**"
        if canal:
            info_str += f" — {canal}"
        if duracion > 0:
            info_str += f" · {duracion/60:.0f} min"
        status.info(info_str)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Paso 2: Descargar audio
            progreso.progress(5, text="📥 Descargando audio de YouTube...")
            audio_path = descargar_audio_youtube(url, tmpdir)
            progreso.progress(15, text="Audio descargado. Iniciando transcripción...")

            # Paso 3: Transcribir con Whisper
            def _prog_whisper(pct, msg):
                p = int(15 + pct * 55)  # Mapear 0-1 a 15-70
                progreso.progress(min(p, 70), text=f"🎙️ {msg}")

            segmentos = transcribir_audio(
                audio_path,
                modelo=whisper_modelo,
                idioma=idioma_code,
                on_progreso=_prog_whisper,
            )

        if not segmentos:
            st.error("La transcripción está vacía. El video puede no tener audio audible.")
            return

        # Guardar caché
        guardar_cache(url, titulo, segmentos, whisper_modelo)
        progreso.progress(72, text="Transcripción guardada. Analizando temas con IA...")

        # Paso 4: Analizar temas con Claude
        status_claude = st.empty()
        resultado_temas = generar_timestamps_con_claude(segmentos, status_placeholder=status_claude)
        temas = resultado_temas.get("temas", [])

        if not temas:
            st.error("Claude no identificó temas. Intenta de nuevo o edita la transcripción.")
            return

        texto_timestamps = formatear_timestamps(temas)
        progreso.progress(100, text=f"✅ ¡{len(temas)} timestamps generados!")
        status.empty()

        # Guardar y pasar a fase completado
        st.session_state.update({
            "segmentos": segmentos,
            "temas": resultado_temas,
            "timestamps_texto": texto_timestamps,
            "url": url,
            "titulo_video": titulo,
            "whisper_modelo": whisper_modelo,
            "fase": "completado",
        })
        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        progreso.empty()
        status.empty()
        st.error(f"**Error:** {e}")


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2: TRANSCRITO (revisar/editar antes de analizar)
# ──────────────────────────────────────────────────────────────────────────────

def _fase_transcrito():
    segmentos = st.session_state.get("segmentos", [])
    if not segmentos:
        st.session_state["fase"] = "inicio"
        st.rerun()

    titulo = st.session_state.get("titulo_video", "")
    if titulo:
        st.markdown(f"### 📺 {titulo}")

    st.success(f"Transcripción lista — {len(segmentos)} segmentos · {segmentos[-1]['end']/60:.1f} min")

    st.markdown("### 📝 Transcripción (editable)")
    st.caption("Puedes corregir errores de transcripción antes de identificar los temas.")

    # Generar texto editable si es la primera vez
    if "transcripcion_texto" not in st.session_state:
        lineas = []
        for seg in segmentos:
            m = int(seg["start"] // 60)
            s = int(seg["start"] % 60)
            lineas.append(f"[{m:02d}:{s:02d}] {seg['text'].strip()}")
        st.session_state["transcripcion_texto"] = "\n".join(lineas)

    transcripcion_editada = st.text_area(
        "Transcripción",
        value=st.session_state["transcripcion_texto"],
        height=460,
        key="editor_transcripcion",
        label_visibility="collapsed",
    )
    st.session_state["transcripcion_texto"] = transcripcion_editada

    # Re-parsear segmentos desde el texto editado
    segmentos_editados = _parsear_transcripcion_editada(transcripcion_editada, segmentos)
    if segmentos_editados:
        st.session_state["segmentos"] = segmentos_editados

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        analizar = st.button(
            "🤖 IDENTIFICAR TEMAS CON IA",
            type="primary",
            use_container_width=True,
            key="btn_analizar",
        )
    with c2:
        retranscribir = st.button(
            "🔄 Transcribir de nuevo",
            use_container_width=True,
            key="btn_retranscribir",
        )

    if retranscribir:
        url = st.session_state.get("url", "")
        if url:
            borrar_cache(url)
        for k in ["segmentos", "transcripcion_texto", "temas", "timestamps_texto"]:
            st.session_state.pop(k, None)
        st.session_state["fase"] = "inicio"
        st.rerun()

    if not analizar:
        return

    # Analizar con Claude
    progreso = st.progress(60, text="🤖 Identificando temas con IA...")
    status_claude = st.empty()
    try:
        segs_actuales = st.session_state.get("segmentos", [])
        if not segs_actuales:
            st.error("No hay transcripción disponible.")
            st.session_state["fase"] = "inicio"
            return

        resultado_temas = generar_timestamps_con_claude(segs_actuales, status_placeholder=status_claude)
        temas = resultado_temas.get("temas", [])
        texto_timestamps = formatear_timestamps(temas)

        st.session_state.update({
            "temas": resultado_temas,
            "timestamps_texto": texto_timestamps,
            "fase": "completado",
        })
        progreso.progress(100, text=f"✅ ¡{len(temas)} temas identificados!")
        time.sleep(0.4)
        st.rerun()

    except Exception as e:
        progreso.empty()
        status_claude.empty()
        st.error(f"**Error al analizar temas:** {e}")


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3: COMPLETADO (mostrar timestamps)
# ──────────────────────────────────────────────────────────────────────────────

def _fase_completado():
    temas_data = st.session_state.get("temas", {})
    timestamps_texto = st.session_state.get("timestamps_texto", "")
    temas = temas_data.get("temas", [])

    if not temas:
        st.session_state["fase"] = "inicio"
        st.rerun()

    titulo_video = st.session_state.get("titulo_video", "")
    if titulo_video:
        st.markdown(f"### 📺 {titulo_video}")

    titulo_sugerido = temas_data.get("titulo_video", "")
    if titulo_sugerido:
        st.info(f"💡 **Título sugerido:** {titulo_sugerido}")

    st.success(f"Se identificaron **{len(temas)}** temas")

    # ── Timestamps editables ──────────────────────────────────────────────────
    st.markdown("### 📋 Timestamps para YouTube")
    st.caption("Edita si necesitas ajustes, luego copia desde el bloque de código.")

    timestamps_editados = st.text_area(
        "Timestamps",
        value=timestamps_texto,
        height=max(220, len(temas) * 28),
        key="timestamps_editados",
        label_visibility="collapsed",
    )

    # Vista previa para copiar fácilmente
    st.markdown("#### 📎 Copia y pega en la descripción de YouTube")
    st.code(timestamps_editados, language=None)

    # ── Tabla de temas ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Resumen de temas")

    # Construir datos de la tabla
    filas = []
    for i, tema in enumerate(temas):
        seg = tema["timestamp_seconds"]
        h = int(seg // 3600)
        m = int((seg % 3600) // 60)
        s = int(seg % 60)
        ts = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        if i + 1 < len(temas):
            dur_seg = temas[i + 1]["timestamp_seconds"] - seg
            dur_str = f"{dur_seg/60:.1f} min"
        else:
            dur_str = "hasta el final"

        filas.append({"Timestamp": ts, "Tema": tema["titulo"], "Duración": dur_str})

    # Mostrar como tabla
    import pandas as pd
    st.dataframe(
        pd.DataFrame(filas),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
            "Tema": st.column_config.TextColumn("Tema", width="large"),
            "Duración": st.column_config.TextColumn("Duración", width="small"),
        },
    )

    # ── Botones finales ───────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📝 Editar transcripción", use_container_width=True, key="btn_editar"):
            st.session_state["fase"] = "transcrito"
            st.rerun()
    with c2:
        if st.button("🤖 Re-analizar temas", use_container_width=True, key="btn_reanalizar"):
            st.session_state.pop("temas", None)
            st.session_state.pop("timestamps_texto", None)
            st.session_state["fase"] = "transcrito"
            st.rerun()
    with c3:
        if st.button("🆕 Nuevo video", use_container_width=True, key="btn_nuevo"):
            claves = [k for k in st.session_state.keys() if not k.startswith("_")]
            for k in claves:
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parsear_transcripcion_editada(texto: str, segmentos_orig: list) -> list:
    """Re-parsea segmentos desde el texto editado por el usuario."""
    resultado = []
    for linea in texto.strip().split("\n"):
        linea = linea.strip()
        if not linea:
            continue
        m = re.match(r"\[(\d+):(\d+)\]\s*(.*)", linea)
        if m:
            mm, ss, texto_seg = m.groups()
            t_start = int(mm) * 60 + int(ss)
            mejor_end = t_start + 5.0
            for orig in segmentos_orig:
                if abs(orig["start"] - t_start) <= 2.0:
                    mejor_end = orig["end"]
                    break
            resultado.append({
                "start": float(t_start),
                "end": mejor_end,
                "text": texto_seg.strip(),
            })
    return resultado


def _mostrar_instrucciones():
    st.markdown("""
    **¿Cómo funciona?**

    1. **Pega el enlace** del podcast, live o video largo de YouTube
    2. **Elige la calidad** de transcripción (Rápida es suficiente para la mayoría)
    3. **Selecciona el idioma** del audio
    4. Hacemos clic en **Generar Timestamps** y esperamos:
       - Se descarga solo el audio (sin video)
       - Whisper transcribe localmente (puede tomar varios minutos en videos largos)
       - Claude IA identifica los temas y asigna timestamps
    5. **Copia los timestamps** y pégalos en la descripción de tu video en YouTube

    > Las transcripciones se guardan en disco para que no tengas que volver a procesar el mismo video.
    """)


def _inyectar_css():
    st.markdown("""
    <style>
    /* Header principal */
    .tt-header {
        text-align: center;
        padding: 1.5rem 0 1.2rem;
        margin-bottom: 1rem;
    }
    .tt-logo {
        font-size: 3rem;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .tt-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .tt-header p {
        color: #888;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Caja de caché */
    .tt-cache-box {
        display: flex;
        align-items: center;
        gap: 16px;
        background: linear-gradient(135deg, #1a3a2a, #1a2e2a);
        border: 1px solid #2a6a3a;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0 16px;
    }
    .tt-cache-icon { font-size: 2rem; }
    .tt-cache-titulo {
        font-weight: 700;
        color: #69F0AE;
        font-size: 1.05rem;
    }
    .tt-cache-info {
        color: #ddd;
        font-size: 0.95rem;
        margin-top: 2px;
    }
    .tt-cache-meta {
        color: #888;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* Reducir padding del bloque code para timestamps */
    .stCode pre {
        font-size: 0.88rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
