
# ============================================
# SISTEMA DE MANTENIMIENTO PREDICTIVO
# Dashboard web con Streamlit
# ============================================

# 1. Importar librerías
import streamlit as st
import pandas as pd
import joblib
from groq import Groq
from datetime import datetime


# 2. Configuración de la página
st.set_page_config(
    page_title="Mantenimiento Predictivo",
    page_icon="⚙️",
    layout="wide"
)


# 3. Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró modelo_mantenimiento.pkl. Ejecuta primero entrenar.py.")
    st.stop()

# 3.1 Detectar un modelo de Groq disponible
def obtener_modelo_groq(client):

    modelos_activos = [
        m.id
        for m in client.models.list().data
        if getattr(m, "active", True)
        and not any(
            palabra in m.id.lower()
            for palabra in [
                "whisper", "guard", "tts",
                "playai", "embed", "speech"
            ]
        )
    ]

    if not modelos_activos:
        raise RuntimeError("No hay modelos de chat disponibles.")

    # Preferir un modelo Llama si está disponible
    modelo_seleccionado = next(
        (m for m in modelos_activos if "llama" in m.lower()),
        modelos_activos[0]
    )

    return modelo_seleccionado

# 3.2 Generar orden de trabajo con Groq
def generar_orden_trabajo(
    client,
    modelo_groq,
    temperatura,
    vibracion,
    horas,
    presion,
    estado,
    confianza
):

    # Identificar el nivel de severidad
    nombres_estados = {
        0: "Operación Normal",
        1: "Mantenimiento Preventivo",
        2: "Fallo Crítico"
    }

    diagnostico = nombres_estados[estado]

    # Crear folio
    folio = datetime.now().strftime("OT-%Y%m%d-%H%M%S")

    # Instrucciones para la IA
    instrucciones = """
    Eres un Jefe de Mantenimiento Industrial.

    Tu responsabilidad es elaborar órdenes de trabajo
    claras, profesionales y orientadas a la seguridad.

    Analiza las lecturas de los sensores y el diagnóstico
    proporcionado por el sistema de Machine Learning.

    No cambies ni contradigas la clasificación del modelo.

    Para un mantenimiento preventivo, recomienda una
    inspección programada.

    Para un fallo crítico, prioriza la seguridad del
    personal, la detención segura del equipo y la
    intervención de personal técnico autorizado.

    No inventes mediciones ni afirmes que una avería
    está confirmada únicamente por una predicción.

    Entrega la respuesta en Markdown con esta estructura:

    # Orden de Trabajo Técnico

    - Folio:
    - Nivel de severidad:
    - Diagnóstico:

    ## Acciones técnicas
    1.
    2.
    3.

    ## Equipo de Protección Personal (EPP)

    ## Recomendación final
    """

    # Datos que recibirá la IA
    solicitud = f"""
    Genera una orden de trabajo con los siguientes datos:

    Folio: {folio}
    Diagnóstico del Machine Learning: {diagnostico}
    Confianza estimada del modelo: {confianza:.2%}

    Lecturas de sensores:
    - Temperatura: {temperatura} °C
    - Vibración: {vibracion} mm/s
    - Horas de operación: {horas}
    - Presión: {presion} PSI

    Estado clasificado: {estado}
    """

    # Solicitar respuesta a Groq
    respuesta = client.chat.completions.create(
        model=modelo_groq,
        messages=[
            {
                "role": "system",
                "content": instrucciones
            },
            {
                "role": "user",
                "content": solicitud
            }
        ],
        temperature=0.3,
        max_completion_tokens=700
    )

    return respuesta.choices[0].message.content


# 4. Título de la aplicación
st.title("⚙️ Sistema Predictivo de Mantenimiento Industrial")
st.write(
    "Monitoreo de sensores y clasificación del estado operativo "
    "de la maquinaria mediante Machine Learning."
)




# 5. Crear controles y botón en la barra lateral

st.sidebar.header("Lecturas de los sensores")

temperatura = st.sidebar.slider(
    "Temperatura (°C)",
    min_value=40.0,
    max_value=110.0,
    value=60.0,
    step=0.5
)

vibracion = st.sidebar.slider(
    "Vibración (mm/s)",
    min_value=0.0,
    max_value=12.0,
    value=1.5,
    step=0.1
)

horas = st.sidebar.slider(
    "Horas de operación",
    min_value=0,
    max_value=5000,
    value=300,
    step=50
)

presion = st.sidebar.slider(
    "Presión (PSI)",
    min_value=70.0,
    max_value=160.0,
    value=92.0,
    step=0.5
)

st.sidebar.divider()

analizar = st.sidebar.button(
    "Analizar maquinaria",
    type="primary",
    use_container_width=True
)


# 6. Guardar el resultado del análisis

if analizar:

    datos_sensor = pd.DataFrame([{
        "temperatura_c": temperatura,
        "vibracion_mms": vibracion,
        "horas_operacion": horas,
        "presion_psi": presion
    }])

    estado = int(modelo.predict(datos_sensor)[0])

    probabilidades = modelo.predict_proba(datos_sensor)[0]

    probabilidades_por_clase = dict(
        zip(modelo.classes_, probabilidades)
    )

    confianza = float(probabilidades_por_clase[estado])

    # Guardar el resultado para conservarlo al interactuar con Streamlit
    st.session_state["analisis"] = {
        "temperatura": temperatura,
        "vibracion": vibracion,
        "horas": horas,
        "presion": presion,
        "estado": estado,
        "probabilidades": probabilidades_por_clase,
        "confianza": confianza
    }

    # Reiniciar el reporte anterior
    st.session_state.pop("orden_trabajo", None)


# 7. Mostrar lecturas actuales

st.subheader("Lecturas actuales de la maquinaria")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")


# 8. Mostrar diagnóstico únicamente después del análisis

if "analisis" not in st.session_state:

    st.info(
        "Configura los parámetros de los sensores y presiona "
        "'Analizar maquinaria' para iniciar el diagnóstico."
    )

else:

    resultado = st.session_state["analisis"]

    estado_predicho = resultado["estado"]
    probabilidades_por_clase = resultado["probabilidades"]
    probabilidad_estado = resultado["confianza"]

    # Mostrar advertencia si los sliders cambiaron después del análisis
    valores_actuales = (
        temperatura,
        vibracion,
        horas,
        presion
    )

    valores_analizados = (
        resultado["temperatura"],
        resultado["vibracion"],
        resultado["horas"],
        resultado["presion"]
    )

    if valores_actuales != valores_analizados:

        st.warning(
            "Los parámetros cambiaron. Presiona 'Analizar maquinaria' "
            "para actualizar el diagnóstico."
        )

    else:

        st.divider()

        st.subheader("Diagnóstico del sistema")

        nombres_estados = {
            0: "Operación Normal",
            1: "Mantenimiento Preventivo",
            2: "Fallo Crítico"
        }

        if estado_predicho == 0:

            st.success(
                f"OPERACIÓN NORMAL. Confianza estimada: "
                f"{probabilidad_estado * 100:.2f}%"
            )

            st.write(
                "La maquinaria presenta parámetros compatibles con "
                "una operación estable. Se recomienda continuar "
                "con el monitoreo rutinario."
            )

        elif estado_predicho == 1:

            st.warning(
                f"MANTENIMIENTO PREVENTIVO. Confianza estimada: "
                f"{probabilidad_estado * 100:.2f}%"
            )

            st.write(
                "Se detectan condiciones asociadas con desgaste "
                "moderado. Se recomienda programar una revisión técnica."
            )

        elif estado_predicho == 2:

            st.error(
                f"FALLO CRÍTICO. Confianza estimada: "
                f"{probabilidad_estado * 100:.2f}%"
            )

            st.write(
                "Las lecturas presentan condiciones asociadas con "
                "un posible fallo crítico. Se recomienda activar "
                "el protocolo de seguridad y solicitar una revisión inmediata."
            )

        # 9. Mostrar probabilidades

        st.subheader("Probabilidad estimada por estado")

        for clase in modelo.classes_:

            clase = int(clase)
            probabilidad = probabilidades_por_clase[clase]

            st.write(
                f"{nombres_estados[clase]}: "
                f"{probabilidad * 100:.2f}%"
            )

            st.progress(float(probabilidad))

        # 10. Generar dictamen u orden de trabajo

        st.divider()

        st.subheader("🤖 Asistente de Mantenimiento con IA")

        if estado_predicho == 0:

            st.success(
                "Dictamen de rutina: continuar con el monitoreo "
                "y mantener el programa de mantenimiento preventivo."
            )

        else:

            if st.button(
                "Generar Orden de Trabajo con Groq",
                type="primary"
            ):

                try:

                    api_key = st.secrets["GROQ_API_KEY"]

                    client = Groq(api_key=api_key)

                    with st.spinner("Consultando modelos disponibles..."):

                        modelo_groq = obtener_modelo_groq(client)

                    st.info(
                        f"Modelo de IA seleccionado: {modelo_groq}"
                    )

                    with st.spinner("Generando orden de trabajo..."):

                        orden = generar_orden_trabajo(
                            client,
                            modelo_groq,
                            resultado["temperatura"],
                            resultado["vibracion"],
                            resultado["horas"],
                            resultado["presion"],
                            estado_predicho,
                            probabilidad_estado
                        )

                    st.session_state["orden_trabajo"] = orden

                except Exception as e:

                    st.warning(
                        "No fue posible generar la orden de trabajo "
                        "con Groq. El diagnóstico de Machine Learning "
                        "continúa disponible."
                    )

                    st.caption(
                        f"Detalle del problema: {type(e).__name__}"
                    )

        # Mostrar la orden si ya fue generada

        if "orden_trabajo" in st.session_state:

            st.divider()

            st.subheader("📋 Orden de Trabajo Generada")

            st.markdown(st.session_state["orden_trabajo"])
