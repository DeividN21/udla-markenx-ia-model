# SIMULACIÓN DE BASE DE DATOS
MAPEO_PRODUCTOS = {
    1: {"nombre": "Empaque de vidrio", "tags": ["premium", "no_eco_friendly"]},
    2: {"nombre": "Empaque de plástico reciclado", "tags": ["eco_friendly", "innovador"]},
    4: {"nombre": "Lata de aluminio", "tags": ["economico"]}
}

MAPEO_PROMOCIONES = {
    1: {"nombre": "Campaña en TikTok", "tags": ["jovenes", "digital"]},
    3: {"nombre": "Anuncio de TV", "tags": ["masivo", "tradicional"]}
}

def calcular_puntaje(decision, contexto):
    """
    Motor de reglas para calcular el puntaje de la simulación.
    Compara la decisión del estudiante con el contexto del escenario.
    """
    puntaje_base = 50  # Puntaje neutro (de 0 a 100)
    feedback_lista = [] # Aquí se guardan los mensajes de feedback

    # 1. Analizar el Perfil del Consumidor (FIC)
    perfil = contexto.get('perfilConsumidor', [])
    decision_producto_id = decision.get('idProducto')
    tags_producto = MAPEO_PRODUCTOS.get(decision_producto_id, {}).get('tags', [])

    for factor in perfil:
        detalle = factor.get('detalle', '').lower()
        importancia = factor.get('importancia', 'Baja')
        
        # Ejemplo de regla:
        if "orgánicos" in detalle or "eco" in detalle:
            if importancia == "Alta":
                if "eco_friendly" in tags_producto:
                    puntaje_base += 25
                    feedback_lista.append(
                        "[+] ¡Excelente! Elegiste un producto 'eco-friendly' que es CRUCIAL para este perfil."
                    )
                else:
                    puntaje_base -= 20
                    feedback_lista.append(
                        "[-] ¡Cuidado! El perfil valora lo 'eco-friendly' y tu producto no lo es. Impacto negativo alto."
                    )
            elif importancia == "Media":
                if "eco_friendly" in tags_producto:
                    puntaje_base += 10
                    feedback_lista.append(
                        "[+] Bien. El producto 'eco-friendly' gusta a este perfil."
                    )
                else:
                    puntaje_base -= 5
                    feedback_lista.append(
                        "[-] El perfil prefería algo 'eco-friendly'. Impacto negativo leve."
                    )

    # 2. Analizar la Promoción
    decision_promo_id = decision.get('idPromocion')
    tags_promo = MAPEO_PROMOCIONES.get(decision_promo_id, {}).get('tags', [])
    
    for factor in perfil:
        detalle = factor.get('detalle', '').lower()
        importancia = factor.get('importancia', 'Baja')

        if "redes sociales" in detalle or "joven" in detalle:
            if importancia == "Alta":
                if "digital" in tags_promo:
                    puntaje_base += 20
                    feedback_lista.append(
                        "[+] ¡Acertaste! Usar una promoción digital fue clave para este público joven."
                    )
                elif "tradicional" in tags_promo:
                    puntaje_base -= 15
                    feedback_lista.append(
                        "[-] ¡Error! Usaste TV para un público que solo ve TikTok. Impacto negativo."
                    )

    # (Se pueden añadir más reglas para precio, plaza y macroentorno)

    # 4. Calcular resultado final
    puntaje_final = max(0, min(100, puntaje_base)) # Asegurar que el puntaje esté entre 0 y 100
    
    # Unir el feedback en un solo string
    feedback_final = "\n".join(feedback_lista)
    if not feedback_final:
        feedback_final = "Una decisión estándar. No hubo grandes aciertos ni errores."

    return {
        "puntaje": int(puntaje_final * 10),
        "nivelAceptacion": puntaje_final,
        "feedback": feedback_final
    }