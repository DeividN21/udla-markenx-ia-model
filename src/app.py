from flask import Flask, request, jsonify
import motor_logica

app = Flask(__name__)

@app.route("/api/v1/simulate", methods=['POST'])
def simulate():
    """
    Endpoint principal para la simulación.
    Recibe la decisión del estudiante y el contexto del escenario.
    """
    try:
        data = request.get_json()

        # Validación de todos los datos necesarios
        if 'decision' not in data or 'contexto' not in data:
            return jsonify({"error": "Faltan datos 'decision' o 'contexto'"}), 400

        decision = data.get('decision') # Ej: { "idProducto": 2, "idPrecio": 3, ... }
        contexto = data.get('contexto') # Ej: { "perfilConsumidor": [...], "macroentorno": [...] }

        # 2. Llamar a la lógica de puntuación
        # Función que calculará todo
        resultado = motor_logica.calcular_puntaje(decision, contexto)
        
        # 3. Devolver el resultado
        return jsonify(resultado), 200

    except Exception as e:
        # Manejo de errores
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Se ejecuta el servidor en un puerto que no choque con el backend
    app.run(debug=True, port=5001)