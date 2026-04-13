## Descripcion de los Datos

Para la realizacion de la competencia podran hacer uso de los siguientes archivos.

---

## Archivos

### train.csv — Datos de entrenamiento

Contiene 2,718 consultas de operadores del centro de control de Kuntur Station con sus tool calls correctas. Este archivo representa aproximadamente el 80% del dataset completo. **Contiene datos duplicados y ruidosos** — parte del reto es limpiarlos adecuadamente. Se recomienda particionar este archivo en subconjuntos de entrenamiento y validacion segun la estrategia del grupo.

**Columnas:**
- `id` (str): Identificador unico de la consulta (formato `Q-XXXXX`)
- `query` (str): Consulta en lenguaje natural emitida por un operador del centro de control (en ingles)
- `tool_call` (str): Tool call correcta en formato canonico (variable objetivo)

### test.csv — Consultas de evaluacion

Contiene 766 consultas sin respuesta. Incluye tanto las consultas del score publico como del score privado, mezcladas sin distincion. No hay superposicion con `train.csv`. Los participantes deben predecir la tool call para **todas** las consultas.

**Columnas:**
- `id` (str): Identificador unico de la consulta
- `query` (str): Consulta en lenguaje natural a la cual se debe predecir la tool call

### consultas_centro_control.json — Senal adicional de entrenamiento

Archivo con 810 pares (query, doc_id) que registran consultas historicas del centro de control junto con el documento tecnico que las responde. Algunas entradas incluyen ademas un campo `hard_negative_doc_id` que indica un documento que podria confundirse con el correcto.

**Estructura por entrada:**

    {
      "query": "Which protocol governs radiation lockdown procedures?",
      "doc_id": "MASA-DOC-009",
      "hard_negative_doc_id": "MASA-DOC-012"
    }

### base_conocimiento/ — Corpus de retrieval

Carpeta con los 54 documentos tecnicos de MASA en formato Markdown. Incluye manuales de modulos, protocolos de seguridad (MASA-SEC-001 a MASA-SEC-020), procedimientos operacionales, guias de sistemas, reportes de mision y perfiles de tripulacion. Cada documento se encuentra en `MASA-DOC-XXX/doc.md`.

### tools_definition.json — Definicion de tools

Definicion completa de las 10 tools disponibles en formato JSON, incluyendo nombre, descripcion, parametros con sus valores validos por enumeracion, orden de parametros y ejemplo de uso.

### sample_submission.csv — Formato de ejemplo

Ejemplo del formato de entrega esperado. Contiene una fila por cada `id` de `test.csv` con un valor placeholder (`no_action`) en la columna `tool_call`.

**Columnas:**
- `id` (str): Identificador de la consulta (debe coincidir con `test.csv`)
- `tool_call` (str): Tool call predicha en formato canonico

---

## Formato Canonico de Tool Calls

Las predicciones deben seguir el formato estricto:
- Sin espacios despues de comas
- Parametros en el orden definido por la tool
- Valores string en comillas simples
- Todo en minusculas **excepto `protocol_id` que va en MAYUSCULAS** (e.g., `MASA-SEC-001`)
- Valores numericos sin comillas

**Ejemplos validos:**

`send_alert(module='quetzal',severity='critical',reason='abnormal_temperature')`

`get_telemetry(module='condor',metric='pressure',timeframe_hours=1)`

`activate_protocol(protocol_id='MASA-SEC-009',scope='module_only')`

`calculate_trajectory(maneuver='docking',urgency='planned')`

`no_action`

---

## Nota sobre Vocabulario

Los nombres de los modulos en los documentos tecnicos conservan su grafia original en espanol (Condor, Colibri, Vicuna, Tucan), mientras que los valores validos en las tool calls usan su forma ASCII sin acentos (`condor`, `colibri`, `vicuna`, `tucan`). Tenga en cuenta que estos nombres propios provienen de lenguas indigenas y del espanol, y no forman parte del vocabulario estandar de modelos entrenados en ingles.
