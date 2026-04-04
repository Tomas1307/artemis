# Proyecto ARTEMIS — Enunciado de la Competencia Final

**Asistente de Recuperacion y Toma de decisiones para Misiones Espaciales**

Agencia MASA — Competencia Final MAPLN 2026 — Universidad de los Andes

---

## 1. Contexto

La agencia MASA (Mision Aeroespacial Sudamericana Avanzada) opera la estacion Kuntur Station en orbita baja terrestre. La estacion cuenta con 6 modulos:

| Modulo  | Funcion                          | Notas                     |
|---------|----------------------------------|---------------------------|
| Condor  | Comando y control                | Modulo principal          |
| Quetzal | Laboratorio cientifico           | 4 tripulantes, 12 estaciones |
| Jaguar  | Soporte vital y sistemas criticos| Redundancia clase A       |
| Colibri | Comunicaciones y navegacion      | Antenas de largo alcance  |
| Vicuna  | Almacenamiento y carga           | Puerto de docking         |
| Tucan   | Habitacional / dormitorios       | 6 cabinas individuales    |

La tripulacion actual: Comandante Santiago Reyes, Piloto Ana Valdivia, Especialistas Kai Nakamura y Fatima Al-Hassan, Ingeniero Pavel Kozlov, Oficial Medico Lucia Mendoza.

## 2. La Tarea

Construir un sistema de **RAG + Tool Calling** que, dado un query en lenguaje natural de un operador del centro de control, prediga la tool call correcta en formato canonico.

### Pipeline esperado:

```
Query del operador
       |
       v
ENCODER (bge-small-en-v1.5) --> genera embedding del query
       |
       v
FAISS --> busca documentos relevantes en la base de conocimiento
       |
       v
DECODER (Llama-3.2-1B) recibe: query + contexto recuperado
       |
       v
Genera tool call en formato canonico O "no_action"
```

**Importante:** No todos los queries contienen suficiente informacion para determinar la tool call correcta por si solos. Algunos parametros — particularmente niveles de severidad y protocolos de emergencia — estan definidos exclusivamente en la documentacion tecnica de MASA. El decoder debe aprender a utilizar los documentos recuperados como contexto para determinar estos valores.

## 3. Las 10 Tools

Todas las herramientas utilizan parametros enum (valores de un conjunto cerrado). No hay parametros de texto libre.

### 3.1 Monitoreo y telemetria

**get_telemetry** — Datos de sensores en tiempo real
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `metric`: temperature | pressure | oxygen | radiation | humidity | power
- `timeframe_hours`: 1 | 6 | 12 | 24

**get_crew_status** — Estado de la tripulacion
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `info`: health | location | current_activity | schedule

**get_module_status** — Estado general de un modulo
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `system`: life_support | power | thermal | structural | communications

### 3.2 Alertas y comunicacion

**send_alert** — Alerta a control de mision
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `severity`: low | medium | high | critical
- `reason`: abnormal_temperature | pressure_drop | oxygen_leak | radiation_spike | system_failure | power_fluctuation | communication_loss | structural_damage

**send_message** — Mensaje a tripulante o equipo
- `recipient`: commander | pilot | specialist_1 | specialist_2 | engineer | medical_officer | all_crew
- `priority`: low | medium | high | urgent

### 3.3 Operaciones

**schedule_maintenance** — Programar mantenimiento
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `task`: sensor_repair | filter_replacement | system_calibration | hull_inspection | power_cell_swap | software_update
- `priority`: routine | urgent

**activate_protocol** — Activar protocolo MASA
- `protocol_id`: MASA-SEC-001 ... MASA-SEC-020
- `scope`: module_only | station_wide

**control_system** — Controlar sistema especifico
- `module`: condor | quetzal | jaguar | colibri | vicuna | tucan
- `system`: ventilation | heating | lighting | cooling | filtration
- `action`: activate | deactivate | increase | decrease

### 3.4 Navegacion y mision

**calculate_trajectory** — Calculo orbital
- `maneuver`: orbit_adjustment | docking | debris_avoidance | reentry | station_keeping
- `urgency`: planned | immediate

**request_supply** — Solicitar suministros
- `category`: medical | food | equipment | fuel | spare_parts | scientific
- `urgency`: routine | expedited | emergency

### 3.5 Accion especial

**no_action** — Cuando ninguna tool aplica a la consulta (preguntas informativas, historicas o de contexto general).

## 4. Formato Canonico (Estricto)

Las respuestas deben seguir este formato exacto:

- Sin espacios despues de comas
- Parametros en el orden definido por la tool
- Strings con comillas simples
- Todo en minusculas
- Numeros sin comillas

**Ejemplos:**
```
send_alert(module='quetzal',severity='critical',reason='abnormal_temperature')
get_telemetry(module='condor',metric='pressure',timeframe_hours=1)
calculate_trajectory(maneuver='docking',urgency='planned')
no_action
```

Post-procesamiento del output esta permitido para normalizar el formato.

## 5. Datos Proporcionados

| Archivo | Descripcion |
|---------|-------------|
| `train.csv` | Queries de operadores con la tool call correcta (para entrenamiento del decoder) |
| `consultas_centro_control.json` | Consultas historicas del centro de control con documentos asociados |
| `base_conocimiento/` | Documentos tecnicos de MASA (~54 documentos: manuales, protocolos, procedimientos) |
| `tools_definition.json` | Definicion completa de las 10 tools con parametros validos |
| `test_queries.csv` | Queries de evaluacion — predecir la tool call |
| `sample_submission.csv` | Ejemplo del formato de entrega |

### Formato de `train.csv`:
```csv
id,query,tool_call
Q-00001,"Pull the last 12 hours of oxygen data from Jaguar.",get_telemetry(module='jaguar',metric='oxygen',timeframe_hours=12)
Q-00002,"Kozlov says the air feels heavy in life support — check what is going on.",get_telemetry(module='jaguar',metric='oxygen',timeframe_hours=6)
```

### Formato de `test_queries.csv`:
```csv
id,query
Q-02001,"Nakamura reports a chemical smell in the science lab."
Q-02002,"What was the outcome of the Condor-5 mission?"
```

### Formato de entrega (Kaggle):
```csv
ID,answer
Q-02001,activate_protocol(protocol_id='MASA-SEC-009',scope='module_only')
Q-02002,no_action
```

## 6. Evaluacion

### 6.1 Metrica

**Categorization Accuracy** — Exact match del string completo.

Si el string generado coincide exactamente con el gold standard: 1. Si no: 0. No hay credito parcial.

### 6.2 Tres capas de evaluacion

| Capa | Porcentaje | Descripcion |
|------|-----------|-------------|
| Tablero publico | ~60% | Visible durante la competencia en Kaggle |
| Tablero privado | ~40% | Se revela al cerrar la competencia |
| Preguntas ocultas | ~50-100 | El equipo docente ejecuta el notebook del estudiante |

**Score final** = promedio de las tres capas.

### 6.3 Rubrica

| Item | Porcentaje | Detalle |
|------|-----------|---------|
| Participacion | 20% | Minimo 5 envios a Kaggle |
| Superar baseline | 40% | Promedio de las 3 capas > modelo base (Llama sin fine-tuning). **Si no se supera el baseline, esta porcion es 0.0** |
| Percentil | 40% | < P25: 0% / >= P25: 50% / >= P50: 75% / >= P75: 100% |

## 7. Configuracion Tecnica

| Parametro | Valor |
|-----------|-------|
| Decoder obligatorio | `meta-llama/Llama-3.2-1B` |
| Encoder obligatorio | `BAAI/bge-small-en-v1.5` |
| Equipos | 5 personas |
| Duracion | ~2 semanas |
| GPU disponible | 24GB VRAM, 128GB RAM |
| Framework | PyTorch |
| Idioma del contenido | Ingles (nombres OOV en espanol/quechua) |

## 8. Permitido y Prohibido

### Permitido

- Fine-tuning del encoder (bge-small-en-v1.5)
- Modificar el tokenizer de Llama (agregar tokens especiales)
- Data augmentation con LLMs open-source (ejecutados localmente)
- Reranking y reprompting
- Hybrid search (BM25 + embeddings)
- Post-procesamiento del output (regex, normalizacion de formato)
- Few-shot examples en el prompt
- Curriculum learning, learning rate scheduling
- Cualquier tecnica vista en el curso

### Prohibido

- Modelos pagos (OpenAI, Claude, Cohere, Gemini APIs u otros servicios de pago)
- Datos externos no entregados por el equipo docente
- Usar informacion del test set para entrenar (data leakage)
- Clasificacion con otra red neuronal en lugar del LLM
- Solo prompt engineering sin fine-tuning del decoder
- Usar un decoder diferente a Llama-3.2-1B
- Usar un encoder diferente a bge-small-en-v1.5
- Entrenamiento fuera de PyTorch

## 9. Entregables del Estudiante

### En Kaggle:
- CSV con `ID,answer` (tool call como string)
- Minimo 5 submissions durante la competencia

### En Coursera:
- Notebook de entrenamiento (encoder + decoder)
- Notebook de inferencia (pipeline completo: query -> tool call)
- Checkpoint del decoder (Llama fine-tuneado)
- Checkpoint del encoder (si se realizo fine-tuning)
- Indice FAISS / vectores pre-computados
- Corpus de entrenamiento utilizado

## 10. Timeline

| Momento | Actividad |
|---------|-----------|
| Semana 7 — Anuncio | Publicacion del enunciado, entrega de materiales, apertura de Kaggle |
| Semana 7-8 — Construccion | Chunking, FAISS, primeros fine-tuning del decoder, primeros submissions |
| Semana 8 — Optimizacion | Mejorar retrieval, data augmentation, tecnicas avanzadas |
| Fin Semana 8 — Cierre | Cierre Kaggle, entrega Coursera, evaluacion preguntas ocultas |

---

*MASA // Proyecto ARTEMIS // MAPLN 2026 // Universidad de los Andes*
