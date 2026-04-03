---
name: repo-structure
description: Repository structure guide defining where to create files, when to use each pattern (processor, pipeline, chain_method, LLM, generator, analyzer, service, etc.), and naming conventions. Use when creating new files, modules, or components to ensure consistent architecture across frontend and backend.
---

# Repository Structure Guide

Use this skill whenever creating a new file, module, or component. It defines WHERE each type of code belongs, WHEN to use each pattern, and HOW to structure it.

## Project Layout

```
konecto-internal-pm/
  CLAUDE.md                         # Project rules (read first)
  docs/                             # Reference documentation
  docs/memory/                      # In-repo RAG knowledge base
  app/                              # Backend
    api/                            # REST API (domain-based)
      middleware/                   # Auth, audit, CORS, rate limiting
    database/                       # ORM models, engine, migrations
    pipelines/                      # Orchestrated multi-step processing
    processors/                     # OOP business logic processors
    chain_methods/                  # LLM interaction adapters
    llms/                           # LLM provider implementations
    generators/                     # Content generation classes
    analyzers/                      # Analysis classes
    services/                       # Standalone business services
    factories/                      # Object creation factories
    storage/                        # Storage abstraction layer
    schemas/                        # Shared Pydantic schemas (non-API)
    prompts/                        # LLM prompt templates (YAML)
    utils/                          # Pure utility functions (functional paradigm)
    scripts/                        # Maintenance & migration scripts
    websocket/                      # WebSocket manager + handlers
    tasks/                          # Celery background tasks
    testing/                        # All backend tests
  docker-compose.yml                # Phase 2: Local dev infra
  alembic/                          # Phase 2: DB migrations
```

---

## Backend Patterns: When to Create What

### Pipelines (`app/pipelines/<pipeline_name>/`)

**WHEN:** You need to orchestrate a multi-step process with distinct sequential phases. Clear input, intermediate states, and output.

**Examples:** Invoice generation, project status report generation, billing reconciliation, AI summary pipeline.

**Structure — Facade + Strategy pattern:**
```
app/pipelines/{pipeline_name}/
  __init__.py                 # Exports Facade class + types
  pipeline_facade.py          # Orchestrator (single public entry point)
  settings.py                 # Pipeline configuration (Pydantic BaseModel)
  doc.md                      # Architecture documentation
  schemas/
    __init__.py
    pipeline_config.py        # Input configuration schema
    {domain_schemas}.py       # One file per Pydantic model
  steps/
    __init__.py
    step_01_{name}.py         # Strategy step (one class per file)
    step_02_{name}.py
    step_03_{name}.py
  utils/
    __init__.py
    {utilities}.py            # Pure functions only, no classes
```

**Rules:**
- Facade class is the ONLY public interface — callers never instantiate steps directly
- Steps are numbered and independently testable
- Each step: one class, one file
- Pipeline settings: Pydantic BaseModel local to the pipeline, never in global `config.py`
- Support progress callbacks for UI updates
- Support graceful cancellation via `stop_check` callback

**Skip pipelines when:** single operation, no sequential logic, or simple CRUD.

### Processors (`app/processors/`)

**WHEN:** You need a class that encapsulates heavy business logic for transforming, validating, or processing data.

**Examples:** Invoice line-item calculator, project health score processor, billing rate calculator, delivery schedule processor.

**Pattern:**
```python
from loguru import logger


class InvoiceCalculator:
    """Calculates invoice totals from deliverable line items.

    Applies rate cards, discounts, taxes, and currency conversions
    for the specified client billing configuration.
    """

    def __init__(self, rate_card: dict) -> None:
        self._rate_card = rate_card

    def calculate(self, deliverables: list[dict]) -> dict:
        """Calculate invoice totals for a list of deliverables.

        Args:
            deliverables: List of deliverable dicts with type and hours.

        Returns:
            Dict with subtotal, taxes, discounts, and total amount.
        """
        pass
```

**Rules:**
- One processor per file; file name: `{purpose}_processor.py`
- Constructor receives dependencies (no global state, no singletons)
- Stateless preferred; if stateful, document it explicitly
- Returns data — never raises HTTP exceptions

### Chain Methods (`app/chain_methods/`)

**WHEN:** You need to interact with an LLM for a specific specialized task.

**Examples:** Project status summarizer, client health narrative generator, risk assessment describer, sprint retrospective generator.

**Pattern:**
```python
from app.llms.llm import BaseLLM
from app.prompts.prompt_loader import prompt_loader


class LLMProjectStatusSummarizer:
    """Generates natural-language project status summaries.

    Uses the project-status prompt template to analyze current
    milestone data, deliverable states, and blockers.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def summarize(self, project_context: dict) -> str:
        """Generate a natural-language status summary for a project.

        Args:
            project_context: Dict with milestones, deliverables, and blockers.

        Returns:
            Natural-language status summary string.
        """
        pass
```

**Naming:** Always prefix with `llm_` — e.g., `llm_project_summarizer.py`.

**Rules:**
- One chain method per file
- LLM instance injected via constructor (never instantiated internally)
- Loads prompts from `app/prompts/template/` via `prompt_loader`
- Parses LLM response into structured typed data

### LLMs (`app/llms/`)

**WHEN:** Adding a new LLM provider or modifying the base interface.

**Structure:**
```
app/llms/
  llm.py                      # Abstract base interface
  llm_anthropic.py            # Claude/Anthropic provider
  llm_openai.py               # OpenAI provider
  api_rate_limit_manager.py   # Shared rate limiting across providers
  cache/                      # LLM response caching
```

**Rules:**
- One provider per file: `llm_{provider}.py`
- All providers implement `BaseLLM`
- Provider instantiated via factory based on settings

### Generators (`app/generators/`)

**WHEN:** You need a class that produces content or output artifacts.

**Examples:** Invoice PDF generator, project report generator, timeline export generator, billing statement generator.

**Rules:**
- One generator per file; file name: `{output}_generator.py`
- Generators produce output; processors transform input — these are distinct responsibilities

### Analyzers (`app/analyzers/`)

**WHEN:** You need to inspect or examine data without modifying it.

**Examples:** Project health analyzer, billing anomaly detector, timeline drift analyzer, client churn risk analyzer.

**Rules:**
- Read-only — never modifies input data
- Returns structured analysis results
- File name: `{subject}_analyzer.py`

### Services (`app/services/`)

**WHEN:** Standalone business logic that coordinates between multiple subsystems but does not fit into an API domain or pipeline.

**Rules:**
- Not the same as API domain services (`app/api/<domain>/service.py`)
- Used for cross-cutting, non-HTTP business operations
- Registered as singletons using `@lru_cache(maxsize=1)` when needed

### Factories (`app/factories/`)

**WHEN:** Object creation requires complex logic, configuration, or conditional instantiation.

**Rules:**
- Factory function pattern: `create_{thing}(config) -> Thing`
- Never put factory logic inside the class it creates

### Prompts (`app/prompts/`)

**WHEN:** Adding or modifying LLM prompt templates.

```
app/prompts/
  __init__.py
  prompt_loader.py            # Singleton PromptLoader class + module-level instance
  template/
    {task_name}.yaml          # One YAML file per task domain
```

**Pattern:**
```python
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PromptLoader:
    """Singleton-like loader for YAML prompt configurations.

    Manages caching and mapping between logical prompt types and physical files.
    """

    def __init__(self, prompts_dir: str = "app/prompts") -> None:
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Any] = {}

        self.PROMPT_TYPE_MAPPING = {
            "example_type": {
                "file": "template/example.yaml",
                "key": "example_prompt"
            },
        }

    def load_prompts(self, filename: str) -> Dict[str, Any]:
        """Loads prompts from a YAML file with in-memory caching."""
        if filename not in self._cache:
            file_path = self.prompts_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
            self._cache[filename] = yaml.safe_load(file_path.read_text())
        return self._cache[filename]

    def get_system_message_by_type(self, prompt_type: str) -> str:
        """Retrieves the system message for a specific prompt type."""
        mapping = self._get_mapping(prompt_type)
        return self.get_system_message(mapping["file"], mapping["key"])

    def get_prompt_template_by_type(self, prompt_type: str) -> Optional[str]:
        """Retrieves the prompt template for a specific prompt type."""
        mapping = self._get_mapping(prompt_type)
        return self.get_prompt_template(mapping["file"], mapping["key"])

    def get_config_by_type(self, prompt_type: str) -> Dict[str, Any]:
        """Retrieves the configuration dict for a specific prompt type."""
        mapping = self._get_mapping(prompt_type)
        return self.get_config(mapping["file"], mapping["key"])

    def get_full_prompt_by_type(self, prompt_type: str) -> Dict[str, Any]:
        """Retrieves the complete prompt object for a specific prompt type."""
        mapping = self._get_mapping(prompt_type)
        return self.get_full_prompt(mapping["file"], mapping["key"])

    def _get_mapping(self, prompt_type: str) -> Dict[str, str]:
        """Retrieves the file and key mapping for a given prompt type."""
        if prompt_type not in self.PROMPT_TYPE_MAPPING:
            valid_types = list(self.PROMPT_TYPE_MAPPING.keys())
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {valid_types}")
        return self.PROMPT_TYPE_MAPPING[prompt_type]

    def get_system_message(self, filename: str, prompt_name: str) -> str:
        """Extracts system message from loaded YAML."""
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("system_message", "")

    def get_prompt_template(self, filename: str, prompt_name: str) -> Optional[str]:
        """Extracts prompt template from loaded YAML."""
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("prompt_template")

    def get_config(self, filename: str, prompt_name: str) -> Dict[str, Any]:
        """Extracts configuration dictionary from loaded YAML."""
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {}).get("config", {})

    def get_full_prompt(self, filename: str, prompt_name: str) -> Dict[str, Any]:
        """Extracts the raw dictionary for the prompt entry."""
        prompts = self.load_prompts(filename)
        return prompts.get(prompt_name, {})


prompt_loader = PromptLoader()
```

**Rules:**
- Prompts are ALWAYS YAML files in `template/`, never hardcoded strings
- Module-level singleton: `prompt_loader = PromptLoader()` at bottom of file
- Import everywhere as: `from app.prompts.prompt_loader import prompt_loader`
- Access via: `prompt_loader.get_system_message_by_type("type_name")`
- Register new prompts by adding entries to `PROMPT_TYPE_MAPPING`
- YAML structure per prompt entry: `system_message`, `prompt_template`, `config` (optional)
- Caching: files loaded once, reused on subsequent calls

### Utils (`app/utils/`)

**WHEN:** Pure utility functions that don't belong to any specific pattern.

**PARADIGM: Functional.** Utils follow the functional programming paradigm — they are pure stateless functions, not classes.

```python
def compute_billing_period(start_date: str, end_date: str) -> str:
    """Compute a human-readable billing period label.

    Args:
        start_date: ISO 8601 start date string.
        end_date: ISO 8601 end date string.

    Returns:
        Formatted billing period string (e.g., "Jan 2026 - Mar 2026").
    """
    pass


def normalize_client_name(name: str) -> str:
    """Normalize a client name for storage and comparison.

    Args:
        name: Raw client name string.

    Returns:
        Trimmed, normalized client name.
    """
    pass
```

**Rules:**
- Pure functions ONLY — NO classes whatsoever
- No side effects, no global mutable state
- Functions can call other utils but never import from API layer, services, or processors
- File name describes the domain: `billing_utils.py`, `date_formatters.py`, `validators.py`
- Split when a file exceeds ~200 lines

### Scripts (`app/scripts/`)

**WHEN:** One-off maintenance, migration, or setup scripts.

**Rules:**
- Standalone executables; always include `if __name__ == "__main__":` guard
- File name: `{action}_{target}.py` — e.g., `seed_demo_clients.py`, `backfill_billing_periods.py`

---

## API Layer (see add-api-route skill for full details)

```
app/api/<domain>/
  router.py       # HTTP boundary (always)
  schemas.py      # Pydantic DTOs (always)
  service.py      # Business logic (if non-trivial)
  repository.py   # DB access (if dedicated table)
  helpers.py      # Pure utility functions (if needed)
```

## Dependency Injection

Use `@lru_cache(maxsize=1)` for all service singletons.

```python
from functools import lru_cache

from app.api.<domain>.service import ResourceService


@lru_cache(maxsize=1)
def get_resource_service() -> ResourceService:
    """FastAPI dependency for ResourceService singleton."""
    return ResourceService()
```

---

## Testing

| Location | Type |
|----------|------|
| `app/testing/unit/` | Isolated unit tests |
| `app/testing/integration/` | API tests with DB |
| `app/testing/e2e/` | Full user flow tests |
| `app/testing/pipeline_tests/{name}/` | Per-pipeline tests |

---

## Decision Matrix

| Need | Pattern | Location |
|------|---------|----------|
| Multi-step orchestration | Pipeline (Facade + Strategy) | `app/pipelines/<name>/` |
| OOP business logic / data transformation | Processor | `app/processors/` |
| LLM call for a specific task | Chain Method | `app/chain_methods/` |
| New LLM provider | LLM class | `app/llms/` |
| Output/artifact generation | Generator | `app/generators/` |
| Read-only data inspection | Analyzer | `app/analyzers/` |
| Cross-cutting business logic | Service | `app/services/` |
| Complex object creation | Factory | `app/factories/` |
| LLM prompt template | YAML | `app/prompts/template/` |
| Pure stateless function | Utils (functional) | `app/utils/` |
| One-off maintenance task | Script | `app/scripts/` |
| REST endpoint | API Domain | `app/api/<domain>/` |
| HTTP middleware | Middleware | `app/api/middleware/` |
| WebSocket handler | WebSocket | `app/websocket/` |
| Background task (Celery) | Task | `app/tasks/` |
| DB table | ORM Model | `app/database/models/` |

## Anti-Patterns

- Multiple classes in one file
- Global functions in a file that contains a class
- Imports inside `try/except` or inside methods
- `{domain}_service.py` instead of `<domain>/service.py`
- Tests in project root (always use `app/testing/`)
- `@dataclass` anywhere (Pydantic only)
- Commented-out code
- Hardcoded LLM prompts (always YAML templates)
- Module-level globals or double-checked locking (use `@lru_cache(maxsize=1)`)
- Classes in `utils/` (functional paradigm — pure functions only)
- Chain methods without an actual LLM call
