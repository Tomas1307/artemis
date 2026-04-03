from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from app.config import settings
from app.skeleton.schemas.skeleton_schema import SkeletonSchema


class SkeletonLoader:
    """Service that loads and validates the MASA universe skeleton YAML.

    Reads the skeleton.yaml file, parses it into a validated SkeletonSchema,
    and caches the result for the lifetime of the process. This is the single
    access point for all skeleton data consumed by generation pipelines.

    Usage:
        from app.skeleton.skeleton_loader import skeleton_loader
        skeleton = skeleton_loader.load()
        module = skeleton.modules["jaguar"]

    Raises:
        FileNotFoundError: If the skeleton YAML file does not exist at the
            configured path.
        ValueError: If the YAML content fails Pydantic schema validation.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path or settings.SKELETON_PATH)
        self._cache: SkeletonSchema | None = None

    def load(self) -> SkeletonSchema:
        """Load and validate the skeleton YAML, using cache after first call.

        Returns:
            Fully validated SkeletonSchema instance.

        Raises:
            FileNotFoundError: If skeleton.yaml is not found at the configured path.
            ValueError: If YAML content fails schema validation.
        """
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            raise FileNotFoundError(
                f"Skeleton file not found: {self._path}. "
                "Ensure SKELETON_PATH is set correctly in .env."
            )

        logger.info(f"Loading skeleton from {self._path}")
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))

        try:
            self._cache = SkeletonSchema(**raw)
        except Exception as exc:
            raise ValueError(
                f"Skeleton YAML failed schema validation: {exc}"
            ) from exc

        logger.info(
            f"Skeleton loaded: {len(self._cache.modules)} modules, "
            f"{len(self._cache.security_protocols)} protocols, "
            f"{len(self._cache.operational_procedures)} procedures, "
            f"{len(self._cache.crew)} crew, "
            f"{len(self._cache.missions)} missions."
        )
        return self._cache

    def reload(self) -> SkeletonSchema:
        """Force a fresh load from disk, bypassing the cache.

        Returns:
            Freshly loaded and validated SkeletonSchema instance.
        """
        self._cache = None
        return self.load()


skeleton_loader = SkeletonLoader()
