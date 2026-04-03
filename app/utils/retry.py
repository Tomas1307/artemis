import time
from typing import Callable, TypeVar

from loguru import logger

T = TypeVar("T")

RETRYABLE_ERRORS = (
    "DEGRADED",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "connection",
    "ServiceUnavailable",
)

DEGRADED_DELAY = 60.0


def retry_on_api_error(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 5.0,
    backoff_factor: float = 2.0,
) -> T:
    """Execute a function with retry on API errors.

    Uses exponential backoff for transient errors (429, 5xx). For DEGRADED
    errors (sustained API outage), waits 60 seconds between retries since
    short waits are useless.

    Args:
        func: Callable to execute. Must take no arguments (use lambda or partial).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds for transient errors.
        backoff_factor: Multiplier applied to delay after each retry.

    Returns:
        The return value of func on success.

    Raises:
        Exception: The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user during API call")
            raise
        except Exception as exc:
            last_exception = exc
            error_str = str(exc).lower()

            is_retryable = any(err.lower() in error_str for err in RETRYABLE_ERRORS)

            if not is_retryable:
                logger.error(f"Non-retryable error on attempt {attempt}: {exc}")
                raise

            is_degraded = "degraded" in error_str

            if attempt < max_retries:
                if is_degraded:
                    delay = DEGRADED_DELAY
                    logger.warning(
                        f"API DEGRADED on attempt {attempt}/{max_retries}. "
                        f"Waiting {delay:.0f}s before retry..."
                    )
                else:
                    delay = base_delay * (backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"Retryable error on attempt {attempt}/{max_retries}: {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries} attempts exhausted. Last error: {exc}"
                )

    raise last_exception
