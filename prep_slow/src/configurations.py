import os
from logging import getLogger

from src.constants import CONSTANTS, PLATFORM_ENUM

logger = getLogger(__name__)


class PlatformConfigurations:
    platform = os.getenv("PLATFORM", PLATFORM_ENUM.DOCKER.value)
    if not PLATFORM_ENUM.has_value(platform):
        raise ValueError(f"PLATFORM must be one of {[v.value for v in PLATFORM_ENUM.__members__.values()]}")

class CacheConfigurations:
    cache_host = os.getenv("CACHE_HOST", "redis")
    cache_port = int(os.getenv("CACHE_PORT", 6379))
    queue_name = os.getenv("QUEUE_NAME", "queue")


class RedisCacheConfigurations(CacheConfigurations):
    redis_db = int(os.getenv("REDIS_DB", 0))
    redis_decode_responses = bool(os.getenv("REDIS_DECODE_RESPONSES", True))


class APIConfigurations:
    title = os.getenv("API_TITLE", "ServingPattern")
    description = os.getenv("API_DESCRIPTION", "machine learning system serving patterns")
    version = os.getenv("API_VERSION", "0.1")

class ModelConfigurations:
    vocab_file = os.getenv("VOCAB_FILE", "/prep_slow/models/en-vocab-bert.txt")


logger.info(f"{PlatformConfigurations.__name__}: {PlatformConfigurations.__dict__}")
logger.info(f"{APIConfigurations.__name__}: {APIConfigurations.__dict__}")
