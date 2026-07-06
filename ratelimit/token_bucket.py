import os
import time
import logging
from typing import Optional
import redis.asyncio as aioredis
from fastapi import Request, HTTPException, status

logger = logging.getLogger("aegis.ratelimit")

# 1. Setup global connection pool state
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
redis_client: Optional[aioredis.Redis] = None


def get_redis_client() -> aioredis.Redis:
    """Provides a singleton async Redis instance initialized with clean string decoding."""
    global redis_client
    if redis_client is None:
        # Pass protocol=2 directly into the factory constructor to ensure strict
        # compatibility back to legacy local Windows Redis servers (v3.x/v4.x)
        redis_client = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            protocol=2
        )
    return redis_client


async def close_redis_client() -> None:
    """
    Gracefully closes and clears the singleton Redis client.

    IMPORTANT: callers in other modules should not import the module-level
    `redis_client` variable directly and try to close it themselves.
    `from token_bucket import redis_client` copies the *current value* at
    import time into the importing module's own namespace - it does not stay
    linked to this module's global. Any later reassignment here (e.g. inside
    get_redis_client()) is invisible to that other copy, and closing that
    other copy's reference does nothing to this module's real client. Always
    go through this function instead so there is a single source of truth.
    """
    global redis_client
    if redis_client is not None:
        try:
            await redis_client.close()
        finally:
            redis_client = None


# 2. Atomic Lua Script to handle rate limits cleanly across workers without race conditions
TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Pull current tracking metrics
local state = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(state[1])
local last_updated = tonumber(state[2])

-- Initialize a fresh bucket if this identity is new
if not tokens then
    tokens = max_tokens
    last_updated = now
else
    -- Dynamically add tokens based on how much time has passed
    local elapsed = now - last_updated
    if elapsed > 0 then
        tokens = tokens + (elapsed * refill_rate)
        if tokens > max_tokens then
            tokens = max_tokens
        end
        last_updated = now
    end
end

-- Allow or reject the processing token request
if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))

    -- Safeguard TTL calculation to ensure older Redis versions don't crash on math.ceil
    local ttl = math.floor(max_tokens / refill_rate) + 1
    redis.call('EXPIRE', key, ttl)
    return 1 -- Allowed
else
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_updated', tostring(last_updated))
    return 0 -- Rejected
end
"""


class TokenBucketLimiter:
    """Handles distributed token bucket validation routines directly in Redis memory."""
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.redis = get_redis_client()

    async def is_allowed(self, identifier: str, cost: int = 1) -> bool:
        key = f"ratelimit:{identifier}"
        now = time.time()
        try:
            result = await self.redis.eval(
                TOKEN_BUCKET_LUA, 1, key, self.max_tokens, self.refill_rate, cost, now
            )
            return bool(result)
        except Exception:
            logger.exception("Redis rate limiter down. Defaulting to open gate (fail-open).")
            return True


class RateLimitDependency:
    """FastAPI path dependency wrapper injection class."""
    # 5 requests per minute: burst capacity of 5 tokens, refilling fully over 60s.
    def __init__(self, max_tokens: int = 5, refill_rate: float = 5 / 60):
        self.limiter = TokenBucketLimiter(max_tokens=max_tokens, refill_rate=refill_rate)

    async def __call__(self, request: Request):
        # Fallback to structural incoming IP address routing
        client_ip = request.client.host if request.client else "unknown_ip"
        identifier = f"{request.url.path}:{client_ip}"

        allowed = await self.limiter.is_allowed(identifier, cost=1)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Your token bucket is completely depleted. Please slow down."
            )