from __future__ import annotations

import time
from typing import Optional

import redis.asyncio as redis


class RateLimiter:
    """
    Token bucket using Redis.
    Key space: {namespace}:{key}
    """

    def __init__(self, redis_url: str, bucket_capacity: int, fill_rate_per_s: float, namespace: str = "gw"):
        self.redis_url = redis_url
        self.bucket_capacity = float(bucket_capacity)
        self.fill_rate_per_s = float(fill_rate_per_s)
        self.namespace = namespace
        self._redis: Optional[redis.Redis] = None

    async def _conn(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        return self._redis

    async def consume(self, key: str, tokens: int = 1) -> bool:
        r = await self._conn()
        now = time.time()
        bkey = f"{self.namespace}:{key}"
        # optimistic transaction
        while True:
            try:
                await r.watch(bkey)
                data = await r.hgetall(bkey)
                last_ts = float(data.get("ts", now))
                tokens_now = float(data.get("tokens", self.bucket_capacity))
                # refill
                elapsed = max(0.0, now - last_ts)
                tokens_now = min(self.bucket_capacity, tokens_now + elapsed * self.fill_rate_per_s)
                allowed = tokens_now >= tokens
                tokens_after = tokens_now - tokens if allowed else tokens_now
                p = r.pipeline()
                p.hset(bkey, mapping={"tokens": tokens_after, "ts": now})
                p.expire(bkey, 600)
                await p.execute()
                return allowed
            except redis.WatchError:
                # state changed; retry
                continue
            finally:
                await r.unwatch()
