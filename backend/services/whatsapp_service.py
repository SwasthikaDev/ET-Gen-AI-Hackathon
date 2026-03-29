"""
WhatsApp Business API delivery service.
Batches messages and respects API rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("whatsapp")

# Officer registry (in production: pull from database)
OFFICER_REGISTRY: dict[str, list[str]] = {
    "Bengaluru": ["+919XXXXXXXXX"],
    "Hyderabad": ["+919XXXXXXXXX"],
    "Mumbai": ["+919XXXXXXXXX"],
    "Delhi": ["+919XXXXXXXXX"],
    "Chennai": ["+919XXXXXXXXX"],
}


@dataclass
class DeliveryResult:
    city: str
    recipients: int
    delivered: int
    failed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class WhatsAppService:
    """
    Sends briefings via the WhatsApp Business Cloud API.
    Rate-limit aware: max 80 messages/second per phone number.
    """

    RATE_LIMIT_PER_SEC = 80

    def __init__(
        self,
        token: str | None = None,
        phone_number_id: str | None = None,
    ):
        self.token = token or os.environ.get("WHATSAPP_TOKEN", "")
        self.phone_number_id = phone_number_id or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        self._sent_count = 0

    def is_configured(self) -> bool:
        return bool(self.token and self.phone_number_id)

    async def send_briefing(
        self,
        city: str,
        briefing_text: str,
        recipients: list[str] | None = None,
    ) -> DeliveryResult:
        if not self.is_configured():
            logger.info(f"WhatsApp not configured — skipping delivery for {city}")
            return DeliveryResult(city=city, recipients=0, delivered=0, failed=0)

        numbers = recipients or OFFICER_REGISTRY.get(city, [])
        if not numbers:
            logger.warning(f"No officers registered for {city}")
            return DeliveryResult(city=city, recipients=0, delivered=0, failed=0)

        delivered = 0
        failed = 0
        batch_size = self.RATE_LIMIT_PER_SEC

        for i in range(0, len(numbers), batch_size):
            batch = numbers[i : i + batch_size]
            tasks = [self._send_one(number, briefing_text) for number in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    failed += 1
                    logger.error(f"WA send error: {r}")
                else:
                    delivered += 1

            if i + batch_size < len(numbers):
                await asyncio.sleep(1.1)  # honour 80 msg/s limit

        logger.info(f"WA delivery: {city} — {delivered}/{len(numbers)} delivered")
        return DeliveryResult(city=city, recipients=len(numbers), delivered=delivered, failed=failed)

    async def _send_one(self, phone: str, text: str) -> dict[str, Any]:
        try:
            import httpx

            url = f"https://graph.facebook.com/v19.0/{self.phone_number_id}/messages"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            payload = {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "text",
                "text": {"body": text},
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to send WA message to {phone}: {e}")
            raise
