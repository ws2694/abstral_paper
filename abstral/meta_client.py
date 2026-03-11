"""Meta-agent client factory — provider-aware client creation.

Supports Anthropic (Claude) and OpenAI (GPT-4o) as meta-agent providers,
enabling the Meta-Agent Sensitivity Analysis experiment (§6.9).
"""

from __future__ import annotations

import logging
from typing import Any

import instructor

from abstral.config import MetaAgentConfig

logger = logging.getLogger(__name__)


def create_meta_client(config: MetaAgentConfig) -> Any:
    """Create an instructor-patched client for the configured meta-agent provider.

    Returns a client compatible with instructor's chat.completions.create() API.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        from anthropic import Anthropic
        return instructor.from_anthropic(
            Anthropic(max_retries=5),
        )

    elif provider == "openai":
        from openai import OpenAI
        return instructor.from_openai(
            OpenAI(max_retries=5),
        )

    else:
        raise ValueError(f"Unsupported meta-agent provider: {provider}")


def create_raw_client(config: MetaAgentConfig) -> Any:
    """Create a raw (non-instructor) client for the configured provider.

    Used by the updater which needs raw message responses, not structured output.
    """
    provider = config.provider.lower()

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(max_retries=5)

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(max_retries=5)

    else:
        raise ValueError(f"Unsupported meta-agent provider: {provider}")
