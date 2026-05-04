"""Streaming HTTP client for ARC-AGI tasks against the configured LLM endpoint."""

import os
import json
import urllib.request
import logging

# LLM endpoint configuration via environment variables.
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def _post_anthropic(endpoint: str, payload: dict, retries: int = 5) -> str:
    """POST to the configured LLM endpoint and parse the SSE stream."""
    import random

    url = f"{ANTHROPIC_BASE_URL}/{endpoint}"
    data = json.dumps(payload).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_AUTH_TOKEN,
        'anthropic-version': '2023-06-01'
    }

    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                # Parse SSE stream
                content_parts = []
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])
                            if event_data.get('type') == 'content_block_delta':
                                delta = event_data.get('delta', {})
                                text = delta.get('text', '')
                                if text:
                                    content_parts.append(text)
                        except json.JSONDecodeError:
                            continue

                return ''.join(content_parts)

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limit
                # Exponential backoff with jitter
                base_wait = min(10 * (2 ** attempt), 60)
                jitter = random.uniform(0, base_wait * 0.1)
                wait_time = base_wait + jitter

                logging.warning(f"Rate limit (429). Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{retries})")

                if attempt < retries - 1:
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Max retries exceeded for API request.")
            else:
                raise
        except Exception as e:
            if attempt == retries - 1:
                raise

            # Exponential backoff for other errors
            wait_time = min(5 * (2 ** attempt), 30)
            logging.warning(f"LLM API error (attempt {attempt+1}/{retries}): {e}")

            if attempt < retries - 1:
                import time
                time.sleep(wait_time)

    raise Exception("Max retries exceeded for LLM API")


def generate_arc_solution(prompt: str, temperature: float = 0.7, use_extended_thinking: bool = True) -> str:
    """Generate a solution for ARC-style tasks via the LLM endpoint.

    Args:
        prompt: Task description
        temperature: Sampling temperature
        use_extended_thinking: Enable extended thinking for complex reasoning
    """

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 8000 if use_extended_thinking else 4096,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    # Enable extended thinking with reduced budget for 50 tasks
    if use_extended_thinking:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": 4000  # Reduced from 10000 to save tokens
        }

    return _post_anthropic("messages", payload)


def generate_arc_samples(prompt: str, n: int = 3, temperature: float = 0.8) -> list:
    """Generate multiple solution candidates for ARC-style tasks."""
    samples = []

    for i in range(n):
        try:
            solution = generate_arc_solution(prompt, temperature=temperature)
            samples.append({
                'solution': solution,
                'reasoning': 'Generated via LLM endpoint',
                'abstract_signature': None
            })
        except Exception as e:
            logging.warning(f"Failed to generate sample {i+1}: {e}")

    return samples, 0  # Return 0 tokens for compatibility
