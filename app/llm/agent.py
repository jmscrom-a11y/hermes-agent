import logging
from urllib.parse import urlparse

import requests

from app.config.settings import MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
당신은 Hermes Agent입니다.

역할:
- Python 프로젝트 분석
- 코드 수정 제안
- 버그 수정
- 리팩터링
- 새로운 기능 구현

기존 코드를 최대한 유지하면서 필요한 부분만 수정하세요.
"""


def _normalize_ollama_host(url: str) -> str:
    """Ollama Base URL 에서 네이티브 API 호스트 부분을 추출.

    Ollama 의 네이티브 API 엔드포인트(/api/...)는 호스트에 경로를 포함하지 않습니다.
    /v1, /api 등 경로가 포함되어 있으면 모든 호출 시 해당 경로가 앞에 붙어 404 발생.

    Examples:
        >>> _normalize_ollama_host("http://localhost:11434/v1")
        'http://localhost:11434'
        >>> _normalize_ollama_host("http://localhost:11434/v1/")
        'http://localhost:11434'
        >>> _normalize_ollama_host("http://localhost:11434")
        'http://localhost:11434'
    """
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    return f"{scheme}://{host}:{port}"


def ask_llm(prompt: str) -> str:
    """Ollama 에 채팅 요청을 보내고 응답의 메시지 내용을 반환."""
    host = _normalize_ollama_host(OLLAMA_BASE_URL)
    api_url = f"{host}/api/chat"

    logger.info("Sending chat request to Ollama at %s", api_url)
    print("=" * 80)
    print("PROMPT SENT TO OLLAMA")
    print(prompt)
    print("=" * 80)

    response = requests.post(
        api_url,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Thinking-capable models (e.g. ornith:9b) burn most of their
            # latency on a hidden reasoning pass unless this is off —
            # measured 22.9s vs 0.7s for the same prompt.
            "think": False,
        },
        timeout=600,
    )

    response.raise_for_status()
    return response.json()["message"]["content"]
