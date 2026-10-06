from __future__ import annotations

import json

import requests

OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    """Cliente mínimo para el servicio local de Ollama."""

    def __init__(self, model: str = "llama3.2:3b", base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def status(self) -> tuple[bool, list[str], str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
            return True, models, "OK"
        except Exception as exc:
            return False, [], str(exc)

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0.2, "num_predict": 500},
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
        if response.status_code == 404:
            detail = ""
            try:
                detail = str(response.json().get("error", ""))
            except Exception:
                detail = response.text
            if "not found" in detail.lower():
                raise RuntimeError(
                    f"El modelo '{self.model}' no está descargado en Ollama. "
                    f"En una terminal ejecuta: ollama pull {self.model}"
                )
        response.raise_for_status()
        return response.json()["message"]["content"]


def dumps_context(context: dict) -> str:
    return json.dumps(context, ensure_ascii=False, default=str)
