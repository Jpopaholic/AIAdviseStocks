# Path: src/services/llm_client.py
import json
import time
import logging
import requests
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel

from src.config import config, safe_int

logger = logging.getLogger("UserLLMClient")


class LLMAuthenticationError(Exception):
    """使用者提供的 LLM API Key 無效或無權限時拋出之例外。"""
    pass


class LLMRateLimitError(Exception):
    """使用者提供的 LLM API Key 觸發速率限制 (429) 或配額耗盡時拋出之例外。"""
    pass


class LLMInvocationError(Exception):
    """LLM 調用過程中發生其他錯誤時拋出之例外。"""
    pass


def _mask_secret(text: str, api_key: str) -> str:
    """確保任何錯誤訊息或日誌字串皆不會洩漏使用者傳入的明文 API Key。"""
    if not text:
        return ""
    sanitized = str(text)
    if api_key:
        for k in [x.strip() for x in api_key.split(",") if x.strip()]:
            if len(k) > 4:
                sanitized = sanitized.replace(k, f"{k[:4]}***")
    return sanitized


def detect_provider(
    api_key: str,
    provider_or_model: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    根據使用者指定的 provider、model_name 或 API Key 特徵自動判斷為 'gemini' 或 'openai'。
    同時支援位置參數 detect_provider(api_key, provider) 與關鍵字參數呼叫。
    """
    cand_prov = str(provider or provider_or_model or "").strip().lower()
    if cand_prov in ("gemini", "openai"):
        return cand_prov

    m = str(model_name or provider_or_model or "").strip().lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"

    first_key = str(api_key or "").split(",")[0].strip()
    if first_key.startswith("sk-"):
        return "openai"
    return "gemini"


def resolve_model_name(provider: str, model_name: Optional[str] = None) -> str:
    """根據 provider 與使用者傳入的 model_name 決定最終模型名稱。"""
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    return config.openai_model if str(provider).lower() == "openai" else config.gemini_model


def validate_api_key_format(api_key: str) -> tuple[bool, str]:
    """檢查使用者輸入的 LLM API Key 基本格式是否有效。"""
    cleaned = str(api_key or "").strip()
    if not cleaned:
        return False, "請輸入您的 LLM API Key（支援 Google Gemini 或 OpenAI 金鑰）。"
    first_key = cleaned.split(",")[0].strip()
    if len(first_key) < 10:
        return False, "輸入的 API Key 長度過短，請確認是否完整貼上。"
    return True, ""


def _build_schema_instruction(schema_cls: Optional[Type[BaseModel]]) -> str:
    """將 Pydantic BaseModel 的 JSON Schema 轉為嚴格的結構化提示說明。"""
    if schema_cls is None or not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
        return ""
    schema_dict = schema_cls.model_json_schema()
    schema_str = json.dumps(schema_dict, ensure_ascii=False, indent=2)
    return (
        "\n\n【嚴格 JSON Schema 輸出規定】：\n"
        "請務必只輸出一個合法的 JSON 物件（不要包含 ```json 標籤或額外解釋文字），"
        "且所有欄位名稱與資料型態必須完全符合以下 JSON Schema：\n"
        f"{schema_str}"
    )


def _clean_json_text(raw_text: str) -> str:
    """清理模型偶爾包裹的 Markdown ```json 區塊，提取純 JSON 字串。"""
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _call_gemini_rest(
    api_key: str,
    prompt: str,
    system_instruction: Optional[str],
    model_name: str,
    temperature: float,
    schema_cls: Optional[Type[BaseModel]],
    timeout: int,
    usage_tracker: Optional[Dict[str, int]] = None,
) -> str:
    """
    使用 Google Gemini 官方 REST API (generativelanguage.googleapis.com) 執行當次請求。
    不將 api_key 存入任何全域設定，確保多請求隔離與記憶體安全。
    """
    clean_model = model_name.replace("models/", "").strip() or config.gemini_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"

    full_sys = system_instruction or ""
    if schema_cls is not None:
        full_sys += _build_schema_instruction(schema_cls)

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json" if schema_cls is not None else "text/plain",
        },
    }
    if full_sys.strip():
        payload["systemInstruction"] = {"parts": [{"text": full_sys}]}

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (400, 401, 403):
        err_detail = resp.text[:300]
        if "API_KEY_INVALID" in err_detail or resp.status_code in (401, 403):
            raise LLMAuthenticationError("Gemini API Key 無效或無權限，請檢查您輸入的金鑰是否正確。")
        raise LLMInvocationError(f"Gemini 請求格式或參數錯誤 (HTTP {resp.status_code}): {_mask_secret(err_detail, api_key)}")

    if resp.status_code == 429:
        raise LLMRateLimitError("Gemini API 觸發速率限制或免費額度已達上限 (HTTP 429)。")

    if resp.status_code >= 400:
        raise LLMInvocationError(f"Gemini API 呼叫失敗 (HTTP {resp.status_code}): {_mask_secret(resp.text[:300], api_key)}")

    data = resp.json()

    if usage_tracker is not None:
        usage = data.get("usageMetadata") or {}
        p_tok = safe_int(usage.get("promptTokenCount"), 0)
        c_tok = safe_int(usage.get("candidatesTokenCount"), 0)
        t_tok = safe_int(usage.get("totalTokenCount"), p_tok + c_tok)
        usage_tracker["prompt_tokens"] = usage_tracker.get("prompt_tokens", 0) + p_tok
        usage_tracker["completion_tokens"] = usage_tracker.get("completion_tokens", 0) + c_tok
        usage_tracker["total_tokens"] = usage_tracker.get("total_tokens", 0) + t_tok
        usage_tracker["requests"] = usage_tracker.get("requests", 0) + 1

    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMInvocationError("Gemini API 未回傳任何候選結果 (可能觸發安全過濾)。")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_chunks = [p.get("text", "") for p in parts if "text" in p]
    result_text = "".join(text_chunks).strip()
    if not result_text:
        raise LLMInvocationError("Gemini API 回傳空白內容。")

    return _clean_json_text(result_text) if schema_cls is not None else result_text


def _call_openai_rest(
    api_key: str,
    prompt: str,
    system_instruction: Optional[str],
    model_name: str,
    temperature: float,
    schema_cls: Optional[Type[BaseModel]],
    timeout: int,
    usage_tracker: Optional[Dict[str, int]] = None,
) -> str:
    """
    使用 OpenAI 官方 Chat Completions REST API 執行當次請求。
    """
    clean_model = model_name.strip() if model_name else config.openai_model
    if clean_model.startswith("gemini"):
        clean_model = config.openai_model

    url = "https://api.openai.com/v1/chat/completions"

    full_sys = system_instruction or ""
    if schema_cls is not None:
        full_sys += _build_schema_instruction(schema_cls)

    messages: List[Dict[str, str]] = []
    if full_sys.strip():
        messages.append({"role": "system", "content": full_sys})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": clean_model,
        "messages": messages,
    }
    is_reasoning = any(clean_model.lower().startswith(p) for p in ("o1", "o3", "o4"))
    if not is_reasoning:
        payload["temperature"] = temperature
    if schema_cls is not None:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (401, 403):
        raise LLMAuthenticationError("OpenAI API Key 無效或無權限，請確認您的金鑰是否正確。")
    if resp.status_code == 429:
        raise LLMRateLimitError("OpenAI API 觸發速率限制或帳戶額度不足 (HTTP 429)。")
    if resp.status_code >= 400:
        raise LLMInvocationError(f"OpenAI API 呼叫失敗 (HTTP {resp.status_code}): {_mask_secret(resp.text[:300], api_key)}")

    data = resp.json()
    if usage_tracker is not None:
        usage = data.get("usage") or {}
        p_tok = safe_int(usage.get("prompt_tokens"), 0)
        c_tok = safe_int(usage.get("completion_tokens"), 0)
        t_tok = safe_int(usage.get("total_tokens"), p_tok + c_tok)
        usage_tracker["prompt_tokens"] = usage_tracker.get("prompt_tokens", 0) + p_tok
        usage_tracker["completion_tokens"] = usage_tracker.get("completion_tokens", 0) + c_tok
        usage_tracker["total_tokens"] = usage_tracker.get("total_tokens", 0) + t_tok
        usage_tracker["requests"] = usage_tracker.get("requests", 0) + 1

    choices = data.get("choices") or []
    if not choices:
        raise LLMInvocationError("OpenAI API 未回傳任何結果。")

    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise LLMInvocationError("OpenAI API 回傳空白內容。")

    return _clean_json_text(content) if schema_cls is not None else content.strip()


def call_user_llm(
    api_key: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: int = 90,
    usage_tracker: Optional[Dict[str, int]] = None,
) -> str:
    """
    使用使用者自備的 LLM API Key (支援單把或多把逗號分隔金鑰) 執行 AI 調用，
    具備 Pydantic response_schema 結構化輸出驗證與 429 指數退避重試。
    """
    raw_keys = [k.strip() for k in str(api_key or "").split(",") if k.strip()]
    if not raw_keys:
        raise LLMAuthenticationError("尚未提供 LLM API Key，請先於網頁上方輸入您的自備金鑰。")

    gen_cfg = generation_config or {}
    temperature = float(gen_cfg.get("temperature", 0.0))
    schema_cls = gen_cfg.get("response_schema")

    resolved_provider = detect_provider(raw_keys[0], model_name=model_name, provider=provider)
    resolved_model = (
        str(model_name).strip()
        if (model_name and str(model_name).strip())
        else (config.openai_model if resolved_provider == "openai" else config.gemini_model)
    )

    last_error: Optional[Exception] = None
    delay = 2.0

    for attempt in range(1, max_retries + 1):
        current_key = raw_keys[(attempt - 1) % len(raw_keys)]
        try:
            if resolved_provider == "openai":
                raw_out = _call_openai_rest(
                    api_key=current_key,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=resolved_model,
                    temperature=temperature,
                    schema_cls=schema_cls,
                    timeout=timeout,
                    usage_tracker=usage_tracker,
                )
            else:
                raw_out = _call_gemini_rest(
                    api_key=current_key,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model_name=resolved_model,
                    temperature=temperature,
                    schema_cls=schema_cls,
                    timeout=timeout,
                    usage_tracker=usage_tracker,
                )

            # 若有指定 Pydantic Schema，驗證回傳字串可解析為合法 JSON 並通過 Schema 校驗
            if schema_cls is not None and isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel):
                parsed_dict = json.loads(raw_out)
                validated_obj = schema_cls.model_validate(parsed_dict)
                return json.dumps(validated_obj.model_dump(), ensure_ascii=False)

            return raw_out

        except LLMAuthenticationError:
            # 金鑰無效時不浪費時間重試同一把無效金鑰（除非使用者提供了多把金鑰）
            if len(raw_keys) <= 1:
                raise
            last_error = LLMAuthenticationError("提供的 API Key 無效或無權限。")
        except LLMRateLimitError as rate_err:
            last_error = rate_err
            logger.warning(f"LLM 調用觸發 429 速率限制 (第 {attempt}/{max_retries} 次嘗試)，等待 {delay:.1f} 秒後重試...")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2.0
        except Exception as err:
            sanitized_msg = _mask_secret(str(err), api_key)
            last_error = LLMInvocationError(sanitized_msg)
            logger.warning(f"LLM 調用異常 (第 {attempt}/{max_retries} 次嘗試): {sanitized_msg}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 1.5

    if isinstance(last_error, (LLMAuthenticationError, LLMRateLimitError)):
        raise last_error
    raise LLMInvocationError(f"LLM 調用失敗: {_mask_secret(str(last_error), api_key)}")

