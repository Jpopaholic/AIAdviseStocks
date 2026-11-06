# Path: src/web_server.py
import os
import time
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    config,
    safe_int,
    safe_float,
    get_stock_name,
    AUTO_SELECT_BOARDS,
)
from src.services.stock_fetcher import (
    fetch_taiex_klines,
    fetch_stock_klines,
    clear_stock_cache,
)
from src.services.technical_indicators import compute_all_indicators
from src.services.auto_selector import (
    list_available_boards,
    get_board_candidates_metadata,
    select_stocks_for_analysis,
)
from src.services.llm_client import (
    detect_provider,
    resolve_model_name,
    validate_api_key_format,
    LLMAuthenticationError,
    LLMRateLimitError,
)
from src.agents.regime_agent import generate_market_regime
from src.agents.analyst_agent import generate_analyst_assessments

# 台灣時區 (UTC+8)
TW_TZ = timezone(timedelta(hours=8))

# 設定基礎中文日誌
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("WebServer")

app = FastAPI(
    title="AIAdviseStocks 台股兩層式 AI 選股與技術分析系統",
    description="保留第一層大盤氣候診斷與第二層個股五維度量化評分，支援三大自動選股榜單、BYOK 自備金鑰、Vercel Serverless 與 Markdown 報告匯出",
    version="1.1.0",
)

# 掛載靜態資源目錄
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =====================================================================
# 記憶體執行狀態與報告管理 (No Database Server Architecture)
# =====================================================================
_STATE_LOCK = threading.Lock()
MAX_LOG_LINES = 250
MAX_REPORT_HISTORY = 10

ANALYSIS_STATE: Dict[str, Any] = {
    "is_running": False,
    "stop_requested": False,
    "phase": "IDLE",  # IDLE | LAYER1_REGIME | AUTO_SELECT | LAYER2_ANALYST | COMPLETED | STOPPED | ERROR
    "phase_label": "待命準備中",
    "board_key": "core_top20",
    "board_label": AUTO_SELECT_BOARDS["core_top20"]["name"],
    "progress_current": 0,
    "progress_total": 0,
    "current_stock": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    },
    "logs": [],
    "latest_report": None,
    "report_history": [],
}


def _now_tw_str() -> str:
    return datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")


def append_memory_log(message: str, level: str = "INFO") -> None:
    """將執行訊息寫入記憶體日誌與標準控制台（嚴禁包含任何 API Key）。"""
    ts = _now_tw_str()
    entry = f"[{ts}] [{level}] {message}"
    with _STATE_LOCK:
        ANALYSIS_STATE["logs"].append(entry)
        if len(ANALYSIS_STATE["logs"]) > MAX_LOG_LINES:
            ANALYSIS_STATE["logs"] = ANALYSIS_STATE["logs"][-MAX_LOG_LINES:]
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


def is_stop_requested() -> bool:
    with _STATE_LOCK:
        return bool(ANALYSIS_STATE.get("stop_requested", False))


# =====================================================================
# 結構化 Markdown (.md) 報告生成器
# =====================================================================
REGIME_CN_MAP = {
    "STRONG_BULL": "強勢多頭 (STRONG_BULL)",
    "REBOUND_BULL": "驚驚漲 / 震盪偏多 (REBOUND_BULL)",
    "CALM_RANGE": "低波動橫盤 (CALM_RANGE)",
    "VOLATILE_RANGE": "高波動震盪 (VOLATILE_RANGE)",
    "CORRECTION_BEAR": "震盪修正 (CORRECTION_BEAR)",
    "PANIC_BEAR": "恐慌空頭 (PANIC_BEAR)",
}

POSTURE_CN_MAP = {
    "STRONG_ATTACK": "強攻攻勢 (STRONG_ATTACK)",
    "MODERATE_ATTACK": "穩健進攻 (MODERATE_ATTACK)",
    "CHOPPY_TACTICAL": "震盪靈活 (CHOPPY_TACTICAL)",
    "DEFENSIVE_ACCUMULATION": "防禦承接 (DEFENSIVE_ACCUMULATION)",
    "STRICT_DEFENSE": "極度保守 (STRICT_DEFENSE)",
}


def build_markdown_report(report: Dict[str, Any]) -> str:
    """
    將第一層大盤氣候診斷、三大自動選股榜單標的與第二層個股五維度量化評分排行榜
    彙整生成結構化 Markdown (.md) 報告。
    """
    created_at = report.get("created_at", _now_tw_str())
    board_label = report.get("board_label", "自動選股榜單")
    provider_name = str(report.get("provider", "gemini")).upper()
    model_name = report.get("model_name", "")
    stopped_early = report.get("stopped_early", False)
    usage = report.get("usage") or {}
    regime_data = report.get("regime_assessment") or {}
    scores: List[Dict[str, Any]] = report.get("analyst_scores") or []
    candidates: List[Dict[str, Any]] = report.get("candidates") or []

    regime_raw = regime_data.get("regime", "CALM_RANGE")
    posture_raw = regime_data.get("posture", "CHOPPY_TACTICAL")
    regime_cn = REGIME_CN_MAP.get(regime_raw, regime_raw)
    posture_cn = POSTURE_CN_MAP.get(posture_raw, posture_raw)
    risk_mult = safe_float(regime_data.get("risk_multiplier"), 0.7)
    cash_ratio = safe_float(regime_data.get("target_cash_ratio"), 0.35) * 100.0
    allowed_styles = ", ".join(regime_data.get("allowed_buy_styles") or ["PULLBACK"])
    taiex_close = safe_float(regime_data.get("taiex_close"), 0.0)
    taiex_date = regime_data.get("taiex_date") or ""

    md_lines: List[str] = [
        f"# 📊 AIAdviseStocks 台股兩層式 AI 量化分析報告",
        "",
        f"- **報告生成時間 (UTC+8)**：`{created_at}`",
        f"- **選股榜單模式**：**{board_label}**",
        f"- **使用 AI 模型**：`{provider_name} / {model_name}`",
        f"- **分析完成檔數**：`{len(scores)} / {len(candidates) or len(scores)} 檔`"
        + (" *(使用者手動提前終止)*" if stopped_early else ""),
        f"- **Token 消耗統計**：Prompt `{usage.get('prompt_tokens', 0):,}` + Completion `{usage.get('completion_tokens', 0):,}` = 總計 `{usage.get('total_tokens', 0):,}` Tokens（呼叫 `{usage.get('calls', 0)}` 次）",
        "",
        "---",
        "",
        "## 🧭 第一層：大盤氣候診斷報告 (Layer 1 Market Regime Assessment)",
        "",
        "| 診斷項目 | 評估結果 |",
        "| :--- | :--- |",
        f"| **大盤加權指數 (TAIEX)** | `{taiex_close:,.2f}` 點 ({taiex_date}) |",
        f"| **六階市場狀態 (Regime)** | **{regime_cn}** |",
        f"| **五階交易姿態 (Posture)** | **{posture_cn}** |",
        f"| **建議最低現金儲備比** | **{cash_ratio:.0f}%** |",
        f"| **風險限額乘數 (Risk Multiplier)** | `{risk_mult:.2f}` |",
        f"| **建議操作型態 (Allowed Styles)** | `{allowed_styles}` |",
        "",
        "### 🎯 戰術特別指令 (Tactical Directive)",
        f"> {regime_data.get('tactical_directive', '無')}",
        "",
        "### 📝 大盤宏觀分析依據",
        f"{regime_data.get('reason', '無')}",
        "",
        "---",
        "",
        f"## 🏆 第二層：個股五維度量化評分排行榜 ({board_label})",
        "",
    ]

    if not scores:
        md_lines.append("*本輪無個股評分資料。*")
    else:
        md_lines.extend([
            "| 排名 | 代號 | 名稱 | 收盤價 | 漲跌幅 | 成交金額(億) | 總分 (100) | 趨勢(20) | 動能(20) | 量能(20) | 安全(20) | 大盤(20) | 信心 |",
            "| :---: | :---: | :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for item in scores:
            rk = item.get("rank", "-")
            code = item.get("stock_code", "")
            name = item.get("stock_name", get_stock_name(code))
            price = safe_float(item.get("price"), 0.0)
            chg = safe_float(item.get("change_pct"), 0.0)
            chg_str = f"+{chg:.2f}%" if chg > 0 else f"{chg:.2f}%"
            turnover_yi = safe_float(item.get("turnover"), 0.0) / 1e8
            tot = safe_int(item.get("total_score"), 0)
            tr = safe_int(item.get("trend_score"), 0)
            mo = safe_int(item.get("momentum_score"), 0)
            vo = safe_int(item.get("volume_score"), 0)
            sa = safe_int(item.get("safety_score"), 0)
            re = safe_int(item.get("regime_score"), 0)
            conf = safe_float(item.get("confidence"), 0.0) * 100.0
            md_lines.append(
                f"| **{rk}** | `{code}` | {name} | {price:,.2f} | {chg_str} | {turnover_yi:,.2f} | **{tot}** | {tr} | {mo} | {vo} | {sa} | {re} | {conf:.0f}% |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## 🔍 個股技術指標與 AI 分析師評分詳情",
            "",
        ])

        for item in scores:
            rk = item.get("rank", "-")
            code = item.get("stock_code", "")
            name = item.get("stock_name", get_stock_name(code))
            tot = safe_int(item.get("total_score"), 0)
            price = safe_float(item.get("price"), 0.0)
            chg = safe_float(item.get("change_pct"), 0.0)
            chg_str = f"+{chg:.2f}%" if chg > 0 else f"{chg:.2f}%"
            ma5 = item.get("ma5")
            ma20 = item.get("ma20")
            ma60 = item.get("ma60")
            rsi14 = item.get("rsi14")
            atr_pct = safe_float(item.get("atr_pct"), 0.0)
            b_label = item.get("board_label") or board_label

            md_lines.extend([
                f"### #{rk} `{code}` {name} — 總分：**{tot} / 100 分**",
                f"- **所屬榜單**：{b_label} ｜ **最新收盤價**：`{price:,.2f}` 元 (`{chg_str}`) ｜ **資料日期**：`{item.get('latest_date', '')}`",
                f"- **五維度得分**：趨勢 `{item.get('trend_score')}/20` ｜ 動能 `{item.get('momentum_score')}/20` ｜ 量能 `{item.get('volume_score')}/20` ｜ 安全防守 `{item.get('safety_score')}/20` ｜ 大盤一致性 `{item.get('regime_score')}/20` （信心指數：`{safe_float(item.get('confidence'), 0.0) * 100:.0f}%`）",
                f"- **關鍵技術指標**：MA5 `{ma5 if ma5 is not None else 'N/A'}` ｜ MA20 `{ma20 if ma20 is not None else 'N/A'}` ｜ MA60 `{ma60 if ma60 is not None else 'N/A'}` ｜ RSI(14) `{rsi14 if rsi14 is not None else 'N/A'}` ｜ ATR 波動率 `{atr_pct:.2f}%`",
                f"- **AI 技術分析理由**：{item.get('reason', '')}",
                "",
            ])

    md_lines.extend([
        "---",
        "*免責聲明：本報告由 AIAdviseStocks 兩層式量化技術分析系統自動產出，僅供技術面研究與學術參考，不包含任何買賣決策建議或投資勸誘。投資人應獨立判斷並自負風險。*",
        "",
    ])
    return "\n".join(md_lines)


# =====================================================================
# 背景兩層式分析工作執行緒 (供本地常駐伺服器模式使用)
# =====================================================================
def _run_two_layer_analysis_task(
    api_key: str,
    provider: str,
    model_name: str,
    board_key: str,
    top_n: int,
    custom_codes: Optional[List[str]] = None,
) -> None:
    usage_tracker = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    board_info = AUTO_SELECT_BOARDS.get(board_key, AUTO_SELECT_BOARDS["core_top20"])
    board_label = board_info["name"]

    try:
        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "LAYER1_REGIME"
            ANALYSIS_STATE["phase_label"] = "第一層：正在擷取 TAIEX 大盤 K 線並診斷市場氣候..."
            ANALYSIS_STATE["progress_current"] = 0
            ANALYSIS_STATE["progress_total"] = top_n + 1

        append_memory_log(
            f"啟動兩層式分析任務｜榜單：【{board_label}】｜模型：{provider.upper()} / {model_name}"
        )
        taiex_klines = fetch_taiex_klines(days=90)
        if is_stop_requested():
            raise InterruptedError("使用者已請求停止分析任務。")

        regime_assessment = generate_market_regime(
            taiex_klines=taiex_klines,
            api_key=api_key,
            model_name=model_name,
            provider=provider,
            usage_tracker=usage_tracker,
        )
        with _STATE_LOCK:
            ANALYSIS_STATE["usage"] = dict(usage_tracker)
            ANALYSIS_STATE["progress_current"] = 1

        if is_stop_requested():
            raise InterruptedError("使用者已請求停止分析任務。")

        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "AUTO_SELECT"
            ANALYSIS_STATE["phase_label"] = f"正在依【{board_label}】規則自動篩選標的並載入日 K 線..."

        selection_data = select_stocks_for_analysis(
            board_key=board_key,
            top_n=top_n,
            kline_days=config.kline_Lookback_days,
            include_taiex=False,
            custom_codes=custom_codes,
        )
        selected_codes: List[str] = selection_data.get("selected_codes") or []
        candidates_meta: List[Dict[str, Any]] = selection_data.get("candidates") or []
        klines_map: Dict[str, List[Dict[str, Any]]] = selection_data.get("klines_map") or {}
        klines_map["TAIEX"] = taiex_klines

        if not selected_codes:
            raise RuntimeError(f"榜單【{board_label}】未能篩選出有效 K 線標的，請稍後重試。")

        if is_stop_requested():
            raise InterruptedError("使用者已請求停止分析任務。")

        total_steps = len(selected_codes) + 1
        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "LAYER2_ANALYST"
            ANALYSIS_STATE["progress_total"] = total_steps

        def _on_stock_progress(idx: int, total: int, code: str, _partial: Optional[Dict[str, Any]]) -> None:
            name = get_stock_name(code)
            with _STATE_LOCK:
                ANALYSIS_STATE["progress_current"] = idx + 1
                ANALYSIS_STATE["progress_total"] = total + 1
                ANALYSIS_STATE["current_stock"] = f"{code} {name}"
                ANALYSIS_STATE["phase_label"] = f"第二層：正在評分 ({idx}/{total}) {code} {name}..."
                ANALYSIS_STATE["usage"] = dict(usage_tracker)
            append_memory_log(f"第二層個股分析 ({idx}/{total})：正在評估 {code} ({name}) 五維度技術指標...")

        analyst_scores = generate_analyst_assessments(
            stock_codes=selected_codes,
            klines_map=klines_map,
            api_key=api_key,
            model_name=model_name,
            provider=provider,
            regime_assessment=regime_assessment,
            candidates_metadata=candidates_meta,
            usage_tracker=usage_tracker,
            should_stop_fn=is_stop_requested,
            progress_callback=_on_stock_progress,
        )

        stopped_early = is_stop_requested()
        now_str = _now_tw_str()
        report_id = datetime.now(TW_TZ).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        report_payload: Dict[str, Any] = {
            "report_id": report_id,
            "created_at": now_str,
            "board_key": board_key,
            "board_label": board_label,
            "provider": provider,
            "model_name": model_name,
            "stopped_early": stopped_early,
            "usage": dict(usage_tracker),
            "regime_assessment": regime_assessment,
            "candidates": candidates_meta,
            "analyst_scores": analyst_scores,
        }
        report_payload["markdown_content"] = build_markdown_report(report_payload)

        with _STATE_LOCK:
            ANALYSIS_STATE["usage"] = dict(usage_tracker)
            ANALYSIS_STATE["latest_report"] = report_payload
            ANALYSIS_STATE["report_history"].insert(0, report_payload)
            if len(ANALYSIS_STATE["report_history"]) > MAX_REPORT_HISTORY:
                ANALYSIS_STATE["report_history"] = ANALYSIS_STATE["report_history"][:MAX_REPORT_HISTORY]
            ANALYSIS_STATE["finished_at"] = now_str
            ANALYSIS_STATE["current_stock"] = ""
            if stopped_early:
                ANALYSIS_STATE["phase"] = "STOPPED"
                ANALYSIS_STATE["phase_label"] = f"已依指令提前終止（已完成 {len(analyst_scores)} 檔評分）"
            else:
                ANALYSIS_STATE["phase"] = "COMPLETED"
                ANALYSIS_STATE["phase_label"] = f"兩層式分析完成！共完成 {len(analyst_scores)} 檔五維度評分"
                ANALYSIS_STATE["progress_current"] = ANALYSIS_STATE["progress_total"]

    except InterruptedError as stop_err:
        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "STOPPED"
            ANALYSIS_STATE["phase_label"] = "分析任務已安全終止"
            ANALYSIS_STATE["finished_at"] = _now_tw_str()
        append_memory_log(str(stop_err), level="WARNING")

    except (LLMAuthenticationError, LLMRateLimitError) as llm_err:
        err_msg = str(llm_err)
        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "ERROR"
            ANALYSIS_STATE["phase_label"] = "LLM 金鑰驗證或配額錯誤"
            ANALYSIS_STATE["error"] = err_msg
            ANALYSIS_STATE["finished_at"] = _now_tw_str()
        append_memory_log(f"AI 調用中斷：{err_msg}", level="ERROR")

    except Exception as e:
        err_msg = f"分析過程發生未預期錯誤: {str(e)}"
        with _STATE_LOCK:
            ANALYSIS_STATE["phase"] = "ERROR"
            ANALYSIS_STATE["phase_label"] = "分析執行發生異常"
            ANALYSIS_STATE["error"] = err_msg
            ANALYSIS_STATE["finished_at"] = _now_tw_str()
        append_memory_log(err_msg, level="ERROR")

    finally:
        with _STATE_LOCK:
            ANALYSIS_STATE["is_running"] = False
            ANALYSIS_STATE["stop_requested"] = False


# =====================================================================
# API 請求模型與路由端點
# =====================================================================
class AnalyzeStartRequest(BaseModel):
    api_key: str = Field(..., description="使用者自備的 LLM API Key (Gemini 或 OpenAI)")
    provider: Optional[str] = Field("auto", description="'auto', 'gemini', 或 'openai'")
    model_name: Optional[str] = Field(None, description="指定模型名稱，如 gemini-2.5-flash 或 gpt-4o-mini")
    board_key: str = Field(
        "core_top20",
        description="三大自動選股榜單代號：'core_top20' (大盤核心榜)、'under100_top20' (百元精選榜)、'popular_etf' (人氣 ETF 榜)",
    )
    top_n: int = Field(20, ge=1, le=20, description="挑選分析檔數上限 (預設 20 檔)")
    custom_codes: Optional[List[str]] = Field(None, description="選填：自訂額外納入分析的股票代號")


class StepLayer2StockRequest(BaseModel):
    api_key: str = Field(..., description="使用者自備的 LLM API Key")
    provider: Optional[str] = Field("auto", description="'auto', 'gemini', 或 'openai'")
    model_name: Optional[str] = Field(None, description="指定模型名稱")
    stock_code: str = Field(..., description="單一待分析股票代號")
    candidate_meta: Optional[Dict[str, Any]] = Field(None, description="該股榜單中繼資料")
    regime_assessment: Optional[Dict[str, Any]] = Field(None, description="第一層大盤氣候診斷結果")


class StepMarkdownRequest(BaseModel):
    board_key: str = Field("core_top20")
    board_label: str = Field("大盤核心榜（成交金額 Top 20）")
    provider: str = Field("gemini")
    model_name: str = Field("gemini-2.5-flash")
    stopped_early: bool = Field(False)
    usage: Dict[str, Any] = Field(default_factory=dict)
    regime_assessment: Dict[str, Any] = Field(default_factory=dict)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    analyst_scores: List[Dict[str, Any]] = Field(default_factory=list)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    """提供主頁深色金融分析儀表板介面 (src/static/index.html)。"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse(
            content="<h1>AIAdviseStocks Web Server 已啟動，但尚未建立 src/static/index.html</h1>",
            status_code=200,
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/api/boards")
async def api_list_boards() -> Dict[str, Any]:
    """取得三大自動選股榜單定義與系統預設模型資訊。"""
    return {
        "boards": list_available_boards(),
        "default_models": {
            "gemini": config.gemini_model,
            "openai": config.openai_model,
        },
        "recommended_models": [
            {"provider": "gemini", "model": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (推薦・高速低成本)"},
            {"provider": "gemini", "model": "gemini-2.5-pro", "label": "Gemini 2.5 Pro (深度推理)"},
            {"provider": "openai", "model": "gpt-4o-mini", "label": "OpenAI GPT-4o mini (高速穩定)"},
            {"provider": "openai", "model": "gpt-4o", "label": "OpenAI GPT-4o (高精準度)"},
        ],
    }


@app.get("/api/boards/{board_key}/preview")
async def api_preview_board(
    board_key: str,
    top_n: int = Query(20, ge=1, le=30),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    """預覽三大自動選股榜單的候選股票清單（含最新收盤價、漲跌幅與成交金額）。"""
    if board_key not in AUTO_SELECT_BOARDS:
        raise HTTPException(status_code=400, detail=f"不支援的選股榜單代號: {board_key}")
    try:
        if refresh:
            clear_stock_cache()
        candidates = get_board_candidates_metadata(
            board_key=board_key,
            top_n=top_n,
            force_refresh=refresh,
        )
        board_info = AUTO_SELECT_BOARDS[board_key]
        return {
            "board_key": board_key,
            "board_name": board_info["name"],
            "board_description": board_info["description"],
            "count": len(candidates),
            "candidates": candidates,
            "updated_at": _now_tw_str(),
        }
    except Exception as e:
        logger.error(f"預覽榜單 {board_key} 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"預覽榜單失敗: {str(e)}")


# =====================================================================
# Vercel Serverless 無狀態逐步執行端點 (防止 Serverless Timeout 與跨實例失憶)
# =====================================================================
@app.post("/api/step/layer1")
async def api_step_layer1(req: AnalyzeStartRequest) -> Dict[str, Any]:
    """
    [Serverless 步驟 1]：驗證使用者 API Key、擷取 TAIEX K 線執行第一層大盤氣候診斷，
    並回傳所選三大榜單的入選候選股票清單。單次請求約 2~4 秒內完成，完美相容 Vercel Serverless。
    """
    is_valid, err_reason = validate_api_key_format(req.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_reason)

    board_key = req.board_key if req.board_key in AUTO_SELECT_BOARDS else "core_top20"
    board_label = AUTO_SELECT_BOARDS[board_key]["name"]
    resolved_provider = detect_provider(req.api_key, req.provider)
    resolved_model = resolve_model_name(resolved_provider, req.model_name)
    top_n = safe_int(req.top_n, default=20, min_val=1, max_val=20)

    usage_tracker = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    try:
        taiex_klines = fetch_taiex_klines(days=90)
        regime_assessment = generate_market_regime(
            taiex_klines=taiex_klines,
            api_key=req.api_key.strip(),
            model_name=resolved_model,
            provider=resolved_provider,
            usage_tracker=usage_tracker,
        )
        candidates = get_board_candidates_metadata(
            board_key=board_key,
            top_n=top_n,
            force_refresh=False,
        )
        return {
            "board_key": board_key,
            "board_label": board_label,
            "provider": resolved_provider,
            "model_name": resolved_model,
            "regime_assessment": regime_assessment,
            "candidates": candidates,
            "usage": usage_tracker,
            "timestamp": _now_tw_str(),
        }
    except (LLMAuthenticationError, LLMRateLimitError) as llm_err:
        raise HTTPException(status_code=401, detail=str(llm_err))
    except Exception as e:
        logger.error(f"[Step Layer1] 執行失敗: {e}")
        raise HTTPException(status_code=500, detail=f"第一層大盤診斷或選股失敗: {str(e)}")


@app.post("/api/step/layer2-stock")
async def api_step_layer2_stock(req: StepLayer2StockRequest) -> Dict[str, Any]:
    """
    [Serverless 步驟 2]：針對單一入選股票擷取日 K 線、計算技術指標並執行第二層五維度量化評分。
    單次請求約 1.5~3 秒完成，支援前端即時更新排行榜與隨時按 Stop 停止。
    """
    is_valid, err_reason = validate_api_key_format(req.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_reason)

    code = str(req.stock_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少股票代號 stock_code")

    resolved_provider = detect_provider(req.api_key, req.provider)
    resolved_model = resolve_model_name(resolved_provider, req.model_name)
    usage_tracker = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    try:
        stock_klines = fetch_stock_klines(code, days=config.kline_Lookback_days)
        if not stock_klines:
            raise HTTPException(status_code=404, detail=f"無法擷取股票 {code} 的 K 線數據")

        compute_all_indicators(stock_klines)
        taiex_klines = fetch_taiex_klines(days=65)

        klines_map = {
            code: stock_klines,
            "TAIEX": taiex_klines,
        }
        candidates_meta = [req.candidate_meta] if req.candidate_meta else [{"stock_code": code}]

        scores = generate_analyst_assessments(
            stock_codes=[code],
            klines_map=klines_map,
            api_key=req.api_key.strip(),
            model_name=resolved_model,
            provider=resolved_provider,
            regime_assessment=req.regime_assessment,
            candidates_metadata=candidates_meta,
            usage_tracker=usage_tracker,
            inter_call_delay_sec=0.0,
        )
        if not scores:
            raise RuntimeError(f"股票 {code} 未能產出有效評分")

        return {
            "score": scores[0],
            "usage": usage_tracker,
            "timestamp": _now_tw_str(),
        }
    except (LLMAuthenticationError, LLMRateLimitError) as llm_err:
        raise HTTPException(status_code=401, detail=str(llm_err))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Step Layer2 Stock] 評估 {code} 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"評估股票 {code} 失敗: {str(e)}")


@app.post("/api/step/markdown")
async def api_step_build_markdown(req: StepMarkdownRequest) -> Dict[str, Any]:
    """
    [Serverless 步驟 3]：接收已完成的第一層大盤診斷與第二層個股五維度評分列表，
    排序並生成結構化 Markdown (.md) 報告回傳給前端直接預覽與下載。
    """
    scores = list(req.analyst_scores or [])
    scores.sort(
        key=lambda x: (safe_int(x.get("total_score"), 0), safe_float(x.get("confidence"), 0.0)),
        reverse=True,
    )
    for idx, item in enumerate(scores, start=1):
        item["rank"] = idx

    now_str = _now_tw_str()
    report_id = datetime.now(TW_TZ).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    report_payload: Dict[str, Any] = {
        "report_id": report_id,
        "created_at": now_str,
        "board_key": req.board_key,
        "board_label": req.board_label,
        "provider": req.provider,
        "model_name": req.model_name,
        "stopped_early": req.stopped_early,
        "usage": req.usage,
        "regime_assessment": req.regime_assessment,
        "candidates": req.candidates,
        "analyst_scores": scores,
    }
    report_payload["markdown_content"] = build_markdown_report(report_payload)

    with _STATE_LOCK:
        ANALYSIS_STATE["latest_report"] = report_payload
        ANALYSIS_STATE["report_history"].insert(0, report_payload)
        if len(ANALYSIS_STATE["report_history"]) > MAX_REPORT_HISTORY:
            ANALYSIS_STATE["report_history"] = ANALYSIS_STATE["report_history"][:MAX_REPORT_HISTORY]

    return report_payload


# =====================================================================
# 本地常駐背景任務與報告下載端點
# =====================================================================
@app.post("/api/analyze/start")
async def api_start_analysis(req: AnalyzeStartRequest) -> Dict[str, Any]:
    """接收使用者自備的 LLM API Key 與所選自動選股榜單，啟動兩層式背景分析任務。"""
    is_valid, err_reason = validate_api_key_format(req.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_reason)

    board_key = req.board_key if req.board_key in AUTO_SELECT_BOARDS else "core_top20"
    resolved_provider = detect_provider(req.api_key, req.provider)
    resolved_model = resolve_model_name(resolved_provider, req.model_name)
    top_n = safe_int(req.top_n, default=20, min_val=1, max_val=20)

    with _STATE_LOCK:
        if ANALYSIS_STATE["is_running"]:
            raise HTTPException(status_code=409, detail="目前已有正在執行的分析任務，請等待完成或先點擊停止。")

        board_label = AUTO_SELECT_BOARDS[board_key]["name"]
        ANALYSIS_STATE["is_running"] = True
        ANALYSIS_STATE["stop_requested"] = False
        ANALYSIS_STATE["phase"] = "LAYER1_REGIME"
        ANALYSIS_STATE["phase_label"] = "準備啟動第一層大盤氣候診斷..."
        ANALYSIS_STATE["board_key"] = board_key
        ANALYSIS_STATE["board_label"] = board_label
        ANALYSIS_STATE["progress_current"] = 0
        ANALYSIS_STATE["progress_total"] = top_n + 1
        ANALYSIS_STATE["current_stock"] = ""
        ANALYSIS_STATE["started_at"] = _now_tw_str()
        ANALYSIS_STATE["finished_at"] = None
        ANALYSIS_STATE["error"] = None
        ANALYSIS_STATE["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        ANALYSIS_STATE["logs"] = []

    worker = threading.Thread(
        target=_run_two_layer_analysis_task,
        kwargs={
            "api_key": req.api_key.strip(),
            "provider": resolved_provider,
            "model_name": resolved_model,
            "board_key": board_key,
            "top_n": top_n,
            "custom_codes": req.custom_codes,
        },
        daemon=True,
    )
    worker.start()

    return {
        "status": "started",
        "board_key": board_key,
        "board_label": AUTO_SELECT_BOARDS[board_key]["name"],
        "provider": resolved_provider,
        "model_name": resolved_model,
        "top_n": top_n,
        "started_at": ANALYSIS_STATE["started_at"],
    }


@app.post("/api/analyze/stop")
async def api_stop_analysis() -> Dict[str, Any]:
    """設定停止執行旗標 (stop_requested = True)，通知背景任務於下一步驟前安全終止。"""
    with _STATE_LOCK:
        if not ANALYSIS_STATE["is_running"]:
            return {"status": "idle", "message": "目前沒有正在運行的分析任務。"}
        ANALYSIS_STATE["stop_requested"] = True
        ANALYSIS_STATE["phase_label"] = "已發送停止訊號，等待當前步驟完成後安全終止..."

    append_memory_log("收到使用者停止請求 (stop_requested=True)，將於當前個股完成後安全終止。", level="WARNING")
    return {"status": "stopping", "stop_requested": True}


@app.get("/api/analyze/status")
async def api_get_status() -> Dict[str, Any]:
    """輪詢取得目前背景分析進度、記憶體日誌、Token 消耗與最新分析報告。"""
    with _STATE_LOCK:
        history_meta = [
            {
                "report_id": r.get("report_id"),
                "created_at": r.get("created_at"),
                "board_key": r.get("board_key"),
                "board_label": r.get("board_label"),
                "provider": r.get("provider"),
                "model_name": r.get("model_name"),
                "stock_count": len(r.get("analyst_scores") or []),
                "regime": (r.get("regime_assessment") or {}).get("regime"),
                "posture": (r.get("regime_assessment") or {}).get("posture"),
            }
            for r in ANALYSIS_STATE["report_history"]
        ]
        return {
            "is_running": ANALYSIS_STATE["is_running"],
            "stop_requested": ANALYSIS_STATE["stop_requested"],
            "phase": ANALYSIS_STATE["phase"],
            "phase_label": ANALYSIS_STATE["phase_label"],
            "board_key": ANALYSIS_STATE["board_key"],
            "board_label": ANALYSIS_STATE["board_label"],
            "progress_current": ANALYSIS_STATE["progress_current"],
            "progress_total": ANALYSIS_STATE["progress_total"],
            "current_stock": ANALYSIS_STATE["current_stock"],
            "started_at": ANALYSIS_STATE["started_at"],
            "finished_at": ANALYSIS_STATE["finished_at"],
            "error": ANALYSIS_STATE["error"],
            "usage": dict(ANALYSIS_STATE["usage"]),
            "logs": list(ANALYSIS_STATE["logs"]),
            "latest_report": ANALYSIS_STATE["latest_report"],
            "report_history": history_meta,
        }


@app.get("/api/report/latest")
async def api_get_latest_report(report_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """取得最新一次或指定 report_id 的完整兩層式分析報告（含 Markdown 文本）。"""
    with _STATE_LOCK:
        if report_id:
            for r in ANALYSIS_STATE["report_history"]:
                if r.get("report_id") == report_id:
                    return r
            raise HTTPException(status_code=404, detail=f"找不到指定的分析報告: {report_id}")

        if not ANALYSIS_STATE["latest_report"]:
            raise HTTPException(status_code=404, detail="目前尚無已完成的分析報告。")
        return ANALYSIS_STATE["latest_report"]


@app.get("/api/report/markdown")
async def api_download_markdown_report(report_id: Optional[str] = Query(None)) -> Response:
    """下載當次或指定 report_id 的結構化 Markdown (.md) 分析報告檔案。"""
    with _STATE_LOCK:
        target_report = None
        if report_id:
            for r in ANALYSIS_STATE["report_history"]:
                if r.get("report_id") == report_id:
                    target_report = r
                    break
        else:
            target_report = ANALYSIS_STATE["latest_report"]

    if not target_report:
        raise HTTPException(status_code=404, detail="尚無可供下載的分析報告，請先執行一次分析。")

    md_text = target_report.get("markdown_content") or build_markdown_report(target_report)
    board_key = target_report.get("board_key", "report")
    rid = target_report.get("report_id", datetime.now(TW_TZ).strftime("%Y%m%d_%H%M%S"))
    filename = f"AIAdviseStocks_{board_key}_{rid}.md"
    encoded_filename = quote(filename)

    return Response(
        content=md_text.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"; filename*=UTF-8''{encoded_filename}"
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.web_server:app",
        host=config.web_host,
        port=config.web_port,
        reload=False,
    )
