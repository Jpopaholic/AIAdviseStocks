import json
import pytest
from fastapi.testclient import TestClient

from src.config import (
    config,
    safe_int,
    safe_float,
    get_stock_name,
    AUTO_SELECT_BOARDS,
    POPULAR_ETF_CODES,
)
from src.services.technical_indicators import compute_all_indicators
from src.services.stock_fetcher import (
    fetch_taiex_klines,
    fetch_stock_klines,
    fetch_market_turnover_snapshots,
)
from src.services.auto_selector import (
    list_available_boards,
    get_board_candidates_metadata,
    select_stocks_for_analysis,
)
from src.services.llm_client import (
    detect_provider,
    validate_api_key_format,
    _mask_secret,
)
from src.agents.regime_agent import generate_market_regime
from src.agents.analyst_agent import generate_analyst_assessments
from src.web_server import app, build_markdown_report, ANALYSIS_STATE


def test_config_and_safe_casters():
    assert safe_int("18", default=0, min_val=0, max_val=20) == 18
    assert safe_int("999", default=10, min_val=0, max_val=20) == 20
    assert safe_float("0.85", default=0.5, min_val=0.0, max_val=1.0) == 0.85
    assert "core_top20" in AUTO_SELECT_BOARDS
    assert "under100_top20" in AUTO_SELECT_BOARDS
    assert "popular_etf" in AUTO_SELECT_BOARDS
    assert "0050" in POPULAR_ETF_CODES
    assert "0056" in POPULAR_ETF_CODES
    assert "00878" in POPULAR_ETF_CODES
    assert "00929" in POPULAR_ETF_CODES
    assert get_stock_name("2330") == "台積電"


def test_technical_indicators_engine():
    klines = fetch_stock_klines("2330", days=65)
    assert len(klines) >= 60
    compute_all_indicators(klines)
    last_k = klines[-1]
    assert last_k.get("ma5") is not None
    assert last_k.get("ma20") is not None
    assert last_k.get("ma60") is not None
    assert last_k.get("rsi14") is not None
    assert last_k.get("macd") is not None
    assert last_k.get("atr14") is not None


def test_three_auto_selection_leaderboards():
    boards = list_available_boards()
    assert len(boards) == 3

    # 1. 大盤核心榜
    core_cands = get_board_candidates_metadata("core_top20", top_n=5)
    assert len(core_cands) == 5
    assert core_cands[0]["turnover"] >= core_cands[-1]["turnover"]

    # 2. 百元精選榜 (股價 < 100 元)
    under100_cands = get_board_candidates_metadata("under100_top20", top_n=5)
    assert len(under100_cands) > 0
    for c in under100_cands:
        assert c["close"] < 100.0

    # 3. 人氣 ETF 榜
    etf_cands = get_board_candidates_metadata("popular_etf", top_n=6)
    etf_codes = [c["stock_code"] for c in etf_cands]
    assert "0050" in etf_codes
    assert "0056" in etf_codes


def test_llm_client_key_masking_and_validation():
    fake_key = "AIzaSyTestSecretKey1234567890"
    masked = _mask_secret(f"Error with key {fake_key}", fake_key)
    assert fake_key not in masked
    assert detect_provider("sk-proj-1234567890abcdefghij", "auto") == "openai"
    assert detect_provider(fake_key, "auto") == "gemini"
    ok, _ = validate_api_key_format(fake_key)
    assert ok is True


def test_layer1_and_layer2_agents_with_mock_llm():
    taiex_klines = fetch_taiex_klines(days=65)

    def mock_regime_llm(**kwargs):
        return json.dumps({
            "regime": "STRONG_BULL",
            "posture": "STRONG_ATTACK",
            "risk_multiplier": 0.9,
            "target_cash_ratio": 0.10,
            "allowed_buy_styles": ["BREAKOUT", "PULLBACK"],
            "tactical_directive": "大盤均線多頭排列，維持低現金儲備積極布局。",
            "reason": "TAIEX 站穩 MA5、MA20 與 MA60 之上，量能溫和放大。",
        })

    regime_res = generate_market_regime(
        taiex_klines=taiex_klines,
        call_llm_fn=mock_regime_llm,
    )
    assert regime_res["regime"] == "STRONG_BULL"
    assert regime_res["posture"] == "STRONG_ATTACK"
    assert regime_res["target_cash_ratio"] == 0.10

    selection = select_stocks_for_analysis("popular_etf", top_n=2, kline_days=65, include_taiex=False)
    klines_map = selection["klines_map"]
    klines_map["TAIEX"] = taiex_klines

    def mock_analyst_llm(**kwargs):
        return json.dumps({
            "stock_code": "0050",
            "trend_score": 18,
            "momentum_score": 16,
            "volume_score": 15,
            "safety_score": 17,
            "regime_score": 18,
            "confidence": 0.88,
            "reason": "0050 均線呈現多頭排列，MA5 向上穿越 MA20 且 RSI 維持於 60 強勢區間，量價配合得宜且與大盤多頭方向一致。",
        })

    scores = generate_analyst_assessments(
        stock_codes=selection["selected_codes"],
        klines_map=klines_map,
        regime_assessment=regime_res,
        candidates_metadata=selection["candidates"],
        call_llm_fn=mock_analyst_llm,
        inter_call_delay_sec=0.0,
    )
    assert len(scores) == 2
    assert scores[0]["total_score"] == 84
    assert scores[0]["rank"] == 1


def test_web_server_endpoints_and_markdown_export():
    client = TestClient(app)

    # 主頁 HTML
    resp_home = client.get("/")
    assert resp_home.status_code == 200
    assert "AIAdviseStocks" in resp_home.text

    # 三大榜單列表
    resp_boards = client.get("/api/boards")
    assert resp_boards.status_code == 200
    assert len(resp_boards.json()["boards"]) == 3

    # 預覽人氣 ETF 榜
    resp_preview = client.get("/api/boards/popular_etf/preview?top_n=4")
    assert resp_preview.status_code == 200
    assert resp_preview.json()["board_key"] == "popular_etf"

    # 模擬寫入一份報告並測試 Markdown (.md) 下載
    dummy_report = {
        "report_id": "20261106_223000_test01",
        "created_at": "2026-11-06 22:30:00",
        "board_key": "core_top20",
        "board_label": "大盤核心榜（成交金額 Top 20）",
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "stopped_early": False,
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500, "calls": 3},
        "regime_assessment": {
            "regime": "REBOUND_BULL",
            "posture": "MODERATE_ATTACK",
            "risk_multiplier": 0.8,
            "target_cash_ratio": 0.2,
            "allowed_buy_styles": ["PULLBACK"],
            "tactical_directive": "穩健進攻，逢拉回支撐布局。",
            "reason": "指數站上月線 MA20。",
            "taiex_close": 23150.5,
            "taiex_date": "2026-11-06",
        },
        "candidates": [{"stock_code": "2330", "stock_name": "台積電"}],
        "analyst_scores": [
            {
                "rank": 1,
                "stock_code": "2330",
                "stock_name": "台積電",
                "board_key": "core_top20",
                "board_label": "大盤核心榜（成交金額 Top 20）",
                "price": 1050.0,
                "change_pct": 1.94,
                "turnover": 35000000000,
                "latest_date": "2026-11-06",
                "trend_score": 18,
                "momentum_score": 17,
                "volume_score": 16,
                "safety_score": 15,
                "regime_score": 18,
                "total_score": 84,
                "confidence": 0.9,
                "ma5": 1040.0,
                "ma20": 1020.0,
                "ma60": 980.0,
                "rsi14": 62.5,
                "atr_pct": 2.1,
                "volatility_tier": "NORMAL",
                "reason": "台積電站穩所有均線之上，MACD 紅柱擴張且成交量溫和放大，技術面強勢。",
            }
        ],
    }
    dummy_report["markdown_content"] = build_markdown_report(dummy_report)
    ANALYSIS_STATE["latest_report"] = dummy_report
    ANALYSIS_STATE["report_history"] = [dummy_report]

    resp_md = client.get("/api/report/markdown")
    assert resp_md.status_code == 200
    assert "text/markdown" in resp_md.headers["content-type"]
    assert "2330" in resp_md.text
    assert "第一層：大盤氣候診斷報告" in resp_md.text
    assert "第二層：個股五維度量化評分排行榜" in resp_md.text

