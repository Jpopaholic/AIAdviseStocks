# Path: src/services/auto_selector.py
import logging
from typing import List, Dict, Any, Optional

from src.config import (
    config,
    safe_float,
    safe_int,
    get_stock_name,
    get_board_config,
    AUTO_SELECT_BOARDS,
    POPULAR_ETF_CODES,
    CORE_MARKET_FALLBACK_POOL,
    UNDER_100_FALLBACK_POOL,
)
from src.services.stock_fetcher import (
    fetch_market_turnover_snapshots,
    fetch_stock_klines,
    fetch_taiex_klines,
)
from src.services.technical_indicators import compute_all_indicators

logger = logging.getLogger("AutoSelector")


def _build_candidate_entry(
    rank: int,
    stock_code: str,
    board_cfg: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]] = None,
    klines: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """將入選股票封裝為包含代號、中文名稱、最新收盤價、成交金額與榜單標籤之標準結構。"""
    latest_k = klines[-1] if klines else {}
    prev_k = klines[-2] if (klines and len(klines) >= 2) else latest_k

    close_val = safe_float(
        (snapshot or {}).get("close") or latest_k.get("close"),
        0.0,
    )
    prev_close = safe_float(prev_k.get("close"), close_val)
    change_val = round(close_val - prev_close, 2) if prev_close > 0 else 0.0
    change_pct = round((change_val / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

    vol_val = safe_int(
        (snapshot or {}).get("volume") or latest_k.get("volume"),
        0,
    )
    turnover_val = safe_float(
        (snapshot or {}).get("turnover") or latest_k.get("turnover"),
        close_val * vol_val,
    )
    if turnover_val <= 0 and close_val > 0 and vol_val > 0:
        turnover_val = close_val * vol_val

    stock_name = (
        str((snapshot or {}).get("stock_name") or "").strip()
        or get_stock_name(stock_code, allow_network_lookup=False)
    )

    return {
        "rank": rank,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "close": round(close_val, 2),
        "change": change_val,
        "change_pct": change_pct,
        "volume": vol_val,
        "turnover": round(turnover_val, 0),
        "turnover_yi": round(turnover_val / 1e8, 2),
        "date": str(latest_k.get("date") or (snapshot or {}).get("date") or ""),
        "board_id": board_cfg["id"],
        "board_name": board_cfg["name"],
        "board_label": f"{board_cfg['name']} #{rank}",
    }


def get_board_candidates_metadata(
    board_id: str = "core_top20",
    max_count: Optional[int] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    依據指定榜單規則篩選出入選股票中繼資料列表（不強迫拉取全部完整歷史 K 線，適合快速預覽或選股初篩）：
    1. core_top20: 大盤核心榜（成交金額 Top 20）
    2. under100_top20: 百元精選榜（股價 < 100 元 且 成交金額 Top 20）
    3. popular_etf: 人氣 ETF 榜（固定追蹤 0050、0056、00878、00929 等市值與高股息龍頭）
    """
    board_cfg = get_board_config(board_id)
    target_id = board_cfg["id"]
    limit = safe_int(max_count, default=board_cfg.get("top_n", 20), min_val=1, max_val=20)

    # 模式 3：人氣 ETF 榜（固定追蹤龍頭 ETF 清單）
    if target_id == "popular_etf":
        fixed_codes = list(board_cfg.get("fixed_codes") or POPULAR_ETF_CODES)[:limit]
        snapshots = fetch_market_turnover_snapshots(force_refresh=force_refresh)
        snap_map = {s["stock_code"]: s for s in snapshots if s.get("stock_code")}

        results: List[Dict[str, Any]] = []
        for idx, code in enumerate(fixed_codes, start=1):
            snap = snap_map.get(code)
            klines = fetch_stock_klines(code, days=10, force_refresh=force_refresh) if not snap else None
            results.append(_build_candidate_entry(idx, code, board_cfg, snapshot=snap, klines=klines))
        return results

    # 模式 1 與 2：需要全市場成交金額排序快照
    snapshots = fetch_market_turnover_snapshots(force_refresh=force_refresh)

    if target_id == "under100_top20":
        # 2. 百元精選榜：股價 < 100 元 且 成交金額 Top 20（排除 ETF，聚焦百元內個股）
        filtered = [
            s for s in snapshots
            if 0 < safe_float(s.get("close"), 0.0) < 100.0
            and s.get("stock_code") not in POPULAR_ETF_CODES
        ]
        # 若快照中不足 limit 檔，由 UNDER_100_FALLBACK_POOL 補齊並確保股價 < 100 元
        existing_codes = {s["stock_code"] for s in filtered}
        for code in UNDER_100_FALLBACK_POOL:
            if len(filtered) >= limit:
                break
            if code in existing_codes:
                continue
            klines = fetch_stock_klines(code, days=10)
            if klines and 0 < safe_float(klines[-1].get("close"), 0.0) < 100.0:
                latest = klines[-1]
                c_val = safe_float(latest.get("close"), 0.0)
                v_val = safe_int(latest.get("volume"), 0)
                t_val = safe_float(latest.get("turnover"), c_val * v_val)
                filtered.append({
                    "stock_code": code,
                    "stock_name": get_stock_name(code, allow_network_lookup=False),
                    "close": c_val,
                    "volume": v_val,
                    "turnover": t_val,
                    "date": latest.get("date", ""),
                })
                existing_codes.add(code)

        filtered.sort(key=lambda x: safe_float(x.get("turnover"), 0.0), reverse=True)
        top_items = filtered[:limit]
        return [
            _build_candidate_entry(idx, item["stock_code"], board_cfg, snapshot=item)
            for idx, item in enumerate(top_items, start=1)
        ]

    # 預設模式 1：core_top20 大盤核心榜（成交金額 Top 20）
    filtered_core = [
        s for s in snapshots
        if safe_float(s.get("close"), 0.0) > 0
        and s.get("stock_code") not in POPULAR_ETF_CODES
    ]
    existing_core = {s["stock_code"] for s in filtered_core}
    for code in CORE_MARKET_FALLBACK_POOL:
        if len(filtered_core) >= limit:
            break
        if code in existing_core:
            continue
        klines = fetch_stock_klines(code, days=10)
        if klines:
            latest = klines[-1]
            c_val = safe_float(latest.get("close"), 0.0)
            v_val = safe_int(latest.get("volume"), 0)
            t_val = safe_float(latest.get("turnover"), c_val * v_val)
            filtered_core.append({
                "stock_code": code,
                "stock_name": get_stock_name(code, allow_network_lookup=False),
                "close": c_val,
                "volume": v_val,
                "turnover": t_val,
                "date": latest.get("date", ""),
            })
            existing_core.add(code)

    filtered_core.sort(key=lambda x: safe_float(x.get("turnover"), 0.0), reverse=True)
    top_core = filtered_core[:limit]
    return [
        _build_candidate_entry(idx, item["stock_code"], board_cfg, snapshot=item)
        for idx, item in enumerate(top_core, start=1)
    ]


def select_stocks_for_analysis(
    board_id: str = "core_top20",
    max_count: Optional[int] = None,
    history_days: Optional[int] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    執行完整三大榜單自動選股流程：
    1. 依指定榜單 (core_top20 / under100_top20 / popular_etf) 自動篩選出候選標的
    2. 擷取大盤 (TAIEX) 與所有入選個股的歷史日 K 線數據
    3. 呼叫 compute_all_indicators 預先計算所有技術指標 (MA5/20/60, VOL_MA5/20, RSI14, MACD, DMI, ATR14)
    4. 回傳包含榜單資訊、入選個股明細與 klines_map 的完整結構供第一層與第二層 AI 代理分析
    """
    board_cfg = get_board_config(board_id)
    days = safe_int(history_days, default=config.kline_history_days, min_val=30, max_val=365)

    logger.info(f"[自動選股器] 正在執行「{board_cfg['name']} ({board_cfg['id']})」自動選股...")
    raw_candidates = get_board_candidates_metadata(
        board_id=board_cfg["id"],
        max_count=max_count,
        force_refresh=force_refresh,
    )

    klines_map: Dict[str, List[Dict[str, Any]]] = {}

    # 1. 載入並計算大盤 TAIEX 歷史 K 線與技術指標
    taiex_klines = fetch_taiex_klines(days=days, force_refresh=force_refresh)
    if taiex_klines:
        compute_all_indicators(taiex_klines)
        klines_map["TAIEX"] = taiex_klines

    # 2. 載入並計算每檔入選股票之歷史 K 線與技術指標，同步更新最新收盤價與成交金額
    enriched_candidates: List[Dict[str, Any]] = []
    stock_codes: List[str] = []

    for idx, cand in enumerate(raw_candidates, start=1):
        code = cand["stock_code"]
        klines = fetch_stock_klines(code, days=days, force_refresh=force_refresh)
        if not klines:
            continue
        compute_all_indicators(klines)
        klines_map[code] = klines
        stock_codes.append(code)

        updated_entry = _build_candidate_entry(
            rank=len(enriched_candidates) + 1,
            stock_code=code,
            board_cfg=board_cfg,
            snapshot=cand,
            klines=klines,
        )
        # 附加最新關鍵技術指標快照供前端預覽
        latest_k = klines[-1]
        updated_entry["ma5"] = round(safe_float(latest_k.get("ma5"), 0.0), 2) if latest_k.get("ma5") is not None else None
        updated_entry["ma20"] = round(safe_float(latest_k.get("ma20"), 0.0), 2) if latest_k.get("ma20") is not None else None
        updated_entry["ma60"] = round(safe_float(latest_k.get("ma60"), 0.0), 2) if latest_k.get("ma60") is not None else None
        updated_entry["rsi14"] = round(safe_float(latest_k.get("rsi14"), 0.0), 2) if latest_k.get("rsi14") is not None else None
        enriched_candidates.append(updated_entry)

    logger.info(
        f"[自動選股器] 「{board_cfg['name']}」篩選完成，共入選 {len(stock_codes)} 檔標的: {stock_codes}"
    )

    return {
        "board_id": board_cfg["id"],
        "board_name": board_cfg["name"],
        "board_desc": board_cfg["desc"],
        "stock_codes": stock_codes,
        "candidates": enriched_candidates,
        "klines_map": klines_map,
    }


def list_available_boards() -> List[Dict[str, Any]]:
    """回傳系統支援的三大自動選股榜單描述清單。"""
    return [
        {
            "id": cfg["id"],
            "name": cfg["name"],
            "desc": cfg["desc"],
            "top_n": cfg["top_n"],
            "max_price": cfg["max_price"],
            "fixed_codes": cfg["fixed_codes"],
        }
        for cfg in AUTO_SELECT_BOARDS.values()
    ]

