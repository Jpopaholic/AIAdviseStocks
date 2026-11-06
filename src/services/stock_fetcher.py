# Path: src/services/stock_fetcher.py
import time
import math
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.config import (
    config,
    safe_float,
    safe_int,
    get_stock_name,
    register_stock_name,
    get_taiwan_datetime,
    get_taiwan_date_str,
    CORE_MARKET_FALLBACK_POOL,
    UNDER_100_FALLBACK_POOL,
    POPULAR_ETF_CODES,
)

logger = logging.getLogger("StockFetcher")

# 記憶體快取結構 (無外部資料庫依賴)
_KLINES_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
_MARKET_SNAPSHOT_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
_QUOTE_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# 常見台股固定國定假日 (月-日)，供備援模式模擬休市判斷
TW_FIXED_HOLIDAYS_MMDD = {
    "01-01",  # 元旦
    "02-28",  # 和平紀念日
    "04-04",  # 兒童節
    "04-05",  # 清明節
    "05-01",  # 勞動節
    "10-10",  # 國慶日
}


def clear_stock_cache() -> None:
    """清除所有記憶體中的 K 線、報價與全市場排行快取。"""
    global _KLINES_CACHE, _MARKET_SNAPSHOT_CACHE, _QUOTE_CACHE
    _KLINES_CACHE.clear()
    _MARKET_SNAPSHOT_CACHE.clear()
    _QUOTE_CACHE.clear()
    logger.info("已清除所有記憶體股價與 K 線快取。")


def _parse_number(raw: Any, default: float = 0.0) -> float:
    """清理含有逗號或特殊符號之證交所數字字串並轉為 float。"""
    if raw is None:
        return default
    s = str(raw).strip().replace(",", "").replace("--", "").replace("X", "").replace("+", "")
    if not s:
        return default
    return safe_float(s, default)


def _validate_kline_row(row: Dict[str, Any]) -> bool:
    """校驗單筆 K 線資料完整性，過濾空值或異常價格。"""
    o = safe_float(row.get("open"), -1.0)
    h = safe_float(row.get("high"), -1.0)
    l = safe_float(row.get("low"), -1.0)
    c = safe_float(row.get("close"), -1.0)
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return False
    if h < l or h < max(o, c) * 0.999 or l > min(o, c) * 1.001:
        return False
    if not row.get("date"):
        return False
    return True


def _fetch_yahoo_chart_klines(symbol: str, stock_code: str, range_str: str = "1y") -> List[Dict[str, Any]]:
    """
    透過 Yahoo Finance v8 Chart API 快速擷取近一年日 K 線數據 (單次請求即可獲得約 245 根日 K)。
    同時讀取 meta.regularMarketTime 補齊跨日 (00:00 UTC+8) 結算空窗期的最新交易日 K 線。
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_str}&interval=1d"
    resp = requests.get(url, headers=HEADERS, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    result_list = data.get("chart", {}).get("result")
    if not result_list:
        return []

    res = result_list[0]
    meta = res.get("meta") or {}
    timestamps = res.get("timestamp") or []
    quote_list = res.get("indicators", {}).get("quote") or []
    if not timestamps or not quote_list:
        return []

    q = quote_list[0]
    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    volumes = q.get("volume") or []

    tw_tz = timezone(timedelta(hours=8))
    klines_by_date: Dict[str, Dict[str, Any]] = {}

    for i, ts in enumerate(timestamps):
        if i >= len(closes) or closes[i] is None:
            continue
        c_val = safe_float(closes[i], 0.0)
        o_val = safe_float(opens[i] if i < len(opens) and opens[i] is not None else c_val, c_val)
        h_val = safe_float(highs[i] if i < len(highs) and highs[i] is not None else max(o_val, c_val), max(o_val, c_val))
        l_val = safe_float(lows[i] if i < len(lows) and lows[i] is not None else min(o_val, c_val), min(o_val, c_val))
        vol_raw = safe_int(volumes[i] if i < len(volumes) and volumes[i] is not None else 0, 0)

        dt_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tw_tz).strftime("%Y-%m-%d")

        if stock_code == "TAIEX":
            vol_val = int(vol_raw * 1_000_000) if (0 < vol_raw < 10_000_000) else int(vol_raw)
            turnover_val = float(vol_val)
        else:
            vol_val = int(vol_raw)
            turnover_val = float(c_val * vol_val)

        item = {
            "stockCode": stock_code,
            "date": dt_str,
            "open": round(o_val, 2),
            "high": round(h_val, 2),
            "low": round(l_val, 2),
            "close": round(c_val, 2),
            "volume": vol_val,
            "turnover": round(turnover_val, 0),
        }
        if _validate_kline_row(item):
            klines_by_date[dt_str] = item

    # 補齊 Yahoo Finance 在台灣時間午夜 00:00 跨日後，timestamp 尚未寫入前一日收盤但 meta.regularMarketTime 已有的最新交易日
    reg_time = safe_int(meta.get("regularMarketTime"), 0)
    reg_price = safe_float(meta.get("regularMarketPrice"), 0.0)
    if reg_time > 0 and reg_price > 0:
        reg_dt_str = datetime.fromtimestamp(reg_time, tz=timezone.utc).astimezone(tw_tz).strftime("%Y-%m-%d")
        if reg_dt_str not in klines_by_date:
            prev_c = safe_float(meta.get("chartPreviousClose") or meta.get("previousClose"), reg_price)
            o_m = safe_float(meta.get("regularMarketOpen"), prev_c if prev_c > 0 else reg_price)
            h_m = max(safe_float(meta.get("regularMarketDayHigh"), reg_price), o_m, reg_price)
            l_m = min(safe_float(meta.get("regularMarketDayLow") or reg_price, reg_price), o_m, reg_price)
            v_m = safe_int(meta.get("regularMarketVolume"), 0)
            meta_item = {
                "stockCode": stock_code,
                "date": reg_dt_str,
                "open": round(o_m, 2),
                "high": round(h_m, 2),
                "low": round(l_m, 2),
                "close": round(reg_price, 2),
                "volume": v_m,
                "turnover": round(float(reg_price * v_m), 0),
            }
            if _validate_kline_row(meta_item):
                klines_by_date[reg_dt_str] = meta_item

    sorted_klines = sorted(klines_by_date.values(), key=lambda x: x["date"])
    return sorted_klines


def _fetch_twse_stock_month(stock_code: str, date_yyyymmdd: str) -> List[Dict[str, Any]]:
    """從台灣證券交易所 (TWSE) STOCK_DAY API 擷取單月歷史日 K 線。"""
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={date_yyyymmdd}&stockNo={stock_code}"
    resp = requests.get(url, headers=HEADERS, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    if data.get("stat") != "OK" or "data" not in data:
        return []

    klines: List[Dict[str, Any]] = []
    for row in data["data"]:
        try:
            parts = str(row[0]).strip().split("/")
            ad_year = int(parts[0]) + 1911
            iso_date = f"{ad_year}-{parts[1]}-{parts[2]}"
            vol_shares = int(_parse_number(row[1], 0.0))
            turnover = _parse_number(row[2], 0.0)
            o_val = _parse_number(row[3], 0.0)
            h_val = _parse_number(row[4], 0.0)
            l_val = _parse_number(row[5], 0.0)
            c_val = _parse_number(row[6], 0.0)

            item = {
                "stockCode": stock_code,
                "date": iso_date,
                "open": o_val,
                "high": h_val,
                "low": l_val,
                "close": c_val,
                "volume": vol_shares,
                "turnover": turnover if turnover > 0 else round(c_val * vol_shares, 0),
            }
            if _validate_kline_row(item):
                klines.append(item)
        except Exception:
            continue
    return klines


def _fetch_twse_taiex_month(date_yyyymmdd: str) -> List[Dict[str, Any]]:
    """從證交所 MI_5MINS_HIST 與 FMTQIK 擷取大盤單月 K 線與成交金額。"""
    url_kline = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={date_yyyymmdd}"
    url_vol = f"https://www.twse.com.tw/exchangeReport/FMTQIK?response=json&date={date_yyyymmdd}"

    vol_map: Dict[str, int] = {}
    try:
        resp_v = requests.get(url_vol, headers=HEADERS, timeout=8)
        if resp_v.status_code == 200:
            data_v = resp_v.json()
            for row in data_v.get("data", []):
                parts = str(row[0]).strip().split("/")
                iso_date = f"{int(parts[0]) + 1911}-{parts[1]}-{parts[2]}"
                vol_map[iso_date] = int(_parse_number(row[2], 0.0))
    except Exception as e:
        logger.debug(f"FMTQIK 擷取異常: {e}")

    resp_k = requests.get(url_kline, headers=HEADERS, timeout=8)
    resp_k.raise_for_status()
    data_k = resp_k.json()
    if "data" not in data_k:
        return []

    klines: List[Dict[str, Any]] = []
    for row in data_k["data"]:
        try:
            parts = str(row[0]).strip().split("/")
            iso_date = f"{int(parts[0]) + 1911}-{parts[1]}-{parts[2]}"
            o_val = _parse_number(row[1], 0.0)
            h_val = _parse_number(row[2], 0.0)
            l_val = _parse_number(row[3], 0.0)
            c_val = _parse_number(row[4], 0.0)
            turnover = vol_map.get(iso_date, 0)
            item = {
                "stockCode": "TAIEX",
                "date": iso_date,
                "open": o_val,
                "high": h_val,
                "low": l_val,
                "close": c_val,
                "volume": turnover,
                "turnover": float(turnover),
            }
            if _validate_kline_row(item):
                klines.append(item)
        except Exception:
            continue
    return klines


def _generate_fallback_klines(stock_code: str, days: int = 180) -> List[Dict[str, Any]]:
    """
    當外部網路處於離線或封鎖環境時的決定性備援 K 線生成器，
    自動排除週六、週日與台灣固定國定假日，確保非開盤日檢測與歷史回溯正常運作。
    """
    seed_val = sum(ord(ch) * (idx + 1) for idx, ch in enumerate(stock_code))
    if stock_code == "TAIEX":
        base_price = 22500.0
    elif stock_code in ("2330", "2454", "3008", "6669", "3017", "3324"):
        base_price = 950.0 + (seed_val % 400)
    elif stock_code in POPULAR_ETF_CODES:
        base_price = 22.0 + (seed_val % 35)
    elif stock_code in UNDER_100_FALLBACK_POOL:
        base_price = 28.0 + (seed_val % 60)
    else:
        base_price = 115.0 + (seed_val % 180)

    now_dt = get_taiwan_datetime()
    target_total = max(days, 180)
    klines: List[Dict[str, Any]] = []
    price = base_price * 0.90

    curr_dt = now_dt - timedelta(days=int(target_total * 1.6))
    idx = 0
    while curr_dt <= now_dt:
        mm_dd = curr_dt.strftime("%m-%d")
        if curr_dt.weekday() < 5 and mm_dd not in TW_FIXED_HOLIDAYS_MMDD:
            wave = math.sin((idx + seed_val) * 0.23) * 0.015 + math.cos((idx + seed_val) * 0.11) * 0.008
            drift = 0.0012 if (seed_val % 3 != 0) else -0.0004
            open_p = round(price * (1.0 + wave * 0.3), 2)
            close_p = round(max(5.0, price * (1.0 + wave + drift)), 2)
            high_p = round(max(open_p, close_p) * (1.0 + abs(wave) * 0.6 + 0.004), 2)
            low_p = round(min(open_p, close_p) * (1.0 - abs(wave) * 0.5 - 0.004), 2)

            if stock_code in UNDER_100_FALLBACK_POOL and close_p >= 98.0:
                scale = 85.0 / close_p
                open_p = round(open_p * scale, 2)
                close_p = round(close_p * scale, 2)
                high_p = round(high_p * scale, 2)
                low_p = round(low_p * scale, 2)

            if stock_code == "TAIEX":
                vol = int((3600 + (idx * 73 + seed_val) % 1400) * 1e8)
                turnover = float(vol)
            else:
                vol = int(12_000_000 + ((idx * 131 + seed_val * 997) % 45_000_000))
                turnover = round(close_p * vol, 0)

            klines.append({
                "stockCode": stock_code,
                "date": curr_dt.strftime("%Y-%m-%d"),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol,
                "turnover": turnover,
            })
            price = close_p
            idx += 1
        curr_dt += timedelta(days=1)

    return klines[-target_total:]


def _slice_klines_by_date(
    klines: List[Dict[str, Any]],
    days: int,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """依指定基準日 (end_date) 截斷 K 線序列，並取最近 days 根日 K 線。"""
    if end_date:
        clean_end = str(end_date).strip()[:10]
        filtered = [dict(k) for k in klines if str(k.get("date", "")) <= clean_end]
        if filtered:
            return filtered[-days:]
    return [dict(k) for k in klines[-days:]]


def fetch_taiex_klines(
    days: int = 90,
    force_refresh: bool = False,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    擷取台股大盤加權指數 (TAIEX) 的歷史日 K 線數據 (含記憶體 TTL 快取與指定基準日截斷)。
    """
    cache_key = "TAIEX"
    now = time.time()
    if not force_refresh and cache_key in _KLINES_CACHE:
        cached_data, ts = _KLINES_CACHE[cache_key]
        if now - ts < config.cache_ttl_seconds and len(cached_data) >= 30:
            return _slice_klines_by_date(cached_data, days=days, end_date=end_date)

    klines: List[Dict[str, Any]] = []

    # 1. 優先使用 Yahoo Finance ^TWII 快速取得近 1 年日 K 線
    try:
        klines = _fetch_yahoo_chart_klines("^TWII", "TAIEX", range_str="1y")
    except Exception as e:
        logger.debug(f"Yahoo Finance 擷取 TAIEX 失敗，將嘗試證交所 API: {e}")

    # 2. 補充合併證交所當月 MI_5MINS_HIST 最新交易日（防止 Yahoo 在午夜 00:00 跨日結算空窗期漏掉剛收盤之交易日）
    try:
        tw_now = get_taiwan_datetime()
        by_date: Dict[str, Dict[str, Any]] = {item["date"]: item for item in klines if item.get("date")}
        months_to_fetch = 1 if by_date else 4
        for m_offset in range(months_to_fetch):
            dt = tw_now - timedelta(days=28 * m_offset)
            m_klines = _fetch_twse_taiex_month(dt.strftime("%Y%m01"))
            for item in m_klines:
                d_key = item.get("date")
                if d_key and d_key not in by_date:
                    by_date[d_key] = item
        if by_date:
            klines = sorted(by_date.values(), key=lambda x: x["date"])
    except Exception as e:
        logger.debug(f"證交所補充擷取 TAIEX 歷史 K 線略過: {e}")

    # 3. 離線或封鎖環境備援
    if not klines:
        logger.warning("外部大盤 API 暫時無法連線，使用內建備援大盤 K 線序列。")
        klines = _generate_fallback_klines("TAIEX", days=max(days, 180))

    _KLINES_CACHE[cache_key] = (klines, now)
    return _slice_klines_by_date(klines, days=days, end_date=end_date)


def resolve_nearest_trading_date(
    target_date: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    檢查使用者選擇的日期 (YYYY-MM-DD) 是否有台股開盤資料：
    - 若該日有開盤，直接回傳 effective_date = requested_date, date_adjusted = False
    - 若該日無開盤（週末、國定假日、休市或未來日期），自動尋找距離該日期最近的開盤日
      （距離天數相同時優先選擇該日之前的最近交易日），並回傳警告訊息 date_warning。
    """
    raw_req = str(target_date or "").strip()[:10]
    requested_date = raw_req if raw_req else get_taiwan_date_str()

    # 取得大盤完整歷史交易日清單
    taiex_all = fetch_taiex_klines(days=250, force_refresh=force_refresh, end_date=None)
    available_dates = sorted({str(k.get("date", "")).strip() for k in taiex_all if k.get("date")})

    # 若快取中剛好缺少所選日期（例如跨午夜 00:00 後首次查詢），強制重刷一次確認最新交易日
    if requested_date not in available_dates and not force_refresh:
        taiex_all = fetch_taiex_klines(days=250, force_refresh=True, end_date=None)
        available_dates = sorted({str(k.get("date", "")).strip() for k in taiex_all if k.get("date")})

    if not available_dates:
        return {
            "requested_date": requested_date,
            "effective_date": requested_date,
            "is_trading_day": True,
            "date_adjusted": False,
            "date_warning": None,
            "available_min_date": requested_date,
            "available_max_date": requested_date,
        }

    # 若所選日期正好在開盤交易日清單中
    if requested_date in available_dates:
        return {
            "requested_date": requested_date,
            "effective_date": requested_date,
            "is_trading_day": True,
            "date_adjusted": False,
            "date_warning": None,
            "available_min_date": available_dates[0],
            "available_max_date": available_dates[-1],
        }

    # 若所選日期沒有開盤，計算距離每個有效開盤日的天數差距，取最近開盤日
    try:
        req_dt = datetime.strptime(requested_date, "%Y-%m-%d").date()
    except ValueError:
        effective = available_dates[-1]
        return {
            "requested_date": requested_date,
            "effective_date": effective,
            "is_trading_day": False,
            "date_adjusted": True,
            "date_warning": f"⚠️ 輸入的日期格式 ({requested_date}) 無效，系統已自動切換為最新開盤交易日：{effective}。",
            "available_min_date": available_dates[0],
            "available_max_date": available_dates[-1],
        }

    def _distance_key(cand_str: str) -> Tuple[int, int]:
        cand_dt = datetime.strptime(cand_str, "%Y-%m-%d").date()
        diff_days = abs((cand_dt - req_dt).days)
        # 距離相同時，優先選擇 <= requested_date 的前一個開盤日 (0 優先於 1)
        is_after = 0 if cand_dt <= req_dt else 1
        return (diff_days, is_after)

    nearest_date = min(available_dates, key=_distance_key)

    weekday_names = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
    req_wk = weekday_names[req_dt.weekday()]
    near_dt = datetime.strptime(nearest_date, "%Y-%m-%d").date()
    near_wk = weekday_names[near_dt.weekday()]

    if req_dt > datetime.strptime(available_dates[-1], "%Y-%m-%d").date():
        reason_hint = "尚未收盤或為未來/休市日期"
    elif req_dt.weekday() >= 5:
        reason_hint = f"適逢週末休市 ({req_wk})"
    else:
        reason_hint = f"適逢台股休市日 ({req_wk})"

    warning_msg = (
        f"⚠️ 您選擇的日期 {requested_date}（{reason_hint}）當日沒有開盤資料，"
        f"系統已自動為您切換至距離最近的開盤交易日：{nearest_date}（{near_wk}）。"
    )

    return {
        "requested_date": requested_date,
        "effective_date": nearest_date,
        "is_trading_day": False,
        "date_adjusted": True,
        "date_warning": warning_msg,
        "available_min_date": available_dates[0],
        "available_max_date": available_dates[-1],
    }


def fetch_stock_klines(
    stock_code: str,
    days: int = 90,
    force_refresh: bool = False,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    擷取指定台股個股或 ETF 的歷史日 K 線數據 (含記憶體 TTL 快取與指定基準日截斷)。
    """
    clean_code = str(stock_code or "").strip().upper()
    if not clean_code:
        return []
    if clean_code == "TAIEX":
        return fetch_taiex_klines(days=days, force_refresh=force_refresh, end_date=end_date)

    now = time.time()
    if not force_refresh and clean_code in _KLINES_CACHE:
        cached_data, ts = _KLINES_CACHE[clean_code]
        if now - ts < config.cache_ttl_seconds and len(cached_data) >= 20:
            return _slice_klines_by_date(cached_data, days=days, end_date=end_date)

    klines: List[Dict[str, Any]] = []

    # 1. 嘗試 Yahoo Finance (.TW 上市 / .TWO 上櫃)
    for suffix in (".TW", ".TWO"):
        try:
            klines = _fetch_yahoo_chart_klines(f"{clean_code}{suffix}", clean_code, range_str="1y")
            if len(klines) >= 15:
                break
        except Exception:
            continue

    # 2. 若失敗則回退使用證交所 STOCK_DAY API
    if not klines:
        try:
            tw_now = get_taiwan_datetime()
            merged: Dict[str, Dict[str, Any]] = {}
            for m_offset in range(4):
                dt = tw_now - timedelta(days=28 * m_offset)
                m_klines = _fetch_twse_stock_month(clean_code, dt.strftime("%Y%m01"))
                for item in m_klines:
                    merged[item["date"]] = item
            klines = sorted(merged.values(), key=lambda x: x["date"])
        except Exception as e:
            logger.warning(f"證交所擷取 {clean_code} K 線失敗: {e}")

    # 3. 離線備援
    if not klines:
        logger.warning(f"無法自外部網路取得 {clean_code} K 線，啟用內建備援 K 線。")
        klines = _generate_fallback_klines(clean_code, days=max(days, 180))

    _KLINES_CACHE[clean_code] = (klines, now)
    return _slice_klines_by_date(klines, days=days, end_date=end_date)


def fetch_market_turnover_snapshots(
    force_refresh: bool = False,
    target_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    擷取台股市場指定交易日之成交金額與收盤價排行快照，供三大自動選股榜單篩選：
    - 大盤核心榜（成交金額 Top 20）
    - 百元精選榜（股價 < 100 元 且 成交金額 Top 20）
    支援傳入 target_date (YYYY-MM-DD) 取得特定歷史交易日或最新交易日行情。
    """
    global _MARKET_SNAPSHOT_CACHE
    clean_date = str(target_date or "").strip()[:10]
    cache_key = clean_date or "LATEST"
    now = time.time()

    if not force_refresh and cache_key in _MARKET_SNAPSHOT_CACHE:
        cached_list, ts = _MARKET_SNAPSHOT_CACHE[cache_key]
        if cached_list and (now - ts < config.cache_ttl_seconds):
            return [dict(item) for item in cached_list]

    # 判斷是否為最新一個交易日
    taiex_recent = fetch_taiex_klines(days=5, force_refresh=False)
    latest_market_date = taiex_recent[-1]["date"] if taiex_recent else get_taiwan_date_str()
    is_latest_date = (not clean_date) or (clean_date >= latest_market_date)
    effective_snap_date = clean_date if clean_date else latest_market_date

    snapshots_by_code: Dict[str, Dict[str, Any]] = {}

    # 1. 若查詢最新交易日，優先使用 TWSE OpenAPI (STOCK_DAY_ALL) 單次請求高速取得
    if is_latest_date:
        try:
            openapi_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
            resp_open = requests.get(openapi_url, headers=HEADERS, timeout=5)
            if resp_open.status_code == 200:
                rows = resp_open.json()
                if isinstance(rows, list):
                    for r in rows:
                        try:
                            code = str(r.get("Code") or "").strip()
                            if not (len(code) == 4 and code.isdigit()) and code not in POPULAR_ETF_CODES:
                                continue
                            name = str(r.get("Name") or "").strip()
                            close_val = _parse_number(r.get("ClosingPrice"), 0.0)
                            vol_val = int(_parse_number(r.get("TradeVolume"), 0.0))
                            turnover_val = _parse_number(r.get("TradeValue"), 0.0)
                            change_val = _parse_number(r.get("Change"), 0.0)
                            if close_val <= 0 or turnover_val <= 0:
                                continue
                            prev_close = close_val - change_val
                            change_pct = round((change_val / prev_close) * 100.0, 2) if prev_close > 0 else 0.0
                            if name:
                                register_stock_name(code, name)
                            snapshots_by_code[code] = {
                                "stock_code": code,
                                "stock_name": name or get_stock_name(code, allow_network_lookup=False),
                                "close": close_val,
                                "change": round(change_val, 2),
                                "change_pct": change_pct,
                                "volume": vol_val,
                                "turnover": turnover_val,
                                "date": effective_snap_date,
                            }
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"TWSE OpenAPI STOCK_DAY_ALL 擷取失敗: {e}")

    # 2. 若為歷史指定日期（或 OpenAPI 無資料），嘗試使用支援 date=YYYYMMDD 的 TWSE MI_INDEX API
    if not snapshots_by_code:
        try:
            date_compact = effective_snap_date.replace("-", "")
            url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date_compact}&type=ALLBUT0999&response=json"
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                tables = data.get("tables") or []
                for tbl in tables:
                    fields = tbl.get("fields") or []
                    rows = tbl.get("data") or []
                    if "證券代號" in fields and "成交金額" in fields and "收盤價" in fields:
                        idx_code = fields.index("證券代號")
                        idx_name = fields.index("證券名稱") if "證券名稱" in fields else 1
                        idx_vol = fields.index("成交股數") if "成交股數" in fields else 2
                        idx_turn = fields.index("成交金額")
                        idx_close = fields.index("收盤價")
                        idx_diff = fields.index("漲跌價差") if "漲跌價差" in fields else -1
                        idx_sign = fields.index("漲跌(+/-)") if "漲跌(+/-)" in fields else -1
                        for r in rows:
                            try:
                                code = str(r[idx_code]).strip()
                                if not (len(code) == 4 and code.isdigit()) and code not in POPULAR_ETF_CODES:
                                    continue
                                name = str(r[idx_name]).strip()
                                close_val = _parse_number(r[idx_close], 0.0)
                                vol_val = int(_parse_number(r[idx_vol], 0.0))
                                turnover_val = _parse_number(r[idx_turn], 0.0)
                                if close_val <= 0 or turnover_val <= 0:
                                    continue
                                chg_val = 0.0
                                if idx_diff >= 0:
                                    chg_val = _parse_number(r[idx_diff], 0.0)
                                    if idx_sign >= 0 and "-" in str(r[idx_sign]):
                                        chg_val = -abs(chg_val)
                                prev_c = close_val - chg_val
                                chg_pct = round((chg_val / prev_c) * 100.0, 2) if prev_c > 0 else 0.0
                                register_stock_name(code, name)
                                snapshots_by_code[code] = {
                                    "stock_code": code,
                                    "stock_name": name,
                                    "close": close_val,
                                    "change": round(chg_val, 2),
                                    "change_pct": chg_pct,
                                    "volume": vol_val,
                                    "turnover": turnover_val,
                                    "date": effective_snap_date,
                                }
                            except Exception:
                                continue
        except Exception as e:
            logger.debug(f"TWSE MI_INDEX ({effective_snap_date}) 全市場快照擷取失敗: {e}")

    # 3. 若外部證交所全市場快照受限，依 effective_snap_date 截斷候選池 K 線產生當日快照
    if not snapshots_by_code:
        candidate_codes = list(
            dict.fromkeys(CORE_MARKET_FALLBACK_POOL + UNDER_100_FALLBACK_POOL + POPULAR_ETF_CODES)
        )
        for code in candidate_codes:
            klines = _slice_klines_by_date(
                _generate_fallback_klines(code, days=180),
                days=5,
                end_date=effective_snap_date,
            )
            if not klines:
                continue
            latest = klines[-1]
            prev = klines[-2] if len(klines) >= 2 else latest
            close_val = safe_float(latest.get("close"), 0.0)
            prev_val = safe_float(prev.get("close"), close_val)
            chg_val = round(close_val - prev_val, 2)
            chg_pct = round((chg_val / prev_val) * 100.0, 2) if prev_val > 0 else 0.0
            vol_val = safe_int(latest.get("volume"), 0)
            turnover_val = safe_float(latest.get("turnover"), close_val * vol_val)
            name = get_stock_name(code, allow_network_lookup=False)
            snapshots_by_code[code] = {
                "stock_code": code,
                "stock_name": name,
                "close": close_val,
                "change": chg_val,
                "change_pct": chg_pct,
                "volume": vol_val,
                "turnover": turnover_val if turnover_val > 0 else close_val * vol_val,
                "date": latest.get("date", effective_snap_date),
            }

    sorted_snapshots = sorted(
        snapshots_by_code.values(),
        key=lambda x: safe_float(x.get("turnover"), 0.0),
        reverse=True,
    )
    _MARKET_SNAPSHOT_CACHE[cache_key] = (sorted_snapshots, now)
    return [dict(item) for item in sorted_snapshots]


def fetch_realtime_quote(stock_code: str, force_refresh: bool = False) -> Dict[str, Any]:
    """取得單一股票或 TAIEX 的最新報價摘要（優先從最新 K 線或快取回傳）。"""
    clean_code = str(stock_code or "").strip().upper()
    now = time.time()
    if not force_refresh and clean_code in _QUOTE_CACHE:
        cached_q, ts = _QUOTE_CACHE[clean_code]
        if now - ts < 60.0:
            return dict(cached_q)

    klines = fetch_stock_klines(clean_code, days=5, force_refresh=force_refresh)
    if not klines:
        return {}

    latest = klines[-1]
    prev_close = safe_float(klines[-2]["close"], latest["close"]) if len(klines) >= 2 else latest["close"]
    change = round(latest["close"] - prev_close, 2)
    change_pct = round((change / prev_close * 100.0), 2) if prev_close > 0 else 0.0

    quote = {
        "stockCode": clean_code,
        "stockName": get_stock_name(clean_code, allow_network_lookup=False),
        "date": latest["date"],
        "price": latest["close"],
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "volume": latest["volume"],
        "turnover": latest.get("turnover", round(latest["close"] * latest["volume"], 0)),
        "change": change,
        "changePct": change_pct,
    }
    _QUOTE_CACHE[clean_code] = (quote, now)
    return dict(quote)
