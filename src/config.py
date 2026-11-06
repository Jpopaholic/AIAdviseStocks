# Path: src/config.py
import os
import sys
import json
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

# 設定基礎中文日誌格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Config")

# 解除 Python 3.11+ 預設 4,300 位數整數字串轉換上限，配合 safe_int 自動安全收斂
if hasattr(sys, "set_int_max_str_digits"):
    try:
        sys.set_int_max_str_digits(0)
    except Exception:
        pass


def get_taiwan_datetime() -> datetime:
    """取得目前台灣時區 (UTC+8) 的 datetime 物件。"""
    tw_tz = timezone(timedelta(hours=8))
    return datetime.now(tw_tz)


def get_taiwan_date_str() -> str:
    """取得目前台灣時區 (UTC+8) 的 YYYY-MM-DD 日期字串。"""
    return get_taiwan_datetime().strftime("%Y-%m-%d")


def safe_int(
    val: Any,
    default: int = 0,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> int:
    """
    安全轉換任何數值或字串為整數，自動防護 OverflowError、NaN 與 Inf，並支援上下界收斂。
    """
    if val is None or isinstance(val, bool):
        res = default
    else:
        try:
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    res = default
                else:
                    res = int(val)
            else:
                res = int(float(str(val).strip()))
        except (ValueError, TypeError, OverflowError):
            res = default

    if min_val is not None and res < min_val:
        res = min_val
    if max_val is not None and res > max_val:
        res = max_val
    return res


def safe_float(
    val: Any,
    default: float = 0.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    安全轉換任何數值或字串為浮點數，自動防護 OverflowError、NaN 與 Inf，並支援上下界收斂。
    """
    if val is None or isinstance(val, bool):
        res = default
    else:
        try:
            res = float(str(val).strip())
            if math.isnan(res) or math.isinf(res):
                res = default
        except (ValueError, TypeError, OverflowError):
            res = default

    if min_val is not None and res < min_val:
        res = min_val
    if max_val is not None and res > max_val:
        res = max_val
    return res


# =====================================================================
# 三大自動選股榜單配置 (Three Auto-Selection Leaderboards)
# =====================================================================
# 3. 人氣 ETF 榜固定追蹤清單（市值與高股息龍頭 ETF）
POPULAR_ETF_CODES: List[str] = [
    "0050",    # 元大台灣50
    "0056",    # 元大高股息
    "00878",   # 國泰永續高股息
    "00929",   # 復華台灣科技優息
    "00919",   # 群益台灣精選高息
    "006208",  # 富邦台50
    "00713",   # 元大台灣高息低波
    "00940"    # 元大台灣價值高息
]

# 大盤核心榜預設指標備援池（當外部全市場排行 API 休市或受限時的代表性高成交金額權值/熱門股池）
CORE_MARKET_FALLBACK_POOL: List[str] = [
    "2330", "2317", "2454", "2382", "3231", "2308", "3017", "3324", "2603", "6669",
    "2345", "3711", "2881", "2882", "2357", "2303", "3034", "2379", "2449", "2356",
    "2609", "2615", "2891", "2886", "3008", "2301", "1519", "1513", "2618", "3481"
]

# 百元精選榜預設指標備援池（股價常態低於 100 元且具高成交金額之熱門標的池）
UNDER_100_FALLBACK_POOL: List[str] = [
    "2303", "2618", "3481", "2324", "2883", "2353", "2891", "2884", "2886", "2002",
    "2610", "2409", "2344", "2337", "1301", "1303", "2887", "2890", "2880", "2885",
    "1605", "2609", "2615", "2356", "2371", "2542", "2834", "5880", "2801", "1101"
]

AUTO_SELECT_BOARDS: Dict[str, Dict[str, Any]] = {
    "core_top20": {
        "id": "core_top20",
        "name": "大盤核心榜",
        "desc": "成交金額 Top 20 指標股",
        "description": "成交金額 Top 20 指標股",
        "top_n": 20,
        "max_price": None,
        "fixed_codes": None
    },
    "under100_top20": {
        "id": "under100_top20",
        "name": "百元精選榜",
        "desc": "股價 < 100 元 且 成交金額 Top 20",
        "description": "股價 < 100 元 且 成交金額 Top 20",
        "top_n": 20,
        "max_price": 100.0,
        "fixed_codes": None
    },
    "popular_etf": {
        "id": "popular_etf",
        "name": "人氣 ETF 榜",
        "desc": "固定追蹤 0050、0056、00878、00929 等市值與高股息龍頭",
        "description": "固定追蹤 0050、0056、00878、00929 等市值與高股息龍頭",
        "top_n": len(POPULAR_ETF_CODES),
        "max_price": None,
        "fixed_codes": list(POPULAR_ETF_CODES)
    }
}

# =====================================================================
# 台股內建代表性標的池 (相容舊有預設分類與三大榜單)
# =====================================================================
STOCK_PRESETS: Dict[str, List[str]] = {
    "core_top20": list(CORE_MARKET_FALLBACK_POOL[:20]),
    "under100_top20": list(UNDER_100_FALLBACK_POOL[:20]),
    "popular_etf": list(POPULAR_ETF_CODES),
    "top5": ["2330", "2317", "2454", "2308", "2881"],
    "semiconductor": ["2330", "2454", "2303", "3711", "2379"],
    "finance": ["2881", "2882", "2891", "2886", "2884"],
    "apple": ["2317", "2330", "2382", "3231", "2357"],
    "highdividend": ["3034", "2301", "2356", "2449", "3045"],
    "penny": ["2303", "2618", "2324", "2883", "3481", "2353"],
    "small": ["2303", "2618", "2324", "2883", "3481", "2353"],
    "etf": list(POPULAR_ETF_CODES),
    "all": [
        "2330", "2317", "2454", "2308", "2881", "2882", "2891", "2886", "2884",
        "2303", "3711", "2379", "2382", "3231", "2357", "3034", "2301", "2356",
        "2449", "3045", "2618", "2324", "2883", "3481", "2353"
    ]
}

STOCK_PRESETS_INFO: Dict[str, Dict[str, str]] = {
    "core_top20": {"name": "大盤核心榜", "desc": "全市場成交金額 Top 20 指標股"},
    "under100_top20": {"name": "百元精選榜", "desc": "股價 < 100 元 且 成交金額 Top 20"},
    "popular_etf": {"name": "人氣 ETF 榜", "desc": "固定追蹤 0050、0056、00878、00929 等市值與高股息龍頭"},
    "all": {"name": "全部系統代表股", "desc": "系統內建之 25 檔台股各產業代表性標的"},
    "top5": {"name": "市值前五大", "desc": "台積電、鴻海、聯發科、台達電、富邦金"},
    "semiconductor": {"name": "半導體概念", "desc": "台積電、聯發科、聯電、日月光投控、瑞昱"},
    "apple": {"name": "蘋果/AI代工鏈", "desc": "鴻海、台積電、廣達、緯創、華碩"},
    "finance": {"name": "金融特選股", "desc": "富邦金、國泰金、中信金、兆豐金、玉山金"},
    "highdividend": {"name": "高股息概念", "desc": "聯詠、光寶科、英業達、京元電、台灣大"},
    "penny": {"name": "銅板題材股", "desc": "聯電、長榮航、仁寶、凱基金、群創、宏碁"},
    "etf": {"name": "熱門台股ETF", "desc": "元大台灣50、元大高股息、國泰永續高股息等熱門ETF"}
}

# 股票代號至中文名稱映射對照表
STOCK_NAMES: Dict[str, str] = {
    "TAIEX": "發行量加權股價指數",
    "0050": "元大台灣50",
    "0051": "元大中型100",
    "0056": "元大高股息",
    "00713": "元大台灣高息低波",
    "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息",
    "00929": "復華台灣科技優息",
    "00940": "元大台灣價值高息",
    "006208": "富邦台50",
    "1101": "台泥",
    "1301": "台塑",
    "1303": "南亞",
    "1513": "中興電",
    "1519": "華城",
    "1605": "華新",
    "2002": "中鋼",
    "2301": "光寶科",
    "2303": "聯電",
    "2308": "台達電",
    "2317": "鴻海",
    "2324": "仁寶",
    "2330": "台積電",
    "2337": "旺宏",
    "2344": "華邦電",
    "2345": "智邦",
    "2353": "宏碁",
    "2356": "英業達",
    "2357": "華碩",
    "2371": "大同",
    "2379": "瑞昱",
    "2382": "廣達",
    "2409": "友達",
    "2412": "中華電",
    "2449": "京元電",
    "2454": "聯發科",
    "2542": "興富發",
    "2603": "長榮",
    "2609": "陽明",
    "2610": "華航",
    "2615": "萬海",
    "2618": "長榮航",
    "2801": "彰銀",
    "2834": "臺企銀",
    "2880": "華南金",
    "2881": "富邦金",
    "2882": "國泰金",
    "2883": "凱基金",
    "2884": "玉山金",
    "2885": "元大金",
    "2886": "兆豐金",
    "2887": "台新金",
    "2890": "永豐金",
    "2891": "中信金",
    "3008": "大立光",
    "3017": "奇鋐",
    "3034": "聯詠",
    "3045": "台灣大",
    "3231": "緯創",
    "3324": "雙鴻",
    "3481": "群創",
    "3711": "日月光投控",
    "5880": "合庫金",
    "6669": "緯穎"
}

_dynamic_stock_names: Dict[str, str] = {}


def get_board_config(board_id: str) -> Dict[str, Any]:
    """
    根據自動選股榜單識別碼取得榜單設定。若傳入未知識別碼，預設回傳大盤核心榜 ('core_top20')。
    """
    clean_id = str(board_id or "").strip().lower()
    if clean_id in AUTO_SELECT_BOARDS:
        return dict(AUTO_SELECT_BOARDS[clean_id])
    return dict(AUTO_SELECT_BOARDS["core_top20"])


def resolve_stock_codes(stocks_arg: Optional[str] = None) -> List[str]:
    """
    將傳入的股票代號字串或榜單/套餐名稱解析為不重複的股票代號列表。
    若未傳入或為空，預設回傳大盤核心榜預設清單 ('core_top20')。
    """
    if not stocks_arg or not str(stocks_arg).strip():
        return list(STOCK_PRESETS["core_top20"])

    preset_key = str(stocks_arg).strip().lower()
    if preset_key in STOCK_PRESETS:
        return list(STOCK_PRESETS[preset_key])

    codes: List[str] = []
    for s in str(stocks_arg).split(","):
        token = s.strip()
        if not token:
            continue
        if token.lower() in STOCK_PRESETS:
            codes.extend(STOCK_PRESETS[token.lower()])
        else:
            codes.append(token.upper())

    seen = set()
    return [c for c in codes if not (c in seen or seen.add(c))]


def get_all_candidate_stocks(include_etf: bool = False) -> List[str]:
    """
    取得系統內建自動選股候選池的所有股票代號列表。
    """
    combined = list(CORE_MARKET_FALLBACK_POOL) + list(UNDER_100_FALLBACK_POOL)
    if include_etf:
        combined.extend(POPULAR_ETF_CODES)
    seen = set()
    return [c for c in combined if not (c in seen or seen.add(c))]


def register_stock_name(stock_code: str, stock_name: str) -> None:
    """
    將從市場行情 API 取得的股票代號與中文名稱註冊至記憶體快取中。
    """
    clean_code = str(stock_code or "").strip().upper()
    clean_name = str(stock_name or "").strip()
    if clean_code and clean_name and clean_code not in STOCK_NAMES:
        _dynamic_stock_names[clean_code] = clean_name


def get_stock_name(stock_code: str, allow_network_lookup: bool = True) -> str:
    """
    獲取股票代號對應的繁體中文名稱。
    優先查找內建 STOCK_NAMES 與記憶體快取；若無且允許網路查詢，則向證交所 MIS API 查詢並快取。
    """
    clean_code = str(stock_code or "").strip().upper()
    if not clean_code:
        return ""

    if clean_code in STOCK_NAMES:
        return STOCK_NAMES[clean_code]

    if clean_code in _dynamic_stock_names:
        return _dynamic_stock_names[clean_code]

    if not allow_network_lookup:
        return clean_code

    try:
        import requests
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{clean_code}.tw|otc_{clean_code}.tw"
        response = requests.get(
            url,
            timeout=3,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        if response.status_code == 200:
            data = response.json()
            msg_array = data.get("msgArray", [])
            for item in msg_array:
                name = str(item.get("n", "")).strip()
                if name:
                    _dynamic_stock_names[clean_code] = name
                    logger.info(f"動態獲取股票 {clean_code} 中文名稱: {name}")
                    return name
    except Exception as e:
        logger.debug(f"無法動態查詢股票 {clean_code} 名稱: {e}")

    return clean_code


# =====================================================================
# 環境變數與系統配置載入 (無資料庫伺服器，不含硬編碼 API Key)
# =====================================================================
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

_json_config: Dict[str, Any] = {}
_last_config_load_time: float = 0.0


def _reload_config_json_if_needed() -> None:
    """
    若專案根目錄存在 config.json 或環境變數 CONFIG_JSON，則動態載入非敏感系統設定。
    """
    global _json_config, _last_config_load_time
    env_config_json = os.getenv("CONFIG_JSON")
    if env_config_json:
        if not _json_config:
            try:
                _json_config = json.loads(env_config_json)
            except Exception as e:
                logger.warning(f"解析環境變數 CONFIG_JSON 失敗: {e}")
        return

    config_path = Path(os.getcwd()) / "config.json"
    if config_path.exists():
        try:
            mtime = config_path.stat().st_mtime
            if mtime > _last_config_load_time:
                with open(config_path, "r", encoding="utf-8") as f:
                    _json_config = json.load(f)
                    _last_config_load_time = mtime
        except Exception as e:
            logger.warning(f"讀取 config.json 失敗: {e}")


def get_config_val(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    取得系統配置值。優先自 config.json 讀取，其次自環境變數讀取，若無則回傳預設值。
    """
    _reload_config_json_if_needed()
    val = _json_config.get(key)
    if val is not None and str(val).strip() != "":
        return str(val).strip()
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip() != "":
        return env_val.strip()
    return default


@dataclass(frozen=True)
class AppConfig:
    """
    系統唯讀基礎配置模型（無資料庫伺服器、BYOK 純分析架構、支援三大自動選股榜單）。
    """
    env: str = field(default_factory=lambda: get_config_val("PYTHON_ENV", "development") or "development")
    port: int = field(default_factory=lambda: safe_int(get_config_val("PORT", "8000"), default=8000, min_val=1, max_val=65535))
    timezone: str = field(default_factory=lambda: get_config_val("TAIWAN_STOCK_TIMEZONE", "Asia/Taipei") or "Asia/Taipei")
    default_board: str = field(default_factory=lambda: get_config_val("DEFAULT_AUTO_SELECT_BOARD", "core_top20") or "core_top20")
    default_gemini_model: str = field(default_factory=lambda: get_config_val("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash")
    default_openai_model: str = field(default_factory=lambda: get_config_val("DEFAULT_OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini")
    default_auto_select_count: int = field(default_factory=lambda: safe_int(get_config_val("DEFAULT_AUTO_SELECT_COUNT", "20"), default=20, min_val=1, max_val=20))
    kline_history_days: int = field(default_factory=lambda: safe_int(get_config_val("KLINE_HISTORY_DAYS", "90"), default=90, min_val=30, max_val=365))
    cache_ttl_seconds: int = field(default_factory=lambda: safe_int(get_config_val("STOCK_CACHE_TTL_SECONDS", "900"), default=900, min_val=60, max_val=86400))

    @property
    def gemini_model(self) -> str:
        """讀取預設 Gemini 模型名稱。"""
        return get_config_val("DEFAULT_GEMINI_MODEL", self.default_gemini_model) or self.default_gemini_model

    @property
    def openai_model(self) -> str:
        """讀取預設 OpenAI 模型名稱。"""
        return get_config_val("DEFAULT_OPENAI_MODEL", self.default_openai_model) or self.default_openai_model

    @property
    def log_level(self) -> str:
        """讀取日誌等級 (預設 INFO)。"""
        return (get_config_val("LOG_LEVEL", "INFO") or "INFO").upper()

    @property
    def kline_Lookback_days(self) -> int:
        """讀取 K 線歷史回溯天數 (相容屬性名稱)。"""
        return self.kline_history_days

    @property
    def web_host(self) -> str:
        """讀取 Web 伺服器綁定 Host。"""
        return get_config_val("HOST", "0.0.0.0") or "0.0.0.0"

    @property
    def web_port(self) -> int:
        """讀取 Web 伺服器綁定 Port。"""
        return self.port


# 導出全域配置實例
config = AppConfig()
