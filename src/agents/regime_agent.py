# Path: src/agents/regime_agent.py
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from src.config import config, safe_float, sanitize_compliance_text
from src.services.technical_indicators import compute_all_indicators
from src.services.llm_client import call_user_llm, LLMAuthenticationError, LLMRateLimitError

logger = logging.getLogger("RegimeAgent")


# 1. 定義第一層：大盤氣候診斷模型 (Layer 1 Market Regime Assessment)
class MarketRegimeAssessment(BaseModel):
    regime: str = Field(
        ...,
        description="市場狀態，必須限為 'STRONG_BULL' (強勢多頭), 'REBOUND_BULL' (驚驚漲/震盪偏多), 'CALM_RANGE' (低波動橫盤), 'VOLATILE_RANGE' (高波動震盪), 'CORRECTION_BEAR' (震盪修正), 'PANIC_BEAR' (恐慌空頭)"
    )
    posture: str = Field(
        ...,
        description="交易姿態，必須限為 'STRONG_ATTACK' (強攻攻勢), 'MODERATE_ATTACK' (穩健進攻), 'CHOPPY_TACTICAL' (震盪靈活), 'DEFENSIVE_ACCUMULATION' (防禦承接), 'STRICT_DEFENSE' (極度保守)"
    )
    risk_multiplier: float = Field(
        ...,
        description="風險限額乘數，介於 0.15 到 1.0 之間。此乘數反映當下大盤風險偏好。"
    )
    target_cash_ratio: float = Field(
        ...,
        description="建議保持的最低現金儲備比例 (%)，介於 0.05 (5%) 到 0.90 (90%) 之間。例如強攻攻勢為 0.05~0.15，穩健進攻為 0.15~0.30，震盪靈活為 0.30~0.50，防禦承接為 0.50~0.75，極度保守為 0.75~0.90。"
    )
    allowed_buy_styles: List[str] = Field(
        ...,
        description="建議的操作型態列表，例如 ['BREAKOUT', 'PULLBACK', 'RANGE_LOW', 'DEFENSIVE_VALUE', 'MICRO_POSITION', 'NONE']"
    )
    tactical_directive: str = Field(
        ...,
        description="針對當前大盤氣候之客觀技術結構與波動風險特徵摘要 (繁體中文)。【法規硬性防線】：嚴禁出現「建議、宜、買進、賣出、停損、停利、觀望、加碼、減碼、進出場、追高、抄底、操作」等具體投資建議或交易指導字眼，僅能客觀描述大盤均線、量能與波動結構特徵。"
    )
    reason: str = Field(
        ...,
        description="判斷當前市場狀態的純客觀數據與宏觀技術分析依據 (繁體中文)，請分析短期均線、價格趨勢與成交量變化，嚴禁包含任何主觀買賣、停損、停利或觀望建議。"
    )


def generate_market_regime(
    taiex_klines: List[Dict[str, Any]],
    api_key: str = "",
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    usage_tracker: Optional[Dict[str, int]] = None,
    call_llm_fn: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    """
    第一層：分析大盤加權指數 (TAIEX) 近 30 日 K 線與技術指標，
    判定當前細粒度市場狀態 (Regime)、交易姿態 (Posture)、建議現金儲備比與戰術特別指令。
    """
    if not taiex_klines:
        logger.warning("[Regime Layer] 未收到大盤 K 線數據，退回預設正常盤整狀態。")
        return {
            "regime": "CALM_RANGE",
            "posture": "CHOPPY_TACTICAL",
            "risk_multiplier": 0.7,
            "target_cash_ratio": 0.35,
            "allowed_buy_styles": ["PULLBACK", "DEFENSIVE_VALUE"],
            "tactical_directive": "無大盤數據，採取中性盤整姿態，維持約 35% 現金儲備，優先考慮逢低拉回標的。",
            "reason": "無可用的大盤 K 線數據，自動退回到預設正常盤整狀態。",
        }

    # 計算大盤的所有技術指標（包含 MA5、MA20、MA60、VOL_MA5、VOL_MA20、RSI、MACD、DMI）
    try:
        compute_all_indicators(taiex_klines)
    except Exception as taiex_ind_err:
        logger.warning(f"[Regime Layer] 計算大盤技術指標失敗: {taiex_ind_err}")

    recent_taiex = taiex_klines[-30:]
    taiex_lines: List[str] = []

    for k in recent_taiex:
        o_price = safe_float(k.get("open") or k.get("openPrice"), 0.0)
        h_price = safe_float(k.get("high") or k.get("highPrice"), 0.0)
        l_price = safe_float(k.get("low") or k.get("lowPrice"), 0.0)
        c_price = safe_float(k.get("close") or k.get("closePrice"), 0.0)
        volume = safe_float(k.get("volume"), 0.0)

        if volume > 0:
            vol_str = f"{int(volume):,} 元 ({volume / 1e8:.1f} 億元)"
        else:
            vol_str = "N/A"
        vol_ma5_str = f"{k['vol_ma5'] / 1e8:.1f} 億元" if (k.get("vol_ma5") and k["vol_ma5"] > 0) else "N/A"
        vol_ma20_str = f"{k['vol_ma20'] / 1e8:.1f} 億元" if (k.get("vol_ma20") and k["vol_ma20"] > 0) else "N/A"

        ma5_str = f"{k['ma5']:.2f}" if k.get("ma5") is not None else "N/A"
        ma20_str = f"{k['ma20']:.2f}" if k.get("ma20") is not None else "N/A"
        ma60_str = f"{k['ma60']:.2f}" if k.get("ma60") is not None else "N/A"
        rsi_str = f"{k['rsi14']:.2f}" if k.get("rsi14") is not None else "N/A"
        macd_str = (
            f"(快線:{k['macd']:.2f}, 慢線:{k['macd_signal']:.2f}, 柱狀圖:{k['macd_hist']:.2f})"
            if (k.get("macd") is not None and k.get("macd_signal") is not None and k.get("macd_hist") is not None)
            else "N/A"
        )
        dmi_str = (
            f"(+DI:{k['plus_di']:.1f}, -DI:{k['minus_di']:.1f}, ADX:{k['adx']:.1f})"
            if (k.get("adx") is not None and k.get("plus_di") is not None and k.get("minus_di") is not None)
            else "N/A"
        )

        taiex_lines.append(
            f"  日期: {k.get('date', '')} | 開盤: {o_price:.2f} | 最高: {h_price:.2f} | 最低: {l_price:.2f} | 收盤: {c_price:.2f} | "
            f"大盤成交金額: {vol_str} | 5日均量: {vol_ma5_str} | 20日均量: {vol_ma20_str} | MA5: {ma5_str} | MA20: {ma20_str} | MA60 (季線): {ma60_str} | RSI: {rsi_str} | MACD: {macd_str} | DMI: {dmi_str}"
        )

    taiex_text = "\n".join(taiex_lines)

    system_instruction = (
        "你是一個資深的台股宏觀市場分析專家，擅長透過大盤指數走勢、成交量變化、均線排列與波動趨勢來判斷當前細粒度的市場狀態 (Market Regime) 與具體的姿態操作戰術。\n"
        "你的任務是分析給定的大盤指數 K 線數據，判定目前的市場狀態 (Regime)、應採取的交易姿態 (Posture)、風險限額乘數 (Risk Multiplier)、建議目標現金比例 (Target Cash Ratio)、允許買進動作型態 (Allowed Buy Styles) 與戰術特別指令 (Tactical Directive)。\n\n"
        "數據中已為您計算好了 MA5、MA20 (月線)、MA60 (季線)、5日均量、20日均量、RSI、MACD 與 DMI 等指標，請務必將這些指標作為您的重要評判依據！大盤成交量以每日市場總成交金額 (元 / 億元) 表示，請結合量能與 5日/20日均量之量價關係（如量增價漲、量縮價跌、量價背離等）進行宏觀剖析。\n\n"
        "【六階市場狀態 (regime) 定義】：\n"
        "- 'STRONG_BULL': 強勢多頭。均線呈強勢多頭排列 (MA5 > MA20 > MA60)，且收盤價站穩所有均線之上，帶量攻高。\n"
        "- 'REBOUND_BULL': 驚驚漲/震盪偏多。價格站穩月線 MA20 上方，但上方有季線反壓或呈現拉回打底後再彈升走勢。\n"
        "- 'CALM_RANGE': 低波動橫盤。無明顯趨勢，價格在 MA20 附近窄幅整理，成交量低迷。\n"
        "- 'VOLATILE_RANGE': 高波動震盪。單日上下波幅劇烈，多空洗盤，方向不明。\n"
        "- 'CORRECTION_BEAR': 震盪修正/初跌段。跌破月線 MA20，高點逐漸降低，但季線 MA60 仍具緩衝力道或尚未出現全面恐慌潰敗。\n"
        "- 'PANIC_BEAR': 恐慌空頭/主跌段。均線空頭排列，收盤價跌破 MA60 季線，長黑帶量下探。\n\n"
        "【五階市場風險佇列 (posture) 與對應量化結構特徵】：\n"
        "1. STRONG_ATTACK (強勢擴張期)：\n"
        "   - 適用氣候：STRONG_BULL\n"
        "   - risk_multiplier = 0.85 ~ 1.00，target_cash_ratio = 0.05 ~ 0.15\n"
        "   - allowed_buy_styles = ['BREAKOUT', 'PULLBACK', 'MOMENTUM']\n"
        "   - 結構特徵：大盤呈現多頭排列與量增突破型態，強勢動能指標佔比高。\n\n"
        "2. MODERATE_ATTACK (溫和偏多期)：\n"
        "   - 適用氣候：REBOUND_BULL\n"
        "   - risk_multiplier = 0.70 ~ 0.85，target_cash_ratio = 0.15 ~ 0.30\n"
        "   - allowed_buy_styles = ['PULLBACK', 'SUPPORT_REBOUND']\n"
        "   - 結構特徵：指數維持月線支撐上方，以回測支撐與均線靠攏型態為主。\n\n"
        "3. CHOPPY_TACTICAL (區間整理期)：\n"
        "   - 適用氣候：CALM_RANGE, VOLATILE_RANGE\n"
        "   - risk_multiplier = 0.50 ~ 0.65，target_cash_ratio = 0.30 ~ 0.50\n"
        "   - allowed_buy_styles = ['RANGE_LOW', 'DEFENSIVE_VALUE']\n"
        "   - 結構特徵：指數處於箱型區間震盪，上下檔支撐與壓力界線明顯，輪動速度快。\n\n"
        "4. DEFENSIVE_ACCUMULATION (修正收斂期)：\n"
        "   - 適用氣候：CORRECTION_BEAR, 高波動劇烈洗盤\n"
        "   - risk_multiplier = 0.25 ~ 0.45，target_cash_ratio = 0.50 ~ 0.75\n"
        "   - allowed_buy_styles = ['DEFENSIVE_VALUE', 'MICRO_POSITION']\n"
        "   - 結構特徵：大盤處於短中期均線修正階段，低乖離與高支撐安全分特徵較具韌性。\n\n"
        "5. STRICT_DEFENSE (高波動收縮期)：\n"
        "   - 適用氣候：PANIC_BEAR\n"
        "   - risk_multiplier = 0.15 ~ 0.25，target_cash_ratio = 0.75 ~ 0.90\n"
        "   - allowed_buy_styles = ['HIGH_SAFETY_ONLY'] (僅限高安全分抗跌型態；若全面重挫無支撐則單獨填 ['NONE']，切勿將 'HIGH_SAFETY_ONLY' 與 'NONE' 同時並陳)\n"
        "   - 結構特徵：大盤均線呈空頭排列且波動率劇烈放大，下檔支撐面臨全面測試。\n\n"
        "【台灣金融法規免責防線（極重要，違反即觸法）】：\n"
        "- 本系統僅提供純客觀的歷史技術指標數據整理與量化統計參數，嚴禁提供任何具體投資建議或交易指導。\n"
        "- 在 `tactical_directive` 與 `reason` 中，【嚴禁出現】以下任何字眼：「建議」、「宜」、「觀望」、「停損」、「止損」、「停利」、「買進」、「賣出」、「加碼」、「減碼」、「進場」、「出場」、「開倉」、「追高」、「抄底」、「承接」、「布局」、「操作」、「控管風險」、「現金為王」、「投資人」。\n"
        "- 請一律以純客觀的「市場技術結構與指標數據狀態陳述」撰寫。\n"
        "你的輸出必須完全符合所規定之 JSON Schema，並一律使用「繁體中文」。"
    )

    user_prompt = (
        "請根據以下大盤加權指數 (TAIEX) 最近 30 天日 K 線數據與預估技術指標，分析當前的市場狀態與細粒度姿態操作戰術：\n\n"
        "【大盤加權指數 (TAIEX) 最近 30 天數據 (最下方為最新一日行情)】：\n"
        f"{taiex_text}\n"
    )

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": MarketRegimeAssessment,
        "temperature": 0.0,
    }

    try:
        if call_llm_fn is not None:
            raw_response = call_llm_fn(
                prompt=user_prompt,
                system_instruction=system_instruction,
                model_name=model_name or config.gemini_model,
                generation_config=generation_config,
            )
        else:
            raw_response = call_user_llm(
                api_key=api_key,
                prompt=user_prompt,
                system_instruction=system_instruction,
                model_name=model_name,
                provider=provider,
                generation_config=generation_config,
                usage_tracker=usage_tracker,
            )

        res = json.loads(raw_response)
        res["risk_multiplier"] = safe_float(res.get("risk_multiplier"), default=0.7, min_val=0.15, max_val=1.0)
        res["target_cash_ratio"] = safe_float(res.get("target_cash_ratio"), default=0.35, min_val=0.05, max_val=0.90)
        raw_styles = res.get("allowed_buy_styles")
        if not isinstance(raw_styles, list) or not raw_styles:
            res["allowed_buy_styles"] = ["PULLBACK"]
        else:
            cleaned_styles = [str(s).strip().upper() for s in raw_styles if str(s).strip()]
            # 若清單中已有具體型態（如 HIGH_SAFETY_ONLY），自動移除矛盾的 NONE
            if len(cleaned_styles) > 1 and "NONE" in cleaned_styles:
                cleaned_styles = [s for s in cleaned_styles if s != "NONE"]
            res["allowed_buy_styles"] = cleaned_styles or ["NONE"]
        res["tactical_directive"] = sanitize_compliance_text(res.get("tactical_directive") or "")
        if not res["tactical_directive"]:
            res["tactical_directive"] = (
                f"當前大盤處於 {res.get('posture', 'CHOPPY_TACTICAL')} 結構，"
                f"量化風險係數為 {res['risk_multiplier']:.2f}，統計保守權重為 {res['target_cash_ratio'] * 100:.0f}%。"
            )
        res["reason"] = sanitize_compliance_text(res.get("reason") or "")
        if not res["reason"]:
            res["reason"] = res["tactical_directive"]

        latest_k = recent_taiex[-1]
        res["taiex_close"] = safe_float(latest_k.get("close"), 0.0)
        res["taiex_date"] = str(latest_k.get("date") or "")
        return res

    except (LLMAuthenticationError, LLMRateLimitError):
        # 金鑰錯誤或配額耗盡直接向上拋出，以便前端立即顯示明確提示
        raise
    except Exception as e:
        logger.error(f"[Regime Layer] 大盤氣候判定失敗: {e}")
        latest_k = recent_taiex[-1] if recent_taiex else {}
        return {
            "regime": "CALM_RANGE",
            "posture": "CHOPPY_TACTICAL",
            "risk_multiplier": 0.7,
            "target_cash_ratio": 0.35,
            "allowed_buy_styles": ["PULLBACK", "DEFENSIVE_VALUE"],
            "tactical_directive": "大盤氣候調用異常，自動退回震盪靈活姿態，建議維持 35% 現金儲備。",
            "reason": f"大盤氣候判定調用出錯，自動回退至預設正常狀態。錯誤詳情: {str(e)}",
            "taiex_close": safe_float(latest_k.get("close"), 0.0),
            "taiex_date": str(latest_k.get("date") or ""),
        }

