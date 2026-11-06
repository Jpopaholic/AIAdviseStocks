# Path: src/agents/analyst_agent.py
import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from src.config import config, safe_int, safe_float, get_stock_name, sanitize_compliance_text
from src.services.technical_indicators import compute_all_indicators
from src.services.llm_client import (
    call_user_llm,
    detect_provider,
    LLMAuthenticationError,
    LLMRateLimitError,
)

logger = logging.getLogger("AnalystAgent")


# =====================================================================
# 1. 定義第二層：個股五維度量化評分模型 (Structured Outputs)
# =====================================================================
class AnalystStockScore(BaseModel):
    stock_code: str = Field(
        ...,
        description="必須填寫股票代號字串，例如 '2330' 或 '00878'。此欄位必須與輸入的股票代號完全一致。"
    )
    trend_score: int = Field(
        ...,
        description="趨勢得分 (0 到 20 分)。評估均線排列（MA5、MA20、MA60）與價格波段高低點。多頭排列、價格站穩在均線之上得高分；空頭排列或價格跌破均線得低分。"
    )
    momentum_score: int = Field(
        ...,
        description="動能得分 (0 到 20 分)。評估 RSI、MACD 柱狀圖多空動能強弱與黃金/死亡交叉狀態。動能轉強、柱狀圖翻紅、黃金交叉得高分；動能消退、柱狀圖翻綠、死亡交叉得低分。"
    )
    volume_score: int = Field(
        ...,
        description="成交量得分 (0 到 20 分)。評估成交量是否價漲量增、量價配合度、VOL_MA5 與 VOL_MA20 關係。放量突破、量價配合得高分；無量盤整或量價背離得低分。"
    )
    safety_score: int = Field(
        ...,
        description="安全與防守得分 (0 到 20 分)。評估下方支撐力道與防守空間。股價回檔至強支撐、乖離率小、防守空間大得高分（代表安全）；股價突破上檔阻力但乖離率過大、高檔超買或支撐跌破得低分。"
    )
    regime_score: int = Field(
        ...,
        description="與大盤一致性得分 (0 到 20 分)。結合第一層大盤加權指數狀態 (regime) 與交易姿態 (posture)。若大盤多頭且個股強於大盤得高分，大盤空頭/防禦或大盤氣候不佳時，根據交易姿態適度調降此得分。"
    )
    confidence: float = Field(
        ...,
        description="分析信心指數 (0.0 到 1.0)。反映你對該個股技術線圖與指標分析結果的把握度。"
    )
    reason: str = Field(
        ...,
        description="該檔股票的純客觀技術指標數據摘要與評分依據（使用繁體中文，限定 80 ~ 150 字，禁止超出）。【法規硬性防線】：嚴禁出現「建議、宜、觀望、停損、止損、停利、買進、賣出、加碼、減碼、進場、出場、布局、承接、追高、抄底、操作、控管風險、目標價」等任何具體投資建議或操作指導字眼，結尾僅能客觀描述均線、動能、量能或支撐壓力結構狀態。"
    )


class AnalystAssessment(BaseModel):
    scores: List[AnalystStockScore] = Field(
        ...,
        description="所有股票的分析評分列表。必須包含所有輸入分析的股票，每檔股票各一筆。"
    )


# 系統預設的金融量化技術分析技能清單 (引導五維度評分)
DEFAULT_TRADING_SKILLS: List[str] = [
    "均線交叉策略 (Moving Average Cross): 當短線均線 (MA5) 向上突破月線 (MA20) 且有量能配合時，視為黃金交叉（應給予高趨勢/動能得分）；跌破時為死亡交叉（應降低趨勢/動能得分並調降總評）。",
    "相對強弱指標 (RSI): 評估短期超買與超賣狀態。RSI > 70 視為超買過熱（應調低安全/防守得分），RSI < 30 視為超賣超跌（若出現量縮止跌訊號，可調高安全得分代表防守空間佳）。",
    "平滑異同移動平均線 (MACD): MACD 柱狀圖紅綠柱代表多空動能。柱狀圖翻紅、快線向上突破慢線時動能轉強（應給予高動能得分）；柱狀圖翻綠、快線跌破慢線時動能走弱（應調低動能得分）。",
    "趨向指標 (DMI): ADX > 25 代表趨勢顯著。+DI 向上穿越 -DI 代表多頭強勢（趨勢/動能得分高）；-DI 向上穿越 +DI 代表空頭強勢（趨勢得分低，安全得分低）。",
    "成交量均線 (VOL MA): VOL_MA5 與 VOL_MA20 判斷量能配合。價漲量增（收盤價高於昨日且成交量大於 VOL_MA5）代表多頭動能強（成交量得分高），價漲量縮或放量下跌為量價背離（成交量得分低）。",
    "動態風險控制與防守空間評估 (Dynamic Risk Control): 評估當前股價相較於 MA20、MA60 支撐線或波段低點的乖離距離與 ATR 波動率。若股價接近強支撐區且乖離率小，則防守空間好（安全得分高）；若面臨高檔乖離過大或跌破重要支撐，則防守空間差（安全得分低）。"
]


def _format_regime_context(
    regime_assessment: Optional[Dict[str, Any]],
    taiex_klines: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """整理第一層大盤氣候診斷摘要，供第二層個股分析師評估 regime_score 使用。"""
    lines: List[str] = []
    if taiex_klines:
        latest_taiex = taiex_klines[-1]
        c_val = safe_float(latest_taiex.get("close"), 0.0)
        ma20_val = latest_taiex.get("ma20")
        ma60_val = latest_taiex.get("ma60")
        ma20_str = f"{ma20_val:.2f}" if ma20_val is not None else "N/A"
        ma60_str = f"{ma60_val:.2f}" if ma60_val is not None else "N/A"
        lines.append(
            f"● 當前大盤加權指數 (TAIEX) 收盤價: {c_val:.2f} 點 | MA20: {ma20_str} | MA60: {ma60_str}"
        )

    if regime_assessment:
        regime = regime_assessment.get("regime", "CALM_RANGE")
        posture = regime_assessment.get("posture", "CHOPPY_TACTICAL")
        cash_ratio = safe_float(regime_assessment.get("target_cash_ratio"), 0.35) * 100
        styles = ", ".join(regime_assessment.get("allowed_buy_styles") or ["PULLBACK"])
        directive = regime_assessment.get("tactical_directive", "")
        lines.append(
            f"● 第一層大盤氣候診斷：市場狀態={regime} | 交易姿態={posture} | "
            f"建議現金儲備比={cash_ratio:.0f}% | 建議操作型態=[{styles}]"
        )
        if directive:
            lines.append(f"● 第一層戰術特別指令：{directive}")

    return "\n".join(lines) if lines else "● 大盤狀態：正常盤整 (CALM_RANGE / CHOPPY_TACTICAL)"


def generate_analyst_assessments(
    stock_codes: List[str],
    klines_map: Dict[str, List[Dict[str, Any]]],
    api_key: str = "",
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    regime_assessment: Optional[Dict[str, Any]] = None,
    candidates_metadata: Optional[List[Dict[str, Any]]] = None,
    extra_skills: Optional[List[str]] = None,
    usage_tracker: Optional[Dict[str, int]] = None,
    should_stop_fn: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, str, Optional[Dict[str, Any]]], None]] = None,
    call_llm_fn: Optional[Callable[..., str]] = None,
    inter_call_delay_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    第二層：個股量化技術分析師代理。
    結合第一層大盤氣候診斷結果與金融分析 Skills，對三大榜單自動挑選出的個股
    逐一進行五維度量化評分（趨勢、動能、量能、安全、大盤一致性，各 0~20 分），
    由 Python 加總計算 total_score (0~100) 並按總分由高至低排序回傳。
    """
    if not stock_codes:
        return []

    # 建立候選股中繼資料索引（如榜單標籤、成交金額、漲跌幅）
    meta_by_code: Dict[str, Dict[str, Any]] = {}
    if candidates_metadata:
        for item in candidates_metadata:
            c = str(item.get("stock_code", "")).strip()
            if c:
                meta_by_code[c] = item

    # 1. 確保所有個股與大盤皆已完成技術指標計算
    for code in list(stock_codes) + ["TAIEX"]:
        klines = klines_map.get(code, [])
        if klines:
            try:
                if "ma5" not in klines[-1] or "rsi14" not in klines[-1]:
                    compute_all_indicators(klines)
            except Exception as ind_err:
                logger.warning(f"[Analyst Layer] 計算股票 {code} 技術指標失敗: {ind_err}")

    # 2. 準備金融量化分析技能清單
    skills = list(DEFAULT_TRADING_SKILLS)
    if extra_skills:
        skills.extend([s for s in extra_skills if s and s.strip()])
    skills_text = "\n".join([f"{idx + 1}. {s}" for idx, s in enumerate(skills)])

    # 3. 過濾有效個股
    valid_stock_codes = [c for c in stock_codes if c != "TAIEX" and klines_map.get(c)]
    if not valid_stock_codes:
        return []

    generation_config_analyst = {
        "response_mime_type": "application/json",
        "response_schema": AnalystStockScore,
        "temperature": 0.0,
    }

    taiex_context_text = _format_regime_context(
        regime_assessment=regime_assessment,
        taiex_klines=klines_map.get("TAIEX", []),
    )

    resolved_provider = detect_provider(api_key or "dummy", provider)
    if inter_call_delay_sec is None:
        inter_call_delay_sec = 0.3 if resolved_provider == "openai" else 1.2

    analyst_system_instruction = (
        "你是一個極其資深且敏銳的台股量化投資分析師。你熟悉台股市場特性與技術線圖分析。\n"
        "你的任務是分析給定個股的日 K 線數據，進行獨立且客觀的量化評分與技術指標分析。\n"
        "你需要為該檔個股進行五個維度的評分 (各個維度為 0 ~ 20 分，總分會由 Python 程式加總計算，你不需回傳總分，也不需填寫股票收盤價)。\n\n"
        "【重要評分提示】：\n"
        "- 你的評分是為了反映個股強弱。如果個股呈現多頭排列、放量突破、MACD翻紅或RSI強勢等信號，請勇敢地給予高分（趨勢得分與動能得分可達16~20分）。\n"
        "- 請拉開評分差距，切忌盲目保守。強勢股給予高分，弱勢股給予低分。\n"
        "- 切忌僅憑單一負面訊號就直接全盤否定個股價值。\n"
        "- 請同時為你的分析結果給出信心指數 `confidence` (0.0 到 1.0)。\n\n"
        "量化評分維度指引：\n"
        "1. 趨勢得分 (trend_score, 0 ~ 20 分)：評估均線排列（MA5、MA20、MA60）與價格波段高低點。多頭排列、價格站穩在均線之上得高分；空頭排列或價格跌破均線得低分。\n"
        "2. 動能得分 (momentum_score, 0 ~ 20 分)：評估 RSI、MACD 柱狀圖多空動能強弱與黃金/死亡交叉狀態。動能轉強、柱狀圖翻紅、黃金交叉得高分；動能消退、柱狀圖翻綠、死亡交叉得低分。\n"
        "3. 成交量得分 (volume_score, 0 ~ 20 分)：評估成交量是否價漲量增、量價配合度、VOL_MA5 與 VOL_MA20 關係。放量突破、量價配合得高分；無量盤整或量價背離得低分。\n"
        "4. 安全與防守得分 (safety_score, 0 ~ 20 分)：評估下方支撐力道與防守空間。股價回檔至強支撐、乖離率小、防守空間大得高分（代表安全）；股價突破上檔阻力但乖離率過大、高檔超買或支撐跌破得低分。\n"
        "5. 大盤一致性得分 (regime_score, 0 ~ 20 分)：結合第一層大盤加權指數狀態與交易姿態。若大盤多頭且個股強於大盤得高分，大盤空頭/防禦或大盤氣候不佳時，根據交易姿態適度調降此得分。\n\n"
        f"你的金融量化分析技能包含：\n{skills_text}\n\n"
        "請嚴格遵守以下指示：\n"
        "1. 你的輸出必須完全符合所規定的 JSON Schema (AnalystStockScore)，不可包含額外文字。\n"
        "2. 你的分析理由請一律使用「繁體中文」，嚴格限制在 80 到 150 字以內，精簡指出關鍵技術指標信號，請勿贅述每日歷史細節。\n"
        "3. 【台灣金融法規免責防線（極重要，違反即觸法）】：\n"
        "   - 本系統僅提供純客觀的歷史技術指標數據整理與量化評分，絕對不可提供任何投資建議或操作指令。\n"
        "   - 在 `reason` 中，【嚴禁出現】以下任何字眼或語意：「建議」、「宜」、「觀望」、「保持觀望」、「停損」、「止損」、「停利」、「買進」、「買入」、「賣出」、「加碼」、「減碼」、「進場」、「出場」、「布局」、「承接」、「追高」、「抄底」、「持有」、「操作」、「買點」、「賣點」、「目標價」、「控管風險」、「現金為王」、「投資人」。\n"
        "   - 文章結尾必須是純客觀的「技術面結構與指標狀態總結」（例如：「整體均線與動能指標呈現空方排列特徵，下檔季線為重要支撐觀察位。」或「綜合量價與指標數據，短線維持多頭排列與量能擴增格局。」），切勿給予任何行為指導。"
    )

    raw_scores: List[Dict[str, Any]] = []
    total_count = len(valid_stock_codes)

    for idx, code in enumerate(valid_stock_codes):
        # 檢查使用者是否發送停止執行旗標 (stop_requested)
        if should_stop_fn is not None and should_stop_fn():
            logger.info(f"[Analyst Layer] 偵測到停止執行旗標 (stop_requested)，於第 {idx + 1}/{total_count} 檔 ({code}) 安全終止。")
            break

        stock_name = get_stock_name(code)
        logger.info(f"[Analyst Layer] 正在評估個股 ({idx + 1}/{total_count}): {code} ({stock_name})...")

        if progress_callback is not None:
            try:
                progress_callback(idx + 1, total_count, code, None)
            except Exception:
                pass

        klines = klines_map.get(code, [])
        recent_klines = klines[-15:]
        klines_lines: List[str] = []

        for k in recent_klines:
            ma5_str = f"{k['ma5']:.2f}" if k.get("ma5") is not None else "N/A"
            ma20_str = f"{k['ma20']:.2f}" if k.get("ma20") is not None else "N/A"
            ma60_str = f"{k['ma60']:.2f}" if k.get("ma60") is not None else "N/A"
            rsi_str = f"{k['rsi14']:.2f}" if k.get("rsi14") is not None else "N/A"

            vol = safe_float(k.get("volume"), 0.0)
            vol_ma5 = k.get("vol_ma5")
            vol_ma20 = k.get("vol_ma20")
            vol_ma5_str = f"{vol_ma5:,.0f}" if vol_ma5 is not None else "N/A"
            vol_ma20_str = f"{vol_ma20:,.0f}" if vol_ma20 is not None else "N/A"
            vol_ratio_str = f" (為5日均量之 {vol / vol_ma5:.1f}倍)" if (vol_ma5 and vol_ma5 > 0) else ""

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
            atr_val = k.get("atr14")
            atr_pct = k.get("atr_pct")
            atr_str = f"{atr_val:.2f} ({atr_pct:.1f}%)" if (atr_val is not None and atr_pct is not None) else "N/A"

            klines_lines.append(
                f"  日期: {k.get('date', '')} | 開盤: {k.get('open')} | 最高: {k.get('high')} | 最低: {k.get('low')} | 收盤: {k.get('close')} | "
                f"MA5: {ma5_str} | MA20: {ma20_str} | MA60 (季線): {ma60_str} | RSI: {rsi_str} | "
                f"成交量: {vol:,.0f}{vol_ratio_str} (5日均量: {vol_ma5_str}, 20日均量: {vol_ma20_str}) | MACD: {macd_str} | DMI: {dmi_str} | ATR(14): {atr_str}"
            )

        klines_text = "\n".join(klines_lines)
        meta_info = meta_by_code.get(code, {})
        board_label = meta_info.get("board_label", "")
        board_line = f"● 入選榜單：{board_label}\n" if board_label else ""

        analyst_user_prompt = (
            f"請針對股票代號 {code} ({stock_name}) 進行個股日 K 線的五維度技術分析與量化評分。\n\n"
            f"{taiex_context_text}\n"
            f"{board_line}\n"
            f"● 股票代號 {code} ({stock_name}) 最近 15 天 K 線數據與技術指標 (最下方為最新一日行情)：\n"
            f"{klines_text}\n\n"
            "請基於上述 K 線、技術指標與第一層大盤氣候，產出該股票的五維度量化評分與純客觀繁體中文技術指標摘要（80~150字，嚴禁出現『建議、觀望、停損、停利、買進、賣出、進出場、操作、控管風險』等任何投資建議字眼）。"
        )

        try:
            if call_llm_fn is not None:
                raw_analyst_response = call_llm_fn(
                    prompt=analyst_user_prompt,
                    system_instruction=analyst_system_instruction,
                    model_name=model_name or config.gemini_model,
                    generation_config=generation_config_analyst,
                )
            else:
                raw_analyst_response = call_user_llm(
                    api_key=api_key,
                    prompt=analyst_user_prompt,
                    system_instruction=analyst_system_instruction,
                    model_name=model_name,
                    provider=provider,
                    generation_config=generation_config_analyst,
                    usage_tracker=usage_tracker,
                )

            s_item = json.loads(raw_analyst_response)
            if isinstance(s_item, dict):
                s_item["stock_code"] = code
                raw_scores.append(s_item)
            else:
                raise ValueError("LLM 未返回有效的 JSON 物件格式")

        except (LLMAuthenticationError, LLMRateLimitError):
            # 金鑰無效或額度耗盡立即向外拋出，讓前端即時顯示錯誤
            raise
        except Exception as e:
            logger.warning(f"[Analyst Layer] 分析股票 {code} ({stock_name}) 失敗: {e}，補上預設中性評分。")
            raw_scores.append({
                "stock_code": code,
                "trend_score": 12,
                "momentum_score": 12,
                "volume_score": 12,
                "safety_score": 12,
                "regime_score": 12,
                "confidence": 0.4,
                "reason": f"該股進行 AI 技術指標評估時發生解析或連線異常，系統自動套用中性基準評分（60分），建議搭配均線與量價結構人工複核。({str(e)[:45]})",
            })

        # 呼叫間隔緩衝，避免瞬間觸發速率限制
        if idx < total_count - 1 and inter_call_delay_sec > 0:
            if should_stop_fn is None or not should_stop_fn():
                time.sleep(inter_call_delay_sec)

    # 4. 由 Python 端安全轉換五維度分數、計算總分 (0~100) 並補充行情指標摘要
    analyst_scores: List[Dict[str, Any]] = []
    seen_codes = set()

    for s_item in raw_scores:
        code = str(s_item.get("stock_code", "")).strip()
        if not code or code == "TAIEX" or code in seen_codes:
            continue
        seen_codes.add(code)

        klines = klines_map.get(code, [])
        latest_k = klines[-1] if klines else {}
        prev_k = klines[-2] if len(klines) >= 2 else latest_k
        meta = meta_by_code.get(code, {})

        price = safe_float(
            latest_k.get("close") or meta.get("close"),
            default=10.0,
            min_val=0.01,
        )
        prev_close = safe_float(prev_k.get("close"), default=price, min_val=0.01)
        change_pct = (
            safe_float(meta.get("change_pct"), default=((price - prev_close) / prev_close * 100.0))
            if prev_close > 0
            else 0.0
        )
        turnover = safe_float(
            meta.get("turnover") or latest_k.get("turnover"),
            default=price * safe_float(latest_k.get("volume"), 0.0),
            min_val=0.0,
        )

        trend = safe_int(s_item.get("trend_score"), default=10, min_val=0, max_val=20)
        momentum = safe_int(s_item.get("momentum_score"), default=10, min_val=0, max_val=20)
        volume = safe_int(s_item.get("volume_score"), default=10, min_val=0, max_val=20)
        safety = safe_int(s_item.get("safety_score"), default=10, min_val=0, max_val=20)
        regime = safe_int(s_item.get("regime_score"), default=10, min_val=0, max_val=20)
        total_score = trend + momentum + volume + safety + regime
        confidence = safe_float(s_item.get("confidence"), default=0.75, min_val=0.0, max_val=1.0)

        atr14_val = safe_float(latest_k.get("atr14"), default=0.0, min_val=0.0)
        atr_pct_val = safe_float(latest_k.get("atr_pct"), default=0.0, min_val=0.0)
        if atr_pct_val >= 3.0:
            vol_tier = "HIGH"
        elif atr_pct_val <= 1.5:
            vol_tier = "LOW"
        else:
            vol_tier = "NORMAL"

        reason_text = sanitize_compliance_text(s_item.get("reason") or "")
        if not reason_text:
            reason_text = "均線與量價結構呈現中性整理格局，各項技術指標處於區間收斂狀態。"
        elif len(reason_text) > 160:
            reason_text = reason_text[:155].rstrip("，、。") + "。"

        score_entry = {
            "stock_code": code,
            "stock_name": meta.get("stock_name") or get_stock_name(code),
            "board_key": meta.get("board_key", ""),
            "board_label": meta.get("board_label", ""),
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "turnover": round(turnover, 0),
            "latest_date": str(latest_k.get("date") or meta.get("latest_date") or ""),
            "trend_score": trend,
            "momentum_score": momentum,
            "volume_score": volume,
            "safety_score": safety,
            "regime_score": regime,
            "total_score": total_score,
            "confidence": round(confidence, 2),
            "ma5": round(safe_float(latest_k.get("ma5"), 0.0), 2) if latest_k.get("ma5") is not None else None,
            "ma20": round(safe_float(latest_k.get("ma20"), 0.0), 2) if latest_k.get("ma20") is not None else None,
            "ma60": round(safe_float(latest_k.get("ma60"), 0.0), 2) if latest_k.get("ma60") is not None else None,
            "rsi14": round(safe_float(latest_k.get("rsi14"), 0.0), 2) if latest_k.get("rsi14") is not None else None,
            "adx": round(safe_float(latest_k.get("adx"), 0.0), 1) if latest_k.get("adx") is not None else None,
            "atr14": round(atr14_val, 2),
            "atr_pct": round(atr_pct_val, 2),
            "volatility_tier": vol_tier,
            "reason": reason_text,
        }
        analyst_scores.append(score_entry)

    # 按總分 (total_score) 與信心指數 (confidence) 由高至低排序，並附加排名
    analyst_scores.sort(key=lambda x: (x["total_score"], x["confidence"]), reverse=True)
    for rank_idx, item in enumerate(analyst_scores, start=1):
        item["rank"] = rank_idx

    return analyst_scores

