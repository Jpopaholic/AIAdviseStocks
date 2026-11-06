# 📊 AIAdviseStocks — 台股兩層式 AI 選股與五維度量化技術分析系統

參照 `@AIAutoStocks` 核心分析架構，專注於 **「第一層：大盤氣候診斷 (Regime Agent)」** 與 **「第二層：個股五維度量化技術分析評分 (Analyst Agent)」**，移除第三層買賣部位決策與下單邏輯，採用 **無資料庫伺服器 (No Database Server)**、**BYOK (使用者自備金鑰、費用自負)** 與 **純 Web 介面 + Markdown (`.md`) 報告匯出** 架構。

---

## ✨ 系統核心特色

1. **兩層式 AI 量化分析架構**
   - **第一層：大盤氣候診斷 (`src/agents/regime_agent.py`)**：分析加權指數 (TAIEX) 近 30 日價量與技術指標，判定六階市場狀態 (`STRONG_BULL` ~ `PANIC_BEAR`)、五階交易姿態 (`STRONG_ATTACK` ~ `STRICT_DEFENSE`)、建議最低現金儲備比與戰術特別指令。
   - **第二層：個股五維度量化評分 (`src/agents/analyst_agent.py`)**：針對入選個股評估五大維度（趨勢 `0~20`、動能 `0~20`、量能 `0~20`、安全防守 `0~20`、大盤一致性 `0~20`），由 Python 加總滿分 `100` 分並按總分高低排序，附上 80~150 字繁體中文關鍵指標分析。
2. **三大自動選股榜單 (`src/services/auto_selector.py`)**
   - **1. 大盤核心榜 (`core_top20`)**：依當日全市場成交金額排序 Top 20 指標股。
   - **2. 百元精選榜 (`under100_top20`)**：股價 `< 100 元` 且當日成交金額排序 Top 20 高流動性親民股。
   - **3. 人氣 ETF 榜 (`popular_etf`)**：固定追蹤 `0050`、`0056`、`00878`、`00929`、`00919`、`006208` 等市值型與高股息龍頭 ETF。
3. **BYOK 自備金鑰架構 (`src/services/llm_client.py`)**
   - 支援使用者於網頁前端輸入自己的 **Google Gemini API Key** (`AIza...`) 或 **OpenAI API Key** (`sk-...`)。
   - 金鑰僅保存在使用者瀏覽器 `LocalStorage`，於當次請求在記憶體中調用，伺服器端絕不寫入磁碟或日誌。
4. **純 Web 呈現與一鍵匯出 Markdown (`.md`) 分析報告**
   - 內建現代深色金融儀表板 (`src/static/index.html`)，支援即時進度顯示、隨時點擊停止 (`Stop Flag`)，並可一鍵下載完整結構化 `.md` 報告。

---

## 🚀 本地啟動方式

```bash
# 1. 安裝依賴套件
pip install -r requirements.txt

# 2. 啟動 FastAPI Web Server
uvicorn src.web_server:app --host 0.0.0.0 --port 8000 --reload
```

啟動後開啟瀏覽器前往：`http://localhost:8000`

---

## ☁️ 部署至 Vercel

本專案已內建 [vercel.json](vercel.json) 與 [api/index.py](api/index.py)，並採用無狀態分步 API (`/api/step/layer1`、`/api/step/layer2-stock`、`/api/step/markdown`) 防止 Serverless Timeout：

1. 將專案推送至 GitHub Repository。
2. 至 Vercel Dashboard 點擊 **Add New... -> Project** 並匯入此 Repository。
3. 無需設定任何必填環境變數，直接點擊 **Deploy** 即可完成部署。

