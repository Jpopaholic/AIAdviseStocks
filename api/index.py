import os
import sys

# 將專案根目錄加入 Python 模組搜尋路徑，確保 Vercel Serverless 環境可正確匯入 src.*
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.web_server import app  # noqa: E402, F401

