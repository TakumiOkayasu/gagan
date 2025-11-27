#!/usr/bin/env python3
"""
GAGAN - 画面テスト用OCRツール

Tesseract OCR + 画像前処理を使用した、画面テストのためのOCRツール。
スクリーンショットからテキストを抽出し、座標情報とともにJSON形式で出力する。
"""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
