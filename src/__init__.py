"""
GAGAN - 画面テスト用OCRツール

Tesseract OCR + 画像前処理を使用した、画面テストのためのOCRツール。
"""

from src.types import BBox, OCRElement, OCRResult
from src.config import OCRConfig, ProcessingOptions

__all__ = [
    "BBox",
    "OCRElement",
    "OCRResult",
    "OCRConfig",
    "ProcessingOptions",
]

__version__ = "1.0.0"
