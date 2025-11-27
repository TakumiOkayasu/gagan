"""型定義モジュール"""

from typing import TypedDict


class BBox(TypedDict):
    """境界ボックスの型定義"""

    x: int
    y: int
    width: int
    height: int


class OCRElement(TypedDict):
    """OCR要素の型定義"""

    id: int
    text: str
    bbox: BBox
    confidence: float


class OCRResult(TypedDict):
    """OCR結果の型定義"""

    elements: list[OCRElement]
    total_elements: int


class RegionResult(TypedDict):
    """テキスト領域検出結果の型定義"""

    id: int
    bbox: BBox
    confidence: float
