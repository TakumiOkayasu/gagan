"""設定モジュール"""

from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Optional

# デフォルトのワーカー数 (CPUコア数、最大8)
DEFAULT_WORKERS = min(cpu_count(), 8)

# 高解像度画像の閾値 (この高さ以上の画像ではアップスケールをスキップ)
_high_resolution_threshold = 1500


def get_high_resolution_threshold() -> int:
    """高解像度閾値を取得する。"""
    return _high_resolution_threshold


def set_high_resolution_threshold(value: int) -> None:
    """高解像度閾値を設定する (--no-upscale オプション用)。"""
    global _high_resolution_threshold
    _high_resolution_threshold = value


# 誤認識しやすい文字のセット
SUSPICIOUS_CHARS = frozenset(["占", "上", "浴", "甲", "丘", "士", "充", "民", "音"])

# UI誤認識補正辞書
UI_CORRECTIONS: dict[str, str] = {
    "0K": "OK",
    "Cance1": "Cancel",
    "C1ose": "Close",
    "Fi1e": "File",
    "Edi七": "Edit",
    "He1p": "Help",
    "Vie\\/\\/": "View",
    "Too1s": "Tools",
    "0ptions": "Options",
    "Sett1ngs": "Settings",
    "App1y": "Apply",
    "De1ete": "Delete",
    "Se1ect": "Select",
    "0pen": "Open",
    "C1ear": "Clear",
    "Ref1esh": "Refresh",
    "Re1oad": "Reload",
}

# 日本語誤認識補正 (カタカナ文脈用)
KATAKANA_CORRECTIONS: dict[str, str] = {
    "一": "ー",
    "口": "ロ",
    "力": "カ",
    "工": "エ",
    "夕": "タ",
    "二": "ニ",
    "八": "ハ",
}


@dataclass
class OCRConfig:
    """OCR設定を保持するデータクラス"""

    lang: str = "jpn+eng"
    psm: Optional[int] = None
    tessdata_dir: Optional[str] = None
    preserve_spaces: bool = True


@dataclass
class ProcessingOptions:
    """処理オプションを保持するデータクラス"""

    detect_rotation: bool = False
    debug: bool = False
    keep_debug_images: bool = False
    no_upscale: bool = False
