#!/usr/bin/env python3
"""
GAGAN - 画面テスト用OCRツール

Tesseract OCR + 画像前処理を使用した、画面テストのためのOCRツール。
スクリーンショットからテキストを抽出し、座標情報とともにJSON形式で出力する。
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

import cv2
import numpy as np
import pytesseract
from PIL import Image

# PaddleOCRは遅延インポート (オプション依存)

# デフォルトのワーカー数 (CPUコア数、最大8)
DEFAULT_WORKERS = min(cpu_count(), 8)

# =============================================================================
# 型定義
# =============================================================================


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


# =============================================================================
# 定数・設定
# =============================================================================

# ガンマ補正用ルックアップテーブル (遅延初期化でメモリ節約)
_gamma_table_cache: dict[float, np.ndarray] = {}


def get_gamma_table(gamma: float) -> np.ndarray:
    """ガンマ補正用ルックアップテーブルを取得する (遅延初期化・ベクトル演算)。"""
    if gamma not in _gamma_table_cache:
        indices = np.arange(256, dtype=np.float32)
        _gamma_table_cache[gamma] = (
            np.power(indices / 255.0, 1.0 / gamma) * 255
        ).astype(np.uint8)
    return _gamma_table_cache[gamma]


# 誤認識しやすい文字のセット
SUSPICIOUS_CHARS = frozenset(["占", "上", "浴", "甲", "丘", "士", "充", "民", "音"])

# UI誤認識補正辞書
UI_CORRECTIONS = {
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
KATAKANA_CORRECTIONS = {
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


# =============================================================================
# 画像変換ユーティリティ
# =============================================================================


def to_grayscale(image: np.ndarray | Image.Image) -> np.ndarray:
    """画像をグレースケールに変換する (PIL/NumPy両対応)。"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    if len(img_array.shape) == 2:
        return img_array
    if img_array.shape[2] == 4:
        # RGBA -> BGR -> Gray
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # RGB -> BGR -> Gray
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def apply_rotation_if_needed(
    gray: np.ndarray, detect_rotation_flag: bool
) -> np.ndarray:
    """必要に応じて回転補正を適用する。"""
    if not detect_rotation_flag:
        return gray
    angle = detect_rotation(gray)
    if abs(angle) > 0.5:
        return rotate_image(gray, -angle)
    return gray


def upscale_image(
    image: np.ndarray, scale: float, interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """画像をスケールアップする。

    Args:
        image: 入力画像
        scale: スケール倍率 (0より大きい値)
        interpolation: 補間方法

    Returns:
        スケール後の画像。スケールが不正な場合は元の画像をそのまま返す。
    """
    if scale <= 0 or not np.isfinite(scale):
        return image

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image

    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return image

    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def crop_with_margin(
    image: np.ndarray, bbox: BBox, margin: int = 10
) -> Optional[np.ndarray]:
    """numpy配列から境界ボックスを切り出す。

    Returns:
        切り出した配列。座標が不正な場合はNone。
    """
    h, w = image.shape[:2]
    x1 = max(0, bbox["x"] - margin)
    y1 = max(0, bbox["y"] - margin)
    x2 = min(w, bbox["x"] + bbox["width"] + margin)
    y2 = min(h, bbox["y"] + bbox["height"] + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()


# =============================================================================
# 画像解析
# =============================================================================


def detect_rotation(image: np.ndarray) -> float:
    """画像の回転角度を検出する。"""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        _, theta = line[0]
        angle = np.degrees(theta) - 90
        if angle < -45:
            angle += 180
        elif angle > 45:
            angle -= 180
        angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        if abs(median_angle) > 0.5:
            return float(median_angle)

    return 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """画像を回転させる。"""
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def detect_theme(image: np.ndarray) -> str:
    """画像の平均輝度からテーマを判定する。"""
    return "dark" if np.mean(image) < 128 else "light"


def calculate_text_density(image: np.ndarray) -> float:
    """画像内のテキスト密度を計算する。"""
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    non_zero = cv2.countNonZero(binary)
    total = image.shape[0] * image.shape[1]
    return non_zero / total if total > 0 else 0.0


def select_optimal_psm(image: np.ndarray) -> int:
    """画像特性に基づいて最適なPSMを選択する。"""
    h, w = image.shape[:2]
    aspect_ratio = w / h if h > 0 else 1.0
    text_density = calculate_text_density(image)

    if text_density < 0.1:
        return 11  # Sparse text
    if aspect_ratio > 5:
        return 7  # Single text line
    if aspect_ratio < 0.3:
        return 4  # Single column
    return 6  # Single uniform block


# =============================================================================
# 画像前処理
# =============================================================================


def upscale_if_needed(
    image: np.ndarray, min_height: int = 1000
) -> tuple[np.ndarray, float]:
    """画像が小さい場合にアップスケールする。"""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0
    if h < min_height:
        scale = min_height / h
        return upscale_image(image, scale), scale
    return image, 1.0


def upscale_small_ui_elements(
    image: np.ndarray, min_text_height: int = 32
) -> tuple[np.ndarray, float]:
    """小さいUI要素の認識精度向上のためスケーリングする。"""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0
    if h < 100:
        scale = max(2.0, min_text_height / 12)
        return upscale_image(image, scale), scale
    return image, 1.0


def apply_sharpening(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """アンシャープマスクでシャープネスを強化する。"""
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)


def preprocess_image_screenshot(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """スクリーンショット専用の前処理を実行する。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)

    # ダークモード検出と自動反転
    if detect_theme(gray) == "dark":
        gray = cv2.bitwise_not(gray)

    gray, _ = upscale_small_ui_elements(gray)
    return cv2.bilateralFilter(gray, 5, 50, 50)


def preprocess_image_light(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """軽量前処理を実行する。二値化を行わない。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)
    return cv2.bilateralFilter(gray, 5, 50, 50)


def preprocess_image_adaptive(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """適応的閾値処理を使用した画像前処理を実行する。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    # 前処理パイプライン
    normalized = cv2.equalizeHist(gray)
    gamma_corrected = cv2.LUT(normalized, get_gamma_table(1.5))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gamma_corrected)

    sharpened = apply_sharpening(contrasted)
    denoised = cv2.bilateralFilter(sharpened, 5, 75, 75)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )

    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def preprocess_image_otsu(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """Otsu二値化を使用した画像前処理を実行する。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    gamma_corrected = cv2.LUT(gray, get_gamma_table(1.2))
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def preprocess_image_inverted(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """白抜き文字用の前処理を実行する。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    inverted = cv2.bitwise_not(gray)
    gamma_corrected = cv2.LUT(inverted, get_gamma_table(1.2))
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# 前処理関数のマッピング
PreprocessFunc = Callable[[np.ndarray | Image.Image, bool], np.ndarray]
PREPROCESS_FUNCTIONS: dict[str, PreprocessFunc] = {
    "adaptive": preprocess_image_adaptive,
    "otsu": preprocess_image_otsu,
    "inverted": preprocess_image_inverted,
    "light": preprocess_image_light,
    "screenshot": preprocess_image_screenshot,
}


# =============================================================================
# 超解像・テキスト検出 (オプション機能)
# =============================================================================


def upscale_with_super_resolution(
    image: np.ndarray, scale: int = 4, model_name: str = "espcn"
) -> np.ndarray:
    """超解像によるアップスケール。"""
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        model_paths = [
            Path(__file__).parent / "models" / f"{model_name.upper()}_x{scale}.pb",
            Path.home() / ".gagan" / "models" / f"{model_name.upper()}_x{scale}.pb",
            Path(f"/usr/share/gagan/models/{model_name.upper()}_x{scale}.pb"),
        ]

        model_path = next((str(p) for p in model_paths if p.exists()), None)

        if model_path is None:
            print(
                "警告: 超解像モデルが見つかりません。INTER_CUBICにフォールバック",
                file=sys.stderr,
            )
            return upscale_image(image, scale)

        sr.readModel(model_path)
        sr.setModel(model_name.lower(), scale)
        return sr.upsample(image)

    except (AttributeError, Exception) as e:
        if isinstance(e, AttributeError):
            print(
                "警告: opencv-contrib-pythonが必要です。INTER_CUBICにフォールバック",
                file=sys.stderr,
            )
        else:
            print(
                f"警告: 超解像処理に失敗しました: {e}。INTER_CUBICにフォールバック",
                file=sys.stderr,
            )
        return upscale_image(image, scale)


def detect_text_regions_east(
    image: np.ndarray, confidence_threshold: float = 0.5, nms_threshold: float = 0.4
) -> list[dict[str, Any]]:
    """EASTテキスト検出器でテキスト領域を検出する。"""
    try:
        model_paths = [
            Path(__file__).parent / "models" / "frozen_east_text_detection.pb",
            Path.home() / ".gagan" / "models" / "frozen_east_text_detection.pb",
            Path("/usr/share/gagan/models/frozen_east_text_detection.pb"),
        ]

        model_path = next((str(p) for p in model_paths if p.exists()), None)
        if model_path is None:
            print(
                "警告: EASTモデルが見つかりません。テキスト検出をスキップ",
                file=sys.stderr,
            )
            return []

        orig_h, orig_w = image.shape[:2]
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32

        if new_w == 0 or new_h == 0:
            return []

        ratio_w, ratio_h = orig_w / new_w, orig_h / new_h
        resized = cv2.resize(image, (new_w, new_h))

        net = cv2.dnn.readNet(model_path)
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), True, False
        )
        net.setInput(blob)

        output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = net.forward(output_layers)

        boxes, confidences = [], []
        num_rows, num_cols = scores.shape[2:4]

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0_data, x1_data = geometry[0, 0, y], geometry[0, 1, y]
            x2_data, x3_data = geometry[0, 2, y], geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                if scores_data[x] < confidence_threshold:
                    continue

                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angles_data[x]
                cos, sin = np.cos(angle), np.sin(angle)

                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                end_x = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
                end_y = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
                start_x, start_y = int(end_x - w), int(end_y - h)

                boxes.append(
                    [
                        int(start_x * ratio_w),
                        int(start_y * ratio_h),
                        int(end_x * ratio_w),
                        int(end_y * ratio_h),
                    ]
                )
                confidences.append(float(scores_data[x]))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
            confidences,
            confidence_threshold,
            nms_threshold,
        )

        return [
            {
                "x": max(0, boxes[i][0]),
                "y": max(0, boxes[i][1]),
                "width": boxes[i][2] - boxes[i][0],
                "height": boxes[i][3] - boxes[i][1],
                "confidence": confidences[i],
            }
            for i in (indices.flatten() if len(indices) > 0 else [])
        ]

    except Exception as e:
        print(f"警告: EASTテキスト検出に失敗しました: {e}", file=sys.stderr)
        return []


# =============================================================================
# OCR実行
# =============================================================================


def execute_ocr(
    image: np.ndarray,
    lang: str = "jpn+eng",
    psm: Optional[int] = None,
    preserve_spaces: bool = True,
    tessdata_dir: Optional[str] = None,
) -> OCRResult:
    """Tesseract OCRを実行し、テキストと座標情報を取得する。"""
    # 空画像チェック
    if image is None or image.size == 0:
        return {"elements": [], "total_elements": 0}

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return {"elements": [], "total_elements": 0}

    if psm is None:
        psm = select_optimal_psm(image)

    config_parts = ["--oem 3", f"--psm {psm}"]
    if preserve_spaces:
        config_parts.append("-c preserve_interword_spaces=1")
    if tessdata_dir:
        config_parts.append(f"--tessdata-dir {tessdata_dir}")

    config = " ".join(config_parts)

    ocr_data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    elements: list[OCRElement] = []
    for i, (text, conf) in enumerate(zip(ocr_data["text"], ocr_data["conf"])):
        text = text.strip()
        conf = int(conf)
        if not text or conf == -1:
            continue

        elements.append(
            {
                "id": len(elements),
                "text": text,
                "bbox": {
                    "x": int(ocr_data["left"][i]),
                    "y": int(ocr_data["top"][i]),
                    "width": int(ocr_data["width"][i]),
                    "height": int(ocr_data["height"][i]),
                },
                "confidence": round(conf / 100.0, 2),
            }
        )

    return {"elements": elements, "total_elements": len(elements)}


# PaddleOCRインスタンス管理 (遅延初期化・メモリ解放対応)
class PaddleOCRManager:
    """PaddleOCRインスタンスを管理するクラス (省メモリ対応)。"""

    _instance: Optional[Any] = None
    _lang: Optional[str] = None

    @classmethod
    def get_instance(cls, lang: str = "japan") -> Any:
        """PaddleOCRインスタンスを取得する (遅延初期化)。"""
        # 言語が変わった場合は再作成
        if cls._instance is not None and cls._lang == lang:
            return cls._instance

        # 既存インスタンスがあれば解放
        cls.release()

        try:
            from paddleocr import PaddleOCR

            # 言語マッピング
            lang_map = {
                "jpn": "japan",
                "eng": "en",
                "jpn+eng": "japan",  # PaddleOCRは単一言語、日本語優先
                "chi_sim": "ch",
                "kor": "korean",
            }
            paddle_lang = lang_map.get(lang, "japan")

            # show_log=Falseで不要な出力を抑制、use_gpu=Falseでメモリ節約
            cls._instance = PaddleOCR(lang=paddle_lang, show_log=False, use_gpu=False)
            cls._lang = lang
            return cls._instance

        except ImportError:
            print(
                "エラー: PaddleOCRがインストールされていません。\n"
                "インストール: pip install paddlepaddle paddleocr",
                file=sys.stderr,
            )
            sys.exit(1)

    @classmethod
    def release(cls) -> None:
        """PaddleOCRインスタンスを解放してメモリを節約する。"""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            cls._lang = None
            # ガベージコレクションを促進
            import gc

            gc.collect()


def execute_ocr_paddleocr(
    image: np.ndarray,
    lang: str = "jpn+eng",
) -> OCRResult:
    """PaddleOCRを使用してOCRを実行する。"""
    # 空画像チェック
    if image is None or image.size == 0:
        return {"elements": [], "total_elements": 0}

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return {"elements": [], "total_elements": 0}

    # グレースケールの場合はBGRに変換 (PaddleOCRはカラー画像を期待)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    ocr = PaddleOCRManager.get_instance(lang)

    # 新しいAPI (predict) を試し、失敗したら旧API (ocr) を使用
    try:
        result = ocr.predict(image)
    except (TypeError, AttributeError):
        result = ocr.ocr(image)

    elements: list[OCRElement] = []

    # 結果の解析 (新旧APIで形式が異なる)
    if result is None:
        return {"elements": elements, "total_elements": 0}

    # 新しいAPI形式の処理
    if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
        for item in result:
            # 新API: 辞書形式
            if isinstance(item, dict) and "rec_texts" in item:
                rec_texts = item.get("rec_texts", [])
                rec_scores = item.get("rec_scores", [])
                dt_polys = item.get("dt_polys", [])

                for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                    text = str(text).strip()
                    if not text:
                        continue

                    # 4点座標からbboxを計算
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))

                    elements.append(
                        {
                            "id": len(elements),
                            "text": text,
                            "bbox": {
                                "x": x_min,
                                "y": y_min,
                                "width": x_max - x_min,
                                "height": y_max - y_min,
                            },
                            "confidence": round(float(score), 2),
                        }
                    )
            # 旧API: リスト形式 [[[[x1,y1],...], (text, conf)], ...]
            elif isinstance(item, list):
                for line in item:
                    if line is None or len(line) < 2:
                        continue
                    try:
                        box, (text, confidence) = line
                        text = str(text).strip()
                        if not text:
                            continue

                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        x_min, x_max = int(min(xs)), int(max(xs))
                        y_min, y_max = int(min(ys)), int(max(ys))

                        elements.append(
                            {
                                "id": len(elements),
                                "text": text,
                                "bbox": {
                                    "x": x_min,
                                    "y": y_min,
                                    "width": x_max - x_min,
                                    "height": y_max - y_min,
                                },
                                "confidence": round(float(confidence), 2),
                            }
                        )
                    except (ValueError, TypeError):
                        continue

    return {"elements": elements, "total_elements": len(elements)}


def execute_ocr_with_engine(
    image: np.ndarray,
    engine: str = "tesseract",
    lang: str = "jpn+eng",
    psm: Optional[int] = None,
    preserve_spaces: bool = True,
    tessdata_dir: Optional[str] = None,
) -> OCRResult:
    """指定されたエンジンでOCRを実行する。"""
    if engine == "paddleocr":
        return execute_ocr_paddleocr(image, lang)
    return execute_ocr(image, lang, psm, preserve_spaces, tessdata_dir)


# =============================================================================
# ベンチマーク機能
# =============================================================================


@dataclass
class BenchmarkResult:
    """ベンチマーク結果を保持するデータクラス"""

    engine: str
    elapsed_time: float
    element_count: int
    avg_confidence: float
    elements: list[OCRElement] = field(default_factory=list)


def run_benchmark(
    image: np.ndarray,
    lang: str = "jpn+eng",
    psm: Optional[int] = None,
    tessdata_dir: Optional[str] = None,
) -> dict[str, BenchmarkResult]:
    """TesseractとPaddleOCRのベンチマークを実行する。"""
    results = {}

    # Tesseract
    print("  Tesseract OCR 実行中...")
    start_time = time.perf_counter()
    tesseract_result = execute_ocr(image, lang, psm, tessdata_dir=tessdata_dir)
    tesseract_time = time.perf_counter() - start_time

    tesseract_elements = tesseract_result["elements"]
    tesseract_avg_conf = (
        sum(e["confidence"] for e in tesseract_elements) / len(tesseract_elements)
        if tesseract_elements
        else 0.0
    )

    results["tesseract"] = BenchmarkResult(
        engine="tesseract",
        elapsed_time=tesseract_time,
        element_count=len(tesseract_elements),
        avg_confidence=tesseract_avg_conf,
        elements=tesseract_elements,
    )

    # PaddleOCR
    try:
        print("  PaddleOCR 実行中...")
        start_time = time.perf_counter()
        paddle_result = execute_ocr_paddleocr(image, lang)
        paddle_time = time.perf_counter() - start_time

        paddle_elements = paddle_result["elements"]
        paddle_avg_conf = (
            sum(e["confidence"] for e in paddle_elements) / len(paddle_elements)
            if paddle_elements
            else 0.0
        )

        results["paddleocr"] = BenchmarkResult(
            engine="paddleocr",
            elapsed_time=paddle_time,
            element_count=len(paddle_elements),
            avg_confidence=paddle_avg_conf,
            elements=paddle_elements,
        )
    except Exception as e:
        print(f"  PaddleOCR エラー: {e}", file=sys.stderr)
        results["paddleocr"] = BenchmarkResult(
            engine="paddleocr",
            elapsed_time=0.0,
            element_count=0,
            avg_confidence=0.0,
            elements=[],
        )

    return results


def print_benchmark_results(results: dict[str, BenchmarkResult]) -> None:
    """ベンチマーク結果を表示する。"""
    print("\n" + "=" * 60)
    print("ベンチマーク結果")
    print("=" * 60)

    # テーブルヘッダー
    print(f"{'エンジン':<15} {'処理時間':>10} {'要素数':>8} {'平均信頼度':>10}")
    print("-" * 60)

    for result in results.values():
        print(
            f"{result.engine:<15} {result.elapsed_time:>9.3f}s "
            f"{result.element_count:>8} {result.avg_confidence:>10.2%}"
        )

    print("-" * 60)

    # 速度比較
    if "tesseract" in results and "paddleocr" in results:
        tess_time = results["tesseract"].elapsed_time
        paddle_time = results["paddleocr"].elapsed_time

        if paddle_time > 0 and tess_time > 0:
            if paddle_time < tess_time:
                speedup = tess_time / paddle_time
                print(f"PaddleOCR は Tesseract より {speedup:.1f}x 高速")
            else:
                speedup = paddle_time / tess_time
                print(f"Tesseract は PaddleOCR より {speedup:.1f}x 高速")

    print("=" * 60 + "\n")


def export_benchmark_json(
    results: dict[str, BenchmarkResult],
    source_image: str,
    resolution: tuple[int, int],
    output_path: Path,
) -> None:
    """ベンチマーク結果をJSONファイルに出力する。"""
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_image": source_image,
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "benchmark": {
            engine: {
                "elapsed_time": result.elapsed_time,
                "element_count": result.element_count,
                "avg_confidence": result.avg_confidence,
                "elements": result.elements,
            }
            for engine, result in results.items()
        },
    }

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ベンチマーク結果を保存しました: {output_path}")


# =============================================================================
# 再OCR処理 (統合版)
# =============================================================================


def _retry_element_ocr(
    elem: OCRElement,
    image_array: np.ndarray,
    lang: str,
    tessdata_dir: Optional[str],
    preprocess_methods: list[str],
    retry_psms: list[int],
    margin: int = 10,
) -> tuple[OCRElement, str]:
    """単一要素の再OCRを実行する (並列処理用の共通関数)。

    Returns:
        (改善後の要素, ログメッセージ)
    """
    region_array = crop_with_margin(image_array, elem["bbox"], margin)
    if region_array is None:
        return elem, f"  スキップ (座標不正): '{elem['text']}'"

    best_result, best_confidence = elem, elem["confidence"]
    original_confidence = elem["confidence"]

    for method_name in preprocess_methods:
        try:
            preprocess_func = PREPROCESS_FUNCTIONS[method_name]
            processed = preprocess_func(region_array, False)

            # 小さい領域はスケールアップ
            h = processed.shape[0]
            if 0 < h < 32:
                processed = upscale_image(processed, 32 / h)

            for psm in retry_psms:
                region_result = execute_ocr(
                    processed, lang, psm=psm, tessdata_dir=tessdata_dir
                )
                for new_elem in region_result["elements"]:
                    if new_elem["confidence"] > best_confidence:
                        best_confidence = new_elem["confidence"]
                        best_result = {
                            "id": elem["id"],
                            "text": new_elem["text"],
                            "bbox": elem["bbox"],
                            "confidence": new_elem["confidence"],
                        }
        except Exception:
            continue

    # ログメッセージ生成
    if best_confidence > original_confidence:
        log_msg = (
            f"  改善: '{elem['text']}' ({original_confidence:.2f}) "
            f"-> '{best_result['text']}' ({best_confidence:.2f})"
        )
    else:
        log_msg = f"  改善なし: '{elem['text']}' ({original_confidence:.2f})"

    return best_result, log_msg


def _retry_char_element_ocr(
    elem: OCRElement,
    image_array: np.ndarray,
    lang: str,
    tessdata_dir: Optional[str],
) -> tuple[OCRElement, str]:
    """単一要素の文字単位再OCRを実行する (並列処理用)。

    Returns:
        (改善後の要素, ログメッセージ)
    """
    region_array = crop_with_margin(image_array, elem["bbox"], margin=5)
    if region_array is None:
        return elem, f"  スキップ (座標不正): '{elem['text']}'"

    original_text = elem["text"]
    original_confidence = elem["confidence"]

    # グレースケール変換と高解像度化
    if len(region_array.shape) == 3:
        gray = cv2.cvtColor(region_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = region_array

    upscaled = upscale_image(gray, 4)
    sharpened = apply_sharpening(upscaled)

    best_text, best_confidence = original_text, original_confidence

    preprocess_variants = [
        sharpened,
        cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ),
        cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]

    for processed in preprocess_variants:
        for psm in [6, 13]:
            try:
                result = execute_ocr(
                    processed, lang, psm=psm, tessdata_dir=tessdata_dir
                )
                for new_elem in result["elements"]:
                    new_text, new_conf = new_elem["text"], new_elem["confidence"]

                    old_count = sum(1 for c in original_text if c in SUSPICIOUS_CHARS)
                    new_count = sum(1 for c in new_text if c in SUSPICIOUS_CHARS)

                    if new_count < old_count and new_conf >= best_confidence * 0.9:
                        best_text, best_confidence = new_text, new_conf
                    elif new_conf > best_confidence:
                        best_text, best_confidence = new_text, new_conf
            except Exception:
                continue

    # ログメッセージ生成
    if best_text != original_text:
        log_msg = (
            f"  文字単位改善: '{original_text}' ({original_confidence:.2f}) "
            f"-> '{best_text}' ({best_confidence:.2f})"
        )
    elif best_confidence > original_confidence:
        log_msg = (
            f"  信頼度改善: '{original_text}' "
            f"({original_confidence:.2f}) -> ({best_confidence:.2f})"
        )
    else:
        log_msg = f"  改善なし: '{original_text}' ({original_confidence:.2f})"

    return {
        "id": elem["id"],
        "text": best_text,
        "bbox": elem["bbox"],
        "confidence": best_confidence,
    }, log_msg


def _run_retry_ocr(
    target_elements: list[OCRElement],
    other_elements: list[OCRElement],
    image_array: np.ndarray,
    retry_func: Callable[..., tuple[OCRElement, str]],
    retry_kwargs: dict[str, Any],
    parallel: bool,
    workers: int,
    description: str,
) -> OCRResult:
    """再OCRの共通処理を実行する。

    Args:
        target_elements: 再OCR対象の要素リスト
        other_elements: そのまま保持する要素リスト
        image_array: 画像配列
        retry_func: 再OCR関数
        retry_kwargs: 再OCR関数の追加引数
        parallel: 並列処理フラグ
        workers: ワーカー数
        description: 処理説明文
    """
    if not target_elements:
        return {
            "elements": other_elements,
            "total_elements": len(other_elements),
        }

    print(f"{description} {len(target_elements)}件に対して再OCR実行中...")

    if parallel and len(target_elements) > 1:
        # 並列処理
        improved_elements = list(other_elements)
        with ThreadPoolExecutor(
            max_workers=min(workers, len(target_elements))
        ) as executor:
            futures = {
                executor.submit(retry_func, elem, image_array, **retry_kwargs): elem[
                    "id"
                ]
                for elem in target_elements
            }

            for future in as_completed(futures):
                result, log_msg = future.result()
                improved_elements.append(result)
                print(log_msg)

        improved_elements.sort(key=lambda x: x["id"])
    else:
        # 順次処理
        improved_elements = list(other_elements)
        for elem in target_elements:
            result, log_msg = retry_func(elem, image_array, **retry_kwargs)
            improved_elements.append(result)
            print(log_msg)
        improved_elements.sort(key=lambda x: x["id"])

    return {"elements": improved_elements, "total_elements": len(improved_elements)}


def retry_low_confidence_ocr(
    original_image: np.ndarray | Image.Image,
    ocr_result: OCRResult,
    confidence_threshold: float = 0.8,
    lang: str = "jpn+eng",
    tessdata_dir: Optional[str] = None,
    parallel: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> OCRResult:
    """信頼度が低い要素に対して、異なる前処理で再OCRを実行する。"""
    low_conf = [
        e for e in ocr_result["elements"] if e["confidence"] < confidence_threshold
    ]
    high_conf = [
        e for e in ocr_result["elements"] if e["confidence"] >= confidence_threshold
    ]

    image_array = (
        np.array(original_image)
        if isinstance(original_image, Image.Image)
        else original_image
    )

    return _run_retry_ocr(
        target_elements=low_conf,
        other_elements=high_conf,
        image_array=image_array,
        retry_func=_retry_element_ocr,
        retry_kwargs={
            "lang": lang,
            "tessdata_dir": tessdata_dir,
            "preprocess_methods": ["adaptive", "otsu", "inverted", "light"],
            "retry_psms": [7, 8, 13],
            "margin": 10,
        },
        parallel=parallel,
        workers=workers,
        description="低信頼度要素",
    )


def contains_suspicious_chars(text: str) -> bool:
    """テキストに誤認識しやすい文字が含まれているかチェックする。"""
    return any(char in SUSPICIOUS_CHARS for char in text)


def retry_character_level_ocr(
    original_image: np.ndarray | Image.Image,
    ocr_result: OCRResult,
    lang: str = "jpn+eng",
    tessdata_dir: Optional[str] = None,
    parallel: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> OCRResult:
    """誤認識しやすい文字を含む要素に対して、文字単位で再OCRを実行する。"""
    suspicious = [
        e for e in ocr_result["elements"] if contains_suspicious_chars(e["text"])
    ]
    non_suspicious = [
        e for e in ocr_result["elements"] if not contains_suspicious_chars(e["text"])
    ]

    image_array = (
        np.array(original_image)
        if isinstance(original_image, Image.Image)
        else original_image
    )

    return _run_retry_ocr(
        target_elements=suspicious,
        other_elements=non_suspicious,
        image_array=image_array,
        retry_func=_retry_char_element_ocr,
        retry_kwargs={
            "lang": lang,
            "tessdata_dir": tessdata_dir,
        },
        parallel=parallel,
        workers=workers,
        description="疑わしい文字を含む要素",
    )


# =============================================================================
# 結果マージ・後処理
# =============================================================================


def calculate_iou(bbox1: BBox, bbox2: BBox) -> float:
    """2つのbounding boxのIoUを計算する。"""
    x1_min, y1_min = bbox1["x"], bbox1["y"]
    x1_max, y1_max = x1_min + bbox1["width"], y1_min + bbox1["height"]
    x2_min, y2_min = bbox2["x"], bbox2["y"]
    x2_max, y2_max = x2_min + bbox2["width"], y2_min + bbox2["height"]

    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = bbox1["width"] * bbox1["height"]
    area2 = bbox2["width"] * bbox2["height"]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def merge_ocr_results(
    result1: OCRResult, result2: OCRResult, iou_threshold: float = 0.5
) -> OCRResult:
    """2つのOCR結果をマージする。重複は信頼度の高い方を採用。"""
    merged_elements: list[OCRElement] = []
    result2_used = [False] * len(result2["elements"])

    for elem1 in result1["elements"]:
        best_match_idx, best_iou = -1, 0.0

        for i, elem2 in enumerate(result2["elements"]):
            if result2_used[i]:
                continue
            iou = calculate_iou(elem1["bbox"], elem2["bbox"])
            if iou > iou_threshold and iou > best_iou:
                best_iou, best_match_idx = iou, i

        if best_match_idx >= 0:
            elem2 = result2["elements"][best_match_idx]
            result2_used[best_match_idx] = True
            merged_elements.append(
                elem1 if elem1["confidence"] >= elem2["confidence"] else elem2
            )
        else:
            merged_elements.append(elem1)

    for i, elem2 in enumerate(result2["elements"]):
        if not result2_used[i]:
            merged_elements.append(elem2)

    for i, elem in enumerate(merged_elements):
        elem["id"] = i

    return {"elements": merged_elements, "total_elements": len(merged_elements)}


def correct_ui_text(text: str) -> str:
    """UI要素の誤認識パターンを修正する。"""
    result = text
    for wrong, correct in UI_CORRECTIONS.items():
        if wrong in result:
            result = result.replace(wrong, correct)
    return result


def correct_japanese_text(text: str) -> str:
    """日本語OCRの誤認識パターンを修正する。"""

    def is_katakana(char: str) -> bool:
        if not char:
            return False
        code = ord(char)
        return (0x30A0 <= code <= 0x30FF) or (0xFF65 <= code <= 0xFF9F)

    chars = list(text)
    result = []

    for i, char in enumerate(chars):
        if char in KATAKANA_CORRECTIONS:
            prev_char = chars[i - 1] if i > 0 else ""
            next_char = chars[i + 1] if i < len(chars) - 1 else ""
            if is_katakana(prev_char) or is_katakana(next_char):
                result.append(KATAKANA_CORRECTIONS[char])
            else:
                result.append(char)
        else:
            result.append(char)

    return "".join(result)


def filter_by_confidence(
    elements: list[OCRElement], min_confidence: float = 0.3
) -> list[OCRElement]:
    """信頼度に基づいて要素をフィルタリングする。"""
    filtered: list[OCRElement] = []
    for elem in elements:
        conf, text_len = elem["confidence"], len(elem["text"])

        if text_len <= 1:
            required_conf = min_confidence * 1.5
        elif text_len <= 2:
            required_conf = min_confidence * 1.2
        else:
            required_conf = min_confidence

        if conf >= required_conf:
            filtered.append(elem)
        elif text_len >= 5 and conf >= min_confidence * 0.8:
            filtered.append(elem)

    return filtered


def convert_to_json(
    ocr_result: OCRResult, source_image: str, resolution: tuple[int, int]
) -> str:
    """OCR結果をJSON形式に変換する。"""
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_image": source_image,
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "elements": ocr_result["elements"],
    }
    return json.dumps(output_data, ensure_ascii=False, indent=2)


# =============================================================================
# メイン処理
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサを構築する。"""
    parser = argparse.ArgumentParser(description="GAGAN - 画面テスト用OCRツール")

    # 基本オプション
    parser.add_argument(
        "images", nargs="+", help="OCRを実行する画像ファイルのパス(複数指定可)"
    )
    parser.add_argument(
        "-o", "--output", help="出力JSONファイル名(単一ファイル時のみ有効)"
    )
    parser.add_argument(
        "--lang", default="jpn+eng", help="OCR言語設定(デフォルト: jpn+eng)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード(前処理後の画像を保存)"
    )
    parser.add_argument(
        "--keep-debug-images",
        action="store_true",
        help="デバッグ画像を削除せず保持する",
    )

    # 前処理モード
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="画像前処理をスキップする"
    )
    parser.add_argument("--light", action="store_true", help="軽量モード(二値化なし)")
    parser.add_argument(
        "--screenshot", action="store_true", help="スクリーンショット専用モード"
    )
    parser.add_argument(
        "--aggressive", action="store_true", help="高精度モード(複数手法併用)"
    )
    parser.add_argument("--inverted", action="store_true", help="白抜き文字モード")
    parser.add_argument(
        "--detect-rotation", action="store_true", help="回転検出と補正を有効化"
    )

    # 精度向上オプション
    parser.add_argument(
        "--fast", action="store_true", help="高速モード(精度向上機能をOFF)"
    )
    parser.add_argument(
        "--psm", type=str, default="auto", help="Page Segmentation Mode"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="最小信頼度フィルタ"
    )
    parser.add_argument("--char-retry", action="store_true", help="文字単位再OCRを実行")
    parser.add_argument(
        "--no-japanese-correct", action="store_true", help="日本語誤認識修正を無効化"
    )

    # 高精度オプション
    parser.add_argument("--best", action="store_true", help="tessdata_bestを使用")
    parser.add_argument("--tessdata-dir", type=str, help="tessdataディレクトリを指定")
    parser.add_argument(
        "--post-process", action="store_true", help="形態素解析による後処理を有効化"
    )

    # 高度なオプション
    parser.add_argument(
        "--super-resolution", action="store_true", help="超解像によるアップスケール"
    )
    parser.add_argument(
        "--text-detection", action="store_true", help="EASTテキスト領域検出"
    )
    parser.add_argument("--max-accuracy", action="store_true", help="最高精度モード")

    # 並列処理オプション
    parser.add_argument("--parallel", action="store_true", help="並列処理を有効化")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"並列処理のワーカー数 (デフォルト: {DEFAULT_WORKERS})",
    )

    # OCRエンジン選択
    parser.add_argument(
        "--engine",
        type=str,
        choices=["tesseract", "paddleocr"],
        default="tesseract",
        help="OCRエンジン (tesseract/paddleocr、デフォルト: tesseract)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="ベンチマークモード (TesseractとPaddleOCRを比較)",
    )

    return parser


def resolve_tessdata_dir(args: argparse.Namespace) -> Optional[str]:
    """tessdata_dirを解決する。"""
    if args.tessdata_dir:
        return args.tessdata_dir

    if not args.best:
        return None

    tessdata_best_paths = [
        "/usr/share/tesseract-ocr/5/tessdata_best",
        "/usr/share/tesseract-ocr/4.00/tessdata_best",
        "/usr/local/share/tessdata_best",
    ]

    for path in tessdata_best_paths:
        if Path(path).exists():
            return path

    print(
        "警告: tessdata_bestが見つかりません。標準tessdataを使用します", file=sys.stderr
    )
    return None


def _process_single_preprocess(
    method_name: str,
    suffix: str,
    image_array: np.ndarray,
    detect_rotation: bool,
    lang: str,
    psm: Optional[int],
    tessdata_dir: Optional[str],
    debug: bool,
    image_path: Path,
) -> tuple[str, OCRResult, Optional[Path]]:
    """単一の前処理を実行する (並列処理用)。"""
    preprocess_func = PREPROCESS_FUNCTIONS[method_name]
    processed = preprocess_func(image_array, detect_rotation)

    debug_path = None
    if debug:
        debug_path = image_path.with_suffix(suffix)
        cv2.imwrite(str(debug_path), processed)

    result = execute_ocr(processed, lang, psm=psm, tessdata_dir=tessdata_dir)
    return method_name, result, debug_path


def process_with_aggressive_mode(
    image_array: np.ndarray,
    args: argparse.Namespace,
    psm: Optional[int],
    tessdata_dir: Optional[str],
    image_path: Path,
    methods: list[tuple[str, str]],
    parallel: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> tuple[OCRResult, Optional[list[Path]]]:
    """高精度モードで処理する (統合版)。"""
    debug_paths: Optional[list[Path]] = [] if args.debug else None

    if parallel and len(methods) > 1:
        # 並列処理
        results: list[OCRResult] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(methods))) as executor:
            futures = {
                executor.submit(
                    _process_single_preprocess,
                    name,
                    suffix,
                    image_array,
                    args.detect_rotation,
                    args.lang,
                    psm,
                    tessdata_dir,
                    args.debug,
                    image_path,
                ): name
                for name, suffix in methods
            }

            for future in as_completed(futures):
                name, result, debug_path = future.result()
                print(f"{name}処理: {result['total_elements']}要素")
                results.append(result)
                if debug_path and debug_paths is not None:
                    debug_paths.append(debug_path)
    else:
        # 順次処理
        results = []
        for name, suffix in methods:
            preprocess_func = PREPROCESS_FUNCTIONS[name]
            processed = preprocess_func(image_array, args.detect_rotation)
            if args.debug and debug_paths is not None:
                debug_path = image_path.with_suffix(suffix)
                cv2.imwrite(str(debug_path), processed)
                debug_paths.append(debug_path)
                print(f"{name}処理済み画像を保存しました: {debug_path}")

            result = execute_ocr(
                processed, args.lang, psm=psm, tessdata_dir=tessdata_dir
            )
            print(f"{name}処理: {result['total_elements']}要素")
            results.append(result)

    # 結果をマージ
    merged = results[0]
    for result in results[1:]:
        merged = merge_ocr_results(merged, result)
    print(f"マージ後: {merged['total_elements']}要素")

    return merged, debug_paths


def process_single_mode(
    image_array: np.ndarray,
    preprocess_func: PreprocessFunc,
    args: argparse.Namespace,
    psm: Optional[int],
    tessdata_dir: Optional[str],
    image_path: Path,
    suffix: str,
    engine: str = "tesseract",
) -> tuple[OCRResult, Optional[Path]]:
    """単一の前処理モードで処理する。"""
    processed = preprocess_func(image_array, args.detect_rotation)
    debug_path = None

    if args.debug:
        debug_path = image_path.with_suffix(suffix)
        cv2.imwrite(str(debug_path), processed)
        print(f"前処理済み画像を保存しました: {debug_path}")

    ocr_result = execute_ocr_with_engine(
        processed, engine, args.lang, psm=psm, tessdata_dir=tessdata_dir
    )
    return ocr_result, debug_path


def apply_post_processing(
    ocr_result: OCRResult,
    image_array: np.ndarray,
    args: argparse.Namespace,
    tessdata_dir: Optional[str],
    parallel: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> OCRResult:
    """後処理を適用する。"""
    # 低信頼度要素の再OCR (信頼度0.8未満は強制的に再OCR)
    ocr_result = retry_low_confidence_ocr(
        image_array,
        ocr_result,
        0.8,  # 固定閾値
        args.lang,
        tessdata_dir,
        parallel=parallel,
        workers=workers,
    )

    # 文字単位再OCR
    if args.char_retry:
        ocr_result = retry_character_level_ocr(
            image_array,
            ocr_result,
            args.lang,
            tessdata_dir,
            parallel=parallel,
            workers=workers,
        )

    # UI誤認識補正
    if args.screenshot:
        for elem in ocr_result["elements"]:
            elem["text"] = correct_ui_text(elem["text"])

    # 日本語誤認識修正
    if not args.no_japanese_correct:
        for elem in ocr_result["elements"]:
            elem["text"] = correct_japanese_text(elem["text"])

    # 信頼度フィルタリング
    if args.min_confidence > 0:
        original_count = len(ocr_result["elements"])
        ocr_result["elements"] = filter_by_confidence(
            ocr_result["elements"], args.min_confidence
        )
        ocr_result["total_elements"] = len(ocr_result["elements"])
        filtered_count = original_count - ocr_result["total_elements"]
        if filtered_count > 0 and args.debug:
            print(f"信頼度フィルタ: {filtered_count}要素を除外")

    # 形態素解析
    if args.post_process:
        try:
            from janome.tokenizer import Tokenizer

            tokenizer = Tokenizer()

            for elem in ocr_result["elements"]:
                tokens = list(tokenizer.tokenize(elem["text"]))
                corrected_parts = []
                for token in tokens:
                    word = token.surface
                    if token.part_of_speech.split(",")[0] == "名詞":
                        word = correct_japanese_text(word)
                    corrected_parts.append(word)
                elem["text"] = "".join(corrected_parts)
        except ImportError:
            print(
                "警告: janomeがインストールされていません。--post-processは無視されます",
                file=sys.stderr,
            )

    return ocr_result


def cleanup_debug_images(
    debug_image_path: Optional[Path | list[Path]], keep: bool
) -> None:
    """デバッグ画像をクリーンアップする。"""
    if not debug_image_path or keep:
        return

    paths = (
        debug_image_path if isinstance(debug_image_path, list) else [debug_image_path]
    )
    for path in paths:
        if path.exists():
            path.unlink()
            print(f"前処理済み画像を削除しました: {path}")


def process_image(
    image_path: Path,
    args: argparse.Namespace,
    tessdata_dir: Optional[str],
    parallel: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> bool:
    """単一の画像を処理する。"""
    debug_image_path: Optional[Path | list[Path]] = None

    try:
        image = Image.open(image_path)
        resolution = (image.width, image.height)

        # 最初にNumPy配列に変換 (以降の処理で再利用)
        image_array = np.array(image)

        # PSMのバリデーション (0-13の範囲)
        psm: Optional[int] = None
        if args.psm != "auto":
            try:
                psm = int(args.psm)
                if not (0 <= psm <= 13):
                    print(
                        f"警告: PSM {psm} は範囲外です (0-13)。autoに設定します",
                        file=sys.stderr,
                    )
                    psm = None
            except ValueError:
                print(
                    f"警告: PSM '{args.psm}' は無効です。autoに設定します",
                    file=sys.stderr,
                )
                psm = None

        # 超解像
        if args.super_resolution:
            if len(image_array.shape) == 3:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

            upscaled = upscale_with_super_resolution(img_bgr, scale=4)
            image_array = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            print(
                f"超解像適用: {resolution} -> "
                f"({image_array.shape[1]}, {image_array.shape[0]})"
            )

        # テキスト領域検出モード
        if args.text_detection:
            if len(image_array.shape) == 3:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

            text_regions = detect_text_regions_east(img_bgr)

            if text_regions:
                print(f"テキスト領域検出: {len(text_regions)}領域")
                all_elements: list[OCRElement] = []
                h, w = image_array.shape[:2]

                for region in text_regions:
                    margin = 5
                    x1 = max(0, region["x"] - margin)
                    y1 = max(0, region["y"] - margin)
                    x2 = min(w, region["x"] + region["width"] + margin)
                    y2 = min(h, region["y"] + region["height"] + margin)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    region_array = image_array[y1:y2, x1:x2]
                    preprocess_func = (
                        preprocess_image_light
                        if args.light
                        else preprocess_image_adaptive
                    )
                    processed = preprocess_func(region_array, args.detect_rotation)

                    region_result = execute_ocr(
                        processed, args.lang, psm=7, tessdata_dir=tessdata_dir
                    )

                    for elem in region_result["elements"]:
                        elem["id"] = len(all_elements)
                        elem["bbox"]["x"] += x1
                        elem["bbox"]["y"] += y1
                        all_elements.append(elem)

                ocr_result: OCRResult = {
                    "elements": all_elements,
                    "total_elements": len(all_elements),
                }
            else:
                print("テキスト領域が検出されませんでした。通常モードで処理")
                processed = preprocess_image_adaptive(image_array, args.detect_rotation)
                ocr_result = execute_ocr(
                    processed, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )

        # 前処理なし
        elif args.no_preprocessing:
            processed = to_grayscale(image_array)
            if args.debug:
                debug_image_path = image_path.with_suffix(".preprocessed.png")
                cv2.imwrite(str(debug_image_path), processed)
                print(f"前処理済み画像を保存しました: {debug_image_path}")
            ocr_result = execute_ocr(
                processed, args.lang, psm=psm, tessdata_dir=tessdata_dir
            )

        # スクリーンショット + 高精度
        elif args.screenshot and args.aggressive:
            print("スクリーンショット高精度モードで処理中...")
            screenshot_psm = psm if psm is not None else 11
            methods = [
                ("screenshot", ".screenshot.png"),
                ("light", ".light.png"),
                ("inverted", ".inverted.png"),
            ]
            ocr_result, debug_image_path = process_with_aggressive_mode(
                image_array,
                args,
                screenshot_psm,
                tessdata_dir,
                image_path,
                methods,
                parallel,
                workers,
            )

        # スクリーンショットモード
        elif args.screenshot:
            screenshot_psm = psm if psm is not None else 11
            ocr_result, debug_image_path = process_single_mode(
                image_array,
                preprocess_image_screenshot,
                args,
                screenshot_psm,
                tessdata_dir,
                image_path,
                ".screenshot.png",
                args.engine,
            )

        # 軽量モード
        elif args.light:
            ocr_result, debug_image_path = process_single_mode(
                image_array,
                preprocess_image_light,
                args,
                psm,
                tessdata_dir,
                image_path,
                ".light.png",
                args.engine,
            )

        # 高精度モード
        elif args.aggressive:
            print("高精度モードで処理中...")
            methods = [
                ("adaptive", ".adaptive.png"),
                ("otsu", ".otsu.png"),
                ("inverted", ".inverted.png"),
            ]
            ocr_result, debug_image_path = process_with_aggressive_mode(
                image_array,
                args,
                psm,
                tessdata_dir,
                image_path,
                methods,
                parallel,
                workers,
            )

        # 白抜き文字モード
        elif args.inverted:
            ocr_result, debug_image_path = process_single_mode(
                image_array,
                preprocess_image_inverted,
                args,
                psm,
                tessdata_dir,
                image_path,
                ".inverted.png",
                args.engine,
            )

        # 通常モード
        else:
            ocr_result, debug_image_path = process_single_mode(
                image_array,
                preprocess_image_adaptive,
                args,
                psm,
                tessdata_dir,
                image_path,
                ".preprocessed.png",
                args.engine,
            )

        # 後処理
        ocr_result = apply_post_processing(
            ocr_result, image_array, args, tessdata_dir, parallel, workers
        )

        # JSON出力
        json_output = convert_to_json(ocr_result, image_path.name, resolution)

        if args.output and len(args.images) == 1:
            output_path = Path(args.output)
        else:
            output_path = image_path.with_suffix(image_path.suffix + ".ocr.json")

        output_path.write_text(json_output, encoding="utf-8")
        print(f"OCR結果を保存しました: {output_path}")
        print(f"認識されたテキスト要素数: {ocr_result['total_elements']}")

        return True

    except Exception as e:
        print(f"エラーが発生しました ({image_path}): {e}", file=sys.stderr)
        return False

    finally:
        cleanup_debug_images(debug_image_path, args.keep_debug_images)


def main() -> int:
    """CLIエントリーポイント。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    # --max-accuracyの展開
    if args.max_accuracy:
        args.best = True
        args.post_process = True
        args.aggressive = True
        args.char_retry = True

    # --fastの処理
    if args.fast:
        args.min_confidence = 0.0
        args.no_japanese_correct = True
        if args.psm == "auto":
            args.psm = "3"

    # 信頼度閾値のバリデーション
    if not (0.0 <= args.min_confidence <= 1.0):
        print(
            f"警告: min-confidence {args.min_confidence} は範囲外です (0.0-1.0)。"
            "0.3に設定します",
            file=sys.stderr,
        )
        args.min_confidence = 0.3

    # 複数ファイル警告
    if len(args.images) > 1 and args.output:
        print("警告: 複数ファイル指定時は-oオプションは無視されます", file=sys.stderr)

    # ワーカー数のバリデーション
    if args.workers < 1:
        print(
            f"警告: workers {args.workers} は無効です。1に設定します",
            file=sys.stderr,
        )
        args.workers = 1

    tessdata_dir = resolve_tessdata_dir(args)
    parallel = args.parallel
    workers = args.workers

    # 存在するファイルのみ抽出
    valid_image_paths: list[Path] = []
    error_count = 0
    for image_file in args.images:
        image_path = Path(image_file)
        if not image_path.exists():
            print(
                f"エラー: 画像ファイルが見つかりません: {image_file}", file=sys.stderr
            )
            error_count += 1
        else:
            valid_image_paths.append(image_path)

    # ベンチマークモード
    if args.benchmark:
        print("ベンチマークモードで実行中...")
        for image_path in valid_image_paths:
            print(f"\n処理中: {image_path}")
            try:
                image = Image.open(image_path)
                resolution = (image.width, image.height)
                image_array = np.array(image)

                # 前処理
                if args.screenshot:
                    processed = preprocess_image_screenshot(
                        image_array, args.detect_rotation
                    )
                elif args.light:
                    processed = preprocess_image_light(
                        image_array, args.detect_rotation
                    )
                else:
                    processed = preprocess_image_adaptive(
                        image_array, args.detect_rotation
                    )

                # PSM解決
                psm = None
                if args.psm != "auto":
                    try:
                        psm = int(args.psm)
                    except ValueError:
                        pass

                # ベンチマーク実行
                results = run_benchmark(processed, args.lang, psm, tessdata_dir)
                print_benchmark_results(results)

                # 結果をJSON出力
                output_path = image_path.with_suffix(
                    image_path.suffix + ".benchmark.json"
                )
                export_benchmark_json(results, image_path.name, resolution, output_path)

            except Exception as e:
                print(f"エラー ({image_path}): {e}", file=sys.stderr)
                error_count += 1

        return 0 if error_count == 0 else 1

    success_count = 0

    # 複数ファイルの並列処理
    if parallel and len(valid_image_paths) > 1:
        print(
            f"並列処理モード: {workers}ワーカーで"
            f"{len(valid_image_paths)}ファイルを処理中..."
        )
        with ThreadPoolExecutor(
            max_workers=min(workers, len(valid_image_paths))
        ) as executor:
            futures = {
                executor.submit(
                    process_image, image_path, args, tessdata_dir, parallel, workers
                ): image_path
                for image_path in valid_image_paths
            }

            for future in as_completed(futures):
                image_path = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"エラー ({image_path}): {e}", file=sys.stderr)
                    error_count += 1
    else:
        # 順次処理
        for image_path in valid_image_paths:
            if process_image(image_path, args, tessdata_dir, parallel, workers):
                success_count += 1
            else:
                error_count += 1

    if len(args.images) > 1:
        print(f"\n処理完了: 成功 {success_count}件, エラー {error_count}件")

    # PaddleOCRインスタンスを解放 (メモリ節約)
    PaddleOCRManager.release()

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
