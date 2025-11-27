"""後処理モジュール"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.config import (
    DEFAULT_WORKERS,
    KATAKANA_CORRECTIONS,
    SUSPICIOUS_CHARS,
    UI_CORRECTIONS,
)
from src.ocr_engines import execute_ocr
from src.preprocessing import (
    PREPROCESS_FUNCTIONS,
    apply_sharpening,
    crop_with_margin,
    upscale_image,
)
from src.types import BBox, OCRElement, OCRResult


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
# 再OCR処理
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

            # 小さい領域のみスケールアップ
            h = processed.shape[0]
            if 0 < h < 32:
                processed = upscale_image(processed, 32 / h)
            elif 32 <= h < 64:
                processed = upscale_image(processed, 2)

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

    h = gray.shape[0]
    if h < 64:
        upscaled = upscale_image(gray, 4)
    elif h < 128:
        upscaled = upscale_image(gray, 2)
    else:
        upscaled = gray
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
    """再OCRの共通処理を実行する。"""
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
