"""OCRエンジンモジュール"""

import gc
import sys
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

from src.preprocessing import select_optimal_psm
from src.types import OCRElement, OCRResult


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

    config_parts = ["--oem 1", f"--psm {psm}", "-c tessedit_do_invert=0"]
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
