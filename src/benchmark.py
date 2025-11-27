"""ベンチマーク機能モジュール"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.ocr_engines import execute_ocr, execute_ocr_paddleocr
from src.types import OCRElement


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
