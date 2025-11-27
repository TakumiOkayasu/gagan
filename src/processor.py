"""画像処理モジュール"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.benchmark import export_benchmark_json, print_benchmark_results, run_benchmark
from src.config import DEFAULT_WORKERS
from src.ocr_engines import PaddleOCRManager, execute_ocr, execute_ocr_with_engine
from src.postprocessing import (
    convert_to_json,
    correct_japanese_text,
    correct_ui_text,
    filter_by_confidence,
    merge_ocr_results,
    retry_character_level_ocr,
    retry_low_confidence_ocr,
)
from src.preprocessing import (
    PREPROCESS_FUNCTIONS,
    PreprocessFunc,
    apply_auto_sharpening,
    preprocess_image_adaptive,
    preprocess_image_inverted,
    preprocess_image_light,
    preprocess_image_screenshot,
    to_grayscale,
)
from src.text_detection import detect_text_regions_east, upscale_with_super_resolution
from src.types import OCRElement, OCRResult


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

        # 自動シャープ化 (リサイズによる文字ボケ軽減)
        image_array = apply_auto_sharpening(image_array)

        # シャープ化画像の保存
        if args.save_sharpened:
            sharpened_path = image_path.parent / f"{image_path.name}.sharpened.png"
            cv2.imwrite(
                str(sharpened_path), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            )
            print(f"シャープ化画像を保存しました: {sharpened_path}")

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


def run_regions_only_mode(
    args: argparse.Namespace, valid_image_paths: list[Path]
) -> int:
    """領域検出のみモードを実行する。"""
    import json
    from datetime import datetime

    print("領域検出のみモードで実行中 (OCRスキップ)...")
    all_results = []
    error_count = 0

    for image_path in valid_image_paths:
        try:
            image = Image.open(image_path)
            resolution = (image.width, image.height)
            image_array = np.array(image)

            # 自動シャープ化
            image_array = apply_auto_sharpening(image_array)

            # シャープ化画像の保存
            if args.save_sharpened:
                sharpened_path = image_path.parent / f"{image_path.name}.sharpened.png"
                if len(image_array.shape) == 3:
                    cv2.imwrite(
                        str(sharpened_path),
                        cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR),
                    )
                else:
                    cv2.imwrite(str(sharpened_path), image_array)
                print(f"シャープ化画像を保存しました: {sharpened_path}")

            # BGR変換 (EAST用)
            if len(image_array.shape) == 2:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # EAST テキスト領域検出
            regions = detect_text_regions_east(img_bgr)

            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "source_image": str(image_path),
                "resolution": {"width": resolution[0], "height": resolution[1]},
                "mode": "regions_only",
                "regions": [
                    {
                        "id": i,
                        "bbox": {
                            "x": r["x"],
                            "y": r["y"],
                            "width": r["width"],
                            "height": r["height"],
                        },
                        "confidence": r["confidence"],
                    }
                    for i, r in enumerate(regions)
                ],
                "total_regions": len(regions),
            }

            all_results.append(result)
            print(f"{image_path}: {len(regions)}領域検出")

        except Exception as e:
            print(f"エラー: {image_path}: {e}", file=sys.stderr)
            error_count += 1

    # 出力ファイル名の決定
    if args.output:
        output_path = Path(args.output)
    else:
        # 単一ファイル: {元ファイルパス}.regions.json
        # 複数ファイル: {画像ディレクトリ}/regions_{タイムスタンプ}.json
        if len(valid_image_paths) == 1:
            img_path = valid_image_paths[0]
            output_path = img_path.parent / f"{img_path.name}.regions.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = valid_image_paths[0].parent
            output_path = output_dir / f"regions_{timestamp}.json"

    # 単一ファイルでも配列で統一 (Claude連携で扱いやすい)
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_images": len(all_results),
        "images": all_results,
    }
    output_json = json.dumps(output_data, ensure_ascii=False, indent=2)

    output_path.write_text(output_json, encoding="utf-8")
    print(f"結果を保存しました: {output_path}")

    return error_count


def run_benchmark_mode(
    args: argparse.Namespace,
    valid_image_paths: list[Path],
    tessdata_dir: Optional[str],
) -> int:
    """ベンチマークモードを実行する。"""
    print("ベンチマークモードで実行中...")
    error_count = 0

    for image_path in valid_image_paths:
        print(f"\n処理中: {image_path}")
        try:
            image = Image.open(image_path)
            resolution = (image.width, image.height)
            image_array = np.array(image)

            # 自動シャープ化 (リサイズによる文字ボケ軽減)
            image_array = apply_auto_sharpening(image_array)

            # 前処理
            if args.screenshot:
                processed = preprocess_image_screenshot(
                    image_array, args.detect_rotation
                )
            elif args.light:
                processed = preprocess_image_light(image_array, args.detect_rotation)
            else:
                processed = preprocess_image_adaptive(image_array, args.detect_rotation)

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
            output_path = image_path.with_suffix(image_path.suffix + ".benchmark.json")
            export_benchmark_json(results, image_path.name, resolution, output_path)

        except Exception as e:
            print(f"エラー ({image_path}): {e}", file=sys.stderr)
            error_count += 1

    return error_count


def run_ocr_mode(
    args: argparse.Namespace,
    valid_image_paths: list[Path],
    tessdata_dir: Optional[str],
    parallel: bool,
    workers: int,
) -> tuple[int, int]:
    """OCRモードを実行する。"""
    success_count = 0
    error_count = 0

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

    if len(valid_image_paths) > 1:
        print(f"\n処理完了: 成功 {success_count}件, エラー {error_count}件")

    # PaddleOCRインスタンスを解放 (メモリ節約)
    PaddleOCRManager.release()

    return success_count, error_count
