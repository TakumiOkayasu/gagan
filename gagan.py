#!/usr/bin/env python3
"""
GAGAN - 画面テスト用OCRツール

Tesseract OCR + 画像前処理を使用した、画面テストのためのOCRツール。
スクリーンショットからテキストを抽出し、座標情報とともにJSON形式で出力する。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract
from PIL import Image


def preprocess_image_adaptive(image: Image.Image) -> np.ndarray:
    """
    適応的閾値処理を使用した画像前処理を実行する。

    Args:
        image: PIL Image形式の入力画像

    Returns:
        前処理済みのOpenCV形式画像(numpy配列)
    """
    # PIL ImageをOpenCV形式に変換
    img_array = np.array(image)

    # 1. グレースケール変換
    if len(img_array.shape) == 3:
        # カラー画像の場合: BGR変換してからグレースケール化
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        # 既にグレースケールの場合: そのまま使用
        gray = img_array

    # 2. 明るさの正規化(ヒストグラム均等化)
    normalized = cv2.equalizeHist(gray)

    # 3. ガンマ補正(明るさ調整) - 色が薄い画像を強調
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(normalized, gamma_table)

    # 4. CLAHE(コントラスト強化) - より積極的なパラメータ
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gamma_corrected)

    # 5. シャープネス強化(アンシャープマスク) - より強力に
    gaussian = cv2.GaussianBlur(contrasted, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrasted, 2.0, gaussian, -1.0, 0)

    # 6. ノイズ除去(バイラテラルフィルタ) - エッジを保持しながらノイズ除去
    denoised = cv2.bilateralFilter(sharpened, 5, 75, 75)

    # 7. 適応的閾値処理による二値化 - より積極的なパラメータ
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # 8. モルフォロジー処理(ノイズ除去とテキスト強調)
    kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return morphed


def preprocess_image_otsu(image: Image.Image) -> np.ndarray:
    """
    Otsu二値化を使用した画像前処理を実行する。
    薄いグレーの文字認識に有効。

    Args:
        image: PIL Image形式の入力画像

    Returns:
        前処理済みのOpenCV形式画像(numpy配列)
    """
    # PIL ImageをOpenCV形式に変換
    img_array = np.array(image)

    # 1. グレースケール変換
    if len(img_array.shape) == 3:
        # カラー画像の場合: BGR変換してからグレースケール化
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        # 既にグレースケールの場合: そのまま使用
        gray = img_array

    # 2. ガンマ補正(明るさ調整) - 薄い文字を強調
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, gamma_table)

    # 3. ノイズ除去(ガウシアンブラー)
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

    # 4. Otsu二値化 - 自動で最適な閾値を決定
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. モルフォロジー処理(ノイズ除去)
    kernel = np.ones((2, 2), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return morphed


def execute_ocr(
    image: np.ndarray,
    lang: str = "jpn+eng"
) -> Dict[str, Any]:
    """
    Tesseract OCRを実行し、テキストと座標情報を取得する。

    Args:
        image: 前処理済みのOpenCV形式画像
        lang: OCR言語設定(デフォルト: "jpn+eng")

    Returns:
        OCR結果を含む辞書
    """
    # OCR実行(詳細データを取得)
    ocr_data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    # 結果を構造化
    elements = []
    element_id = 0

    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])

        # 空のテキストまたは信頼度が低い(-1)ものは除外
        if not text or conf == -1:
            continue

        elements.append({
            "id": element_id,
            "text": text,
            "bbox": {
                "x": int(ocr_data["left"][i]),
                "y": int(ocr_data["top"][i]),
                "width": int(ocr_data["width"][i]),
                "height": int(ocr_data["height"][i])
            },
            "confidence": round(conf / 100.0, 2)
        })
        element_id += 1

    return {
        "elements": elements,
        "total_elements": element_id
    }


def calculate_iou(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
    """
    2つのbounding boxのIoU (Intersection over Union)を計算する。

    Args:
        bbox1: 1つ目のbounding box {"x", "y", "width", "height"}
        bbox2: 2つ目のbounding box {"x", "y", "width", "height"}

    Returns:
        IoU値(0.0-1.0)
    """
    # 各bboxの右下座標を計算
    x1_min, y1_min = bbox1["x"], bbox1["y"]
    x1_max, y1_max = x1_min + bbox1["width"], y1_min + bbox1["height"]
    x2_min, y2_min = bbox2["x"], bbox2["y"]
    x2_max, y2_max = x2_min + bbox2["width"], y2_min + bbox2["height"]

    # 交差領域を計算
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 交差領域の面積
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 各bboxの面積
    area1 = bbox1["width"] * bbox1["height"]
    area2 = bbox2["width"] * bbox2["height"]

    # IoU = 交差面積 / 結合面積
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_ocr_results(
    result1: Dict[str, Any],
    result2: Dict[str, Any],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    2つのOCR結果をマージする。
    重複する要素は信頼度の高い方を採用する。

    Args:
        result1: 1つ目のOCR結果
        result2: 2つ目のOCR結果
        iou_threshold: 重複判定のIoU閾値(デフォルト: 0.5)

    Returns:
        マージされたOCR結果
    """
    merged_elements = []
    result2_used = [False] * len(result2["elements"])

    # result1の各要素について処理
    for elem1 in result1["elements"]:
        best_match_idx = -1
        best_iou = 0.0

        # result2から重複する要素を探す
        for i, elem2 in enumerate(result2["elements"]):
            if result2_used[i]:
                continue

            iou = calculate_iou(elem1["bbox"], elem2["bbox"])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match_idx = i

        # 重複する要素が見つかった場合、信頼度の高い方を採用
        if best_match_idx >= 0:
            elem2 = result2["elements"][best_match_idx]
            result2_used[best_match_idx] = True

            if elem1["confidence"] >= elem2["confidence"]:
                merged_elements.append(elem1)
            else:
                merged_elements.append(elem2)
        else:
            # 重複なし: result1の要素を追加
            merged_elements.append(elem1)

    # result2の未使用要素を追加
    for i, elem2 in enumerate(result2["elements"]):
        if not result2_used[i]:
            merged_elements.append(elem2)

    # ID振り直し
    for i, elem in enumerate(merged_elements):
        elem["id"] = i

    return {
        "elements": merged_elements,
        "total_elements": len(merged_elements)
    }


def convert_to_json(
    ocr_result: Dict[str, Any],
    source_image: str,
    resolution: tuple[int, int]
) -> str:
    """
    OCR結果をJSON形式に変換する。

    Args:
        ocr_result: execute_ocr関数の戻り値
        source_image: 元画像ファイル名
        resolution: 画像解像度(width, height)

    Returns:
        JSON形式の文字列
    """
    output_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_image": source_image,
        "resolution": {
            "width": resolution[0],
            "height": resolution[1]
        },
        "elements": ocr_result["elements"]
    }

    return json.dumps(output_data, ensure_ascii=False, indent=2)


def main() -> int:
    """
    CLIエントリーポイント。

    Returns:
        終了コード(0: 成功, 1: エラー)
    """
    parser = argparse.ArgumentParser(
        description="GAGAN - 画面テスト用OCRツール"
    )
    parser.add_argument(
        "image",
        help="OCRを実行する画像ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output",
        help="出力JSONファイル名(デフォルト: <入力ファイル名>.ocr.json)"
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="画像前処理をスキップする"
    )
    parser.add_argument(
        "--lang",
        default="jpn+eng",
        help="OCR言語設定(デフォルト: jpn+eng)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード(前処理後の画像を保存)"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="高精度モード(複数の二値化手法を併用、処理時間約2倍)"
    )

    args = parser.parse_args()

    debug_image_path = None  # デバッグ用に保存した画像のパス

    try:
        # 画像ファイルの読み込み
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"エラー: 画像ファイルが見つかりません: {args.image}", file=sys.stderr)
            return 1

        image = Image.open(image_path)
        resolution = (image.width, image.height)

        # 画像前処理とOCR実行
        if args.no_preprocessing:
            # 前処理なし: PIL ImageをOpenCV形式に変換(グレースケールのみ)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # カラー画像の場合: グレースケール化
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                processed_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            else:
                # 既にグレースケールの場合: そのまま使用
                processed_image = img_array

            # デバッグモード: 前処理後の画像を保存
            if args.debug:
                debug_image_path = image_path.with_suffix(".preprocessed.png")
                cv2.imwrite(str(debug_image_path), processed_image)
                print(f"前処理済み画像を保存しました: {debug_image_path}")

            # OCR実行
            ocr_result = execute_ocr(processed_image, args.lang)

        elif args.aggressive:
            # 高精度モード: 複数の二値化手法を併用
            print("高精度モードで処理中...")

            # 適応的閾値処理
            processed_adaptive = preprocess_image_adaptive(image)
            if args.debug:
                debug_adaptive_path = image_path.with_suffix(".adaptive.png")
                cv2.imwrite(str(debug_adaptive_path), processed_adaptive)
                print(f"適応的閾値処理済み画像を保存しました: {debug_adaptive_path}")

            ocr_result_adaptive = execute_ocr(processed_adaptive, args.lang)
            print(f"適応的閾値処理: {ocr_result_adaptive['total_elements']}要素")

            # Otsu二値化
            processed_otsu = preprocess_image_otsu(image)
            if args.debug:
                debug_otsu_path = image_path.with_suffix(".otsu.png")
                cv2.imwrite(str(debug_otsu_path), processed_otsu)
                print(f"Otsu二値化済み画像を保存しました: {debug_otsu_path}")

            ocr_result_otsu = execute_ocr(processed_otsu, args.lang)
            print(f"Otsu二値化: {ocr_result_otsu['total_elements']}要素")

            # 結果をマージ
            ocr_result = merge_ocr_results(ocr_result_adaptive, ocr_result_otsu)
            print(f"マージ後: {ocr_result['total_elements']}要素")

            # デバッグモード用の画像削除リストに追加
            if args.debug:
                debug_image_path = [debug_adaptive_path, debug_otsu_path]

        else:
            # 通常モード: 適応的閾値処理のみ
            processed_image = preprocess_image_adaptive(image)

            # デバッグモード: 前処理後の画像を保存
            if args.debug:
                debug_image_path = image_path.with_suffix(".preprocessed.png")
                cv2.imwrite(str(debug_image_path), processed_image)
                print(f"前処理済み画像を保存しました: {debug_image_path}")

            # OCR実行
            ocr_result = execute_ocr(processed_image, args.lang)

        # JSON変換
        json_output = convert_to_json(
            ocr_result,
            image_path.name,
            resolution
        )

        # 出力ファイル名の決定
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = image_path.with_suffix(image_path.suffix + ".ocr.json")

        # JSON出力
        output_path.write_text(json_output, encoding="utf-8")
        print(f"OCR結果を保存しました: {output_path}")
        print(f"認識されたテキスト要素数: {ocr_result['total_elements']}")

        return 0

    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        return 1

    finally:
        # デバッグモードで保存した前処理済み画像を削除
        if debug_image_path:
            if isinstance(debug_image_path, list):
                # aggressiveモード: 複数の画像を削除
                for path in debug_image_path:
                    if path.exists():
                        path.unlink()
                        print(f"前処理済み画像を削除しました: {path}")
            elif debug_image_path.exists():
                # 通常モード: 1つの画像を削除
                debug_image_path.unlink()
                print(f"前処理済み画像を削除しました: {debug_image_path}")


if __name__ == "__main__":
    sys.exit(main())
