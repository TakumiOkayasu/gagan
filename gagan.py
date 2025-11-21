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


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    OCR精度向上のための画像前処理を実行する。

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

        # 画像前処理
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
        else:
            processed_image = preprocess_image(image)

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
        if debug_image_path and debug_image_path.exists():
            debug_image_path.unlink()
            print(f"前処理済み画像を削除しました: {debug_image_path}")


if __name__ == "__main__":
    sys.exit(main())
