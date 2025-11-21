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
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image


def detect_rotation(image: np.ndarray) -> float:
    """
    画像の回転角度を検出する。

    Args:
        image: グレースケール画像

    Returns:
        回転角度(度)
    """
    # エッジ検出
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # ハフ変換で直線を検出
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0.0

    # 検出された直線の角度を集計
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        # -45度から45度の範囲に正規化
        if angle < -45:
            angle += 180
        elif angle > 45:
            angle -= 180
        angles.append(angle)

    # 中央値を使用(外れ値に強い)
    if angles:
        median_angle = np.median(angles)
        # 小さな角度は無視(誤検出対策)
        if abs(median_angle) > 0.5:
            return median_angle

    return 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    画像を回転させる。

    Args:
        image: 入力画像
        angle: 回転角度(度)

    Returns:
        回転後の画像
    """
    if abs(angle) < 0.1:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 回転行列を作成
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 回転後の画像サイズを計算
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 回転行列の平行移動成分を調整
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # 回転実行(白背景で埋める)
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)

    return rotated


def upscale_if_needed(image: np.ndarray, min_height: int = 1000) -> tuple[np.ndarray, float]:
    """
    画像が小さい場合にアップスケールする。

    Args:
        image: 入力画像
        min_height: 最小高さ(ピクセル)

    Returns:
        (アップスケール後の画像, 拡大率)
    """
    h, w = image.shape[:2]

    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled, scale

    return image, 1.0


def preprocess_image_light(image: Image.Image, detect_rotation_flag: bool = False) -> np.ndarray:
    """
    軽量前処理を実行する。
    スクリーンショットなどクリアな画像に最適。二値化を行わない。

    Args:
        image: PIL Image形式の入力画像
        detect_rotation_flag: 回転検出を行うかどうか

    Returns:
        前処理済みのOpenCV形式画像(numpy配列)
    """
    # PIL ImageをOpenCV形式に変換
    img_array = np.array(image)

    # 1. グレースケール変換
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # 2. 回転検出と補正(オプション)
    if detect_rotation_flag:
        angle = detect_rotation(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, -angle)

    # 3. 軽いノイズ除去のみ(エッジを保持)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)

    return denoised


def preprocess_image_adaptive(image: Image.Image, detect_rotation_flag: bool = False) -> np.ndarray:
    """
    適応的閾値処理を使用した画像前処理を実行する。

    Args:
        image: PIL Image形式の入力画像
        detect_rotation_flag: 回転検出を行うかどうか

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

    # 2. 回転検出と補正(オプション)
    if detect_rotation_flag:
        angle = detect_rotation(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, -angle)

    # 3. アップスケール(小さな画像の場合)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    # 4. 明るさの正規化(ヒストグラム均等化)
    normalized = cv2.equalizeHist(gray)

    # 5. ガンマ補正(明るさ調整) - 色が薄い画像を強調
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(normalized, gamma_table)

    # 6. CLAHE(コントラスト強化) - より積極的なパラメータ
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gamma_corrected)

    # 7. シャープネス強化(アンシャープマスク) - より強力に
    gaussian = cv2.GaussianBlur(contrasted, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrasted, 2.0, gaussian, -1.0, 0)

    # 8. ノイズ除去(バイラテラルフィルタ) - エッジを保持しながらノイズ除去
    denoised = cv2.bilateralFilter(sharpened, 5, 75, 75)

    # 9. 適応的閾値処理による二値化 - より積極的なパラメータ
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        3
    )

    # 10. モルフォロジー処理(オープニング: 小さなノイズのみ除去)
    # クロージングは太文字を潰す可能性があるため、オープニングを使用
    kernel = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return morphed


def preprocess_image_otsu(image: Image.Image, detect_rotation_flag: bool = False) -> np.ndarray:
    """
    Otsu二値化を使用した画像前処理を実行する。
    薄いグレーの文字認識に有効。

    Args:
        image: PIL Image形式の入力画像
        detect_rotation_flag: 回転検出を行うかどうか

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

    # 2. 回転検出と補正(オプション)
    if detect_rotation_flag:
        angle = detect_rotation(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, -angle)

    # 3. アップスケール(小さな画像の場合)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    # 4. ガンマ補正(明るさ調整) - 薄い文字を強調
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, gamma_table)

    # 5. ノイズ除去(ガウシアンブラー)
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

    # 6. Otsu二値化 - 自動で最適な閾値を決定
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. モルフォロジー処理(オープニング: 小さなノイズのみ除去)
    # クロージングは太文字を潰す可能性があるため、オープニングを使用
    kernel = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return morphed


def preprocess_image_inverted(image: Image.Image, detect_rotation_flag: bool = False) -> np.ndarray:
    """
    白抜き文字(反転テキスト)用の前処理を実行する。
    暗い背景に白い文字がある場合に有効。

    Args:
        image: PIL Image形式の入力画像
        detect_rotation_flag: 回転検出を行うかどうか

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

    # 2. 回転検出と補正(オプション)
    if detect_rotation_flag:
        angle = detect_rotation(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, -angle)

    # 3. アップスケール(小さな画像の場合)
    gray, _ = upscale_if_needed(gray, min_height=1000)

    # 4. 反転(白抜き文字を黒文字に変換)
    inverted = cv2.bitwise_not(gray)

    # 5. ガンマ補正(明るさ調整)
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(inverted, gamma_table)

    # 6. ノイズ除去
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

    # 7. Otsu二値化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. モルフォロジー処理
    kernel = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return morphed


def execute_ocr(
    image: np.ndarray,
    lang: str = "jpn+eng",
    psm: int = 3
) -> Dict[str, Any]:
    """
    Tesseract OCRを実行し、テキストと座標情報を取得する。

    Args:
        image: 前処理済みのOpenCV形式画像
        lang: OCR言語設定(デフォルト: "jpn+eng")
        psm: Page Segmentation Mode(デフォルト: 3 = 完全に自動)

    Returns:
        OCR結果を含む辞書
    """
    # Tesseract設定
    # PSM: Page Segmentation Mode
    #   3 = Fully automatic page segmentation (デフォルト、画面テストに最適)
    #   6 = Assume a single uniform block of text (単一ブロック)
    #  11 = Sparse text (疎なテキスト)
    # OEM: OCR Engine Mode
    #   3 = Default, based on what is available (最新のLSTMエンジンを使用)
    config = f"--oem 3 --psm {psm}"

    # OCR実行(詳細データを取得)
    ocr_data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
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
        "--light",
        action="store_true",
        help="軽量モード(二値化なし、スクリーンショット向け)"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="高精度モード(複数の二値化手法を併用、処理時間約3倍)"
    )
    parser.add_argument(
        "--detect-rotation",
        action="store_true",
        help="回転検出と補正を有効化(斜めの画像に有効)"
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        help="白抜き文字モード(暗い背景に白文字がある場合)"
    )
    parser.add_argument(
        "--keep-debug-images",
        action="store_true",
        help="デバッグ画像を削除せず保持する(--debugと併用)"
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

        elif args.light:
            # 軽量モード: 二値化なし、スクリーンショット向け
            processed_image = preprocess_image_light(image, args.detect_rotation)

            # デバッグモード: 前処理後の画像を保存
            if args.debug:
                debug_image_path = image_path.with_suffix(".light.png")
                cv2.imwrite(str(debug_image_path), processed_image)
                print(f"前処理済み画像を保存しました: {debug_image_path}")

            # OCR実行
            ocr_result = execute_ocr(processed_image, args.lang)

        elif args.aggressive:
            # 高精度モード: 複数の二値化手法を併用
            print("高精度モードで処理中...")

            debug_paths = []

            # 適応的閾値処理
            processed_adaptive = preprocess_image_adaptive(image, args.detect_rotation)
            if args.debug:
                debug_adaptive_path = image_path.with_suffix(".adaptive.png")
                cv2.imwrite(str(debug_adaptive_path), processed_adaptive)
                debug_paths.append(debug_adaptive_path)
                print(f"適応的閾値処理済み画像を保存しました: {debug_adaptive_path}")

            ocr_result_adaptive = execute_ocr(processed_adaptive, args.lang)
            print(f"適応的閾値処理: {ocr_result_adaptive['total_elements']}要素")

            # Otsu二値化
            processed_otsu = preprocess_image_otsu(image, args.detect_rotation)
            if args.debug:
                debug_otsu_path = image_path.with_suffix(".otsu.png")
                cv2.imwrite(str(debug_otsu_path), processed_otsu)
                debug_paths.append(debug_otsu_path)
                print(f"Otsu二値化済み画像を保存しました: {debug_otsu_path}")

            ocr_result_otsu = execute_ocr(processed_otsu, args.lang)
            print(f"Otsu二値化: {ocr_result_otsu['total_elements']}要素")

            # 結果をマージ
            ocr_result = merge_ocr_results(ocr_result_adaptive, ocr_result_otsu)

            # 白抜き文字処理も追加
            processed_inverted = preprocess_image_inverted(image, args.detect_rotation)
            if args.debug:
                debug_inverted_path = image_path.with_suffix(".inverted.png")
                cv2.imwrite(str(debug_inverted_path), processed_inverted)
                debug_paths.append(debug_inverted_path)
                print(f"反転処理済み画像を保存しました: {debug_inverted_path}")

            ocr_result_inverted = execute_ocr(processed_inverted, args.lang)
            print(f"反転処理: {ocr_result_inverted['total_elements']}要素")

            # 反転処理結果もマージ
            ocr_result = merge_ocr_results(ocr_result, ocr_result_inverted)
            print(f"マージ後: {ocr_result['total_elements']}要素")

            # デバッグモード用の画像削除リストに追加
            if args.debug:
                debug_image_path = debug_paths

        elif args.inverted:
            # 白抜き文字モード
            processed_image = preprocess_image_inverted(image, args.detect_rotation)

            # デバッグモード: 前処理後の画像を保存
            if args.debug:
                debug_image_path = image_path.with_suffix(".inverted.png")
                cv2.imwrite(str(debug_image_path), processed_image)
                print(f"前処理済み画像を保存しました: {debug_image_path}")

            # OCR実行
            ocr_result = execute_ocr(processed_image, args.lang)

        else:
            # 通常モード: 適応的閾値処理のみ
            processed_image = preprocess_image_adaptive(image, args.detect_rotation)

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
        # デバッグモードで保存した前処理済み画像を削除(--keep-debug-imagesが指定されていない場合)
        if debug_image_path and not args.keep_debug_images:
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
