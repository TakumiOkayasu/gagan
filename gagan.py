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
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return rotated


def detect_theme(image: np.ndarray) -> str:
    """
    画像の平均輝度からテーマ(ダーク/ライト)を判定する。
    スクリーンショットのダークモード検出に使用。

    Args:
        image: グレースケール画像

    Returns:
        "dark" または "light"
    """
    mean_brightness = np.mean(image)
    return "dark" if mean_brightness < 128 else "light"


def calculate_text_density(image: np.ndarray) -> float:
    """
    画像内のテキスト密度を計算する。

    Args:
        image: グレースケール画像

    Returns:
        テキスト密度 (0.0-1.0)
    """
    # 適応的閾値処理で二値化
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 非ゼロピクセルの割合を計算
    non_zero = cv2.countNonZero(binary)
    total = image.shape[0] * image.shape[1]

    return non_zero / total if total > 0 else 0.0


def select_optimal_psm(image: np.ndarray) -> int:
    """
    画像特性に基づいて最適なPSM (Page Segmentation Mode) を選択する。

    Args:
        image: グレースケール画像

    Returns:
        最適なPSM値
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h if h > 0 else 1.0

    # テキスト密度を計算
    text_density = calculate_text_density(image)

    # PSM選択ロジック
    # PSM 3: Fully automatic page segmentation (デフォルト)
    # PSM 4: Single column of text
    # PSM 6: Single uniform block of text (UI向け)
    # PSM 7: Single text line
    # PSM 11: Sparse text
    # PSM 12: Sparse text with OSD

    if text_density < 0.05:
        # 非常に疎なテキスト
        return 11
    elif text_density < 0.1:
        # 疎なテキスト (アイコン付きUI等)
        return 11
    elif aspect_ratio > 5:
        # 非常に横長 (単一行)
        return 7
    elif aspect_ratio < 0.3:
        # 非常に縦長 (単一カラム)
        return 4
    else:
        # 通常のUI/スクリーンショット
        return 6


def upscale_if_needed(
    image: np.ndarray, min_height: int = 1000
) -> tuple[np.ndarray, float]:
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


def upscale_with_super_resolution(
    image: np.ndarray, scale: int = 4, model_name: str = "espcn"
) -> np.ndarray:
    """
    超解像 (Super Resolution) によるアップスケール。
    opencv-contrib-python の dnn_superres モジュールを使用。

    Args:
        image: 入力画像
        scale: 拡大倍率 (2, 3, 4)
        model_name: モデル名 (espcn, fsrcnn, lapsrn)

    Returns:
        アップスケール後の画像
    """
    try:
        # dnn_superresモジュールのインポート
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # モデルファイルのパス
        model_paths = [
            Path(__file__).parent / "models" / f"{model_name.upper()}_x{scale}.pb",
            Path.home() / ".gagan" / "models" / f"{model_name.upper()}_x{scale}.pb",
            Path(f"/usr/share/gagan/models/{model_name.upper()}_x{scale}.pb"),
        ]

        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = str(path)
                break

        if model_path is None:
            print(
                "警告: 超解像モデルが見つかりません。INTER_CUBICにフォールバック",
                file=sys.stderr,
            )
            h, w = image.shape[:2]
            return cv2.resize(
                image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
            )

        sr.readModel(model_path)
        sr.setModel(model_name.lower(), scale)
        return sr.upsample(image)

    except AttributeError:
        print(
            "警告: opencv-contrib-pythonが必要です。INTER_CUBICにフォールバック",
            file=sys.stderr,
        )
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(
            f"警告: 超解像処理に失敗しました: {e}。INTER_CUBICにフォールバック",
            file=sys.stderr,
        )
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def detect_text_regions_east(
    image: np.ndarray, confidence_threshold: float = 0.5, nms_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    EASTテキスト検出器でテキスト領域を検出する。

    Args:
        image: 入力画像 (BGR)
        confidence_threshold: 信頼度閾値
        nms_threshold: Non-Maximum Suppression閾値

    Returns:
        検出されたテキスト領域のリスト
    """
    try:
        # モデルファイルのパス
        model_paths = [
            Path(__file__).parent / "models" / "frozen_east_text_detection.pb",
            Path.home() / ".gagan" / "models" / "frozen_east_text_detection.pb",
            Path("/usr/share/gagan/models/frozen_east_text_detection.pb"),
        ]

        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = str(path)
                break

        if model_path is None:
            print(
                "警告: EASTモデルが見つかりません。テキスト検出をスキップ",
                file=sys.stderr,
            )
            return []

        # 画像サイズを32の倍数に調整
        orig_h, orig_w = image.shape[:2]
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32

        if new_w == 0 or new_h == 0:
            return []

        ratio_w = orig_w / new_w
        ratio_h = orig_h / new_h

        # 画像をリサイズ
        resized = cv2.resize(image, (new_w, new_h))

        # ネットワークの読み込み
        net = cv2.dnn.readNet(model_path)

        # blobの作成
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), True, False
        )

        net.setInput(blob)

        # 出力層の取得
        output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        scores, geometry = net.forward(output_layers)

        # 検出結果のデコード
        boxes = []
        confidences = []

        num_rows, num_cols = scores.shape[2:4]
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                if scores_data[x] < confidence_threshold:
                    continue

                offset_x = x * 4.0
                offset_y = y * 4.0

                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                end_x = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
                end_y = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                # 元画像の座標に変換
                start_x = int(start_x * ratio_w)
                start_y = int(start_y * ratio_h)
                end_x = int(end_x * ratio_w)
                end_y = int(end_y * ratio_h)

                boxes.append([start_x, start_y, end_x, end_y])
                confidences.append(float(scores_data[x]))

        # Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
                confidences,
                confidence_threshold,
                nms_threshold,
            )

            result = []
            for i in indices.flatten() if len(indices) > 0 else []:
                box = boxes[i]
                result.append(
                    {
                        "x": max(0, box[0]),
                        "y": max(0, box[1]),
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                        "confidence": confidences[i],
                    }
                )
            return result

        return []

    except Exception as e:
        print(f"警告: EASTテキスト検出に失敗しました: {e}", file=sys.stderr)
        return []


def upscale_small_ui_elements(
    image: np.ndarray, min_text_height: int = 32
) -> tuple[np.ndarray, float]:
    """
    小さいUI要素(ボタンラベル等)の認識精度向上のためスケーリングする。
    Tesseractは文字高さ32px以上で精度が向上する。

    Args:
        image: 入力画像
        min_text_height: 目標となる最小文字高さ(ピクセル)

    Returns:
        (スケール後の画像, 拡大率)
    """
    h, w = image.shape[:2]

    # 画像が小さすぎる場合のみスケール
    # 一般的なUI文字サイズ(12-16px)を32px以上にする
    if h < 100:
        scale = max(2.0, min_text_height / 12)
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled, scale

    return image, 1.0


def preprocess_image_screenshot(
    image: Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """
    スクリーンショット専用の前処理を実行する。
    ダークモード自動検出、UI要素のスケーリングを含む。

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

    # 2. 回転検出と補正(オプション) - スクショでは通常不要
    if detect_rotation_flag:
        angle = detect_rotation(gray)
        if abs(angle) > 0.5:
            gray = rotate_image(gray, -angle)

    # 3. ダークモード検出と自動反転
    theme = detect_theme(gray)
    if theme == "dark":
        gray = cv2.bitwise_not(gray)

    # 4. 小さいUI要素のスケーリング
    gray, _ = upscale_small_ui_elements(gray)

    # 5. 軽いノイズ除去のみ(エッジを保持)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)

    return denoised


def preprocess_image_light(
    image: Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
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


def preprocess_image_adaptive(
    image: Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
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
    gamma_table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
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
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )

    # 10. モルフォロジー処理(オープニング: 小さなノイズのみ除去)
    # クロージングは太文字を潰す可能性があるため、オープニングを使用
    kernel = np.ones((1, 1), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return morphed


def preprocess_image_otsu(
    image: Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
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
    gamma_table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
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


def preprocess_image_inverted(
    image: Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
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
    gamma_table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
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
    psm: Optional[int] = None,
    preserve_spaces: bool = True,
    tessdata_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tesseract OCRを実行し、テキストと座標情報を取得する。

    Args:
        image: 前処理済みのOpenCV形式画像
        lang: OCR言語設定(デフォルト: "jpn+eng")
        psm: Page Segmentation Mode(None=自動選択)
        preserve_spaces: 単語間スペースを保持するか(デフォルト: True)
        tessdata_dir: tessdataディレクトリ(tessdata_best用)

    Returns:
        OCR結果を含む辞書
    """
    # PSM自動選択
    if psm is None:
        psm = select_optimal_psm(image)

    # Tesseract設定
    # PSM: Page Segmentation Mode
    #   3 = Fully automatic page segmentation
    #   4 = Single column of text
    #   6 = Single uniform block of text (UI向け)
    #   7 = Single text line
    #  11 = Sparse text (疎なテキスト)
    # OEM: OCR Engine Mode
    #   3 = Default, based on what is available (最新のLSTMエンジンを使用)
    config_parts = ["--oem 3", f"--psm {psm}"]

    if preserve_spaces:
        config_parts.append("-c preserve_interword_spaces=1")

    if tessdata_dir:
        config_parts.append(f"--tessdata-dir {tessdata_dir}")

    config = " ".join(config_parts)

    # OCR実行(詳細データを取得)
    ocr_data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
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

        elements.append(
            {
                "id": element_id,
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
        element_id += 1

    return {"elements": elements, "total_elements": element_id}


def retry_low_confidence_ocr(
    original_image: Image.Image,
    ocr_result: Dict[str, Any],
    confidence_threshold: float = 0.8,
    lang: str = "jpn+eng",
    tessdata_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    信頼度が低い要素に対して、異なる前処理で再OCRを実行する。

    Args:
        original_image: 元のPIL Image
        ocr_result: 初回OCRの結果
        confidence_threshold: 再OCRを実行する信頼度閾値
        lang: OCR言語設定
        tessdata_dir: tessdataディレクトリ

    Returns:
        改善されたOCR結果
    """
    low_confidence_elements = [
        elem
        for elem in ocr_result["elements"]
        if elem["confidence"] < confidence_threshold
    ]

    if not low_confidence_elements:
        return ocr_result

    print(f"低信頼度要素 {len(low_confidence_elements)}件に対して再OCR実行中...")

    # 異なる前処理手法のリスト
    preprocess_methods = [
        ("adaptive", preprocess_image_adaptive),
        ("otsu", preprocess_image_otsu),
        ("inverted", preprocess_image_inverted),
        ("light", preprocess_image_light),
    ]

    improved_elements = []

    for elem in ocr_result["elements"]:
        if elem["confidence"] >= confidence_threshold:
            improved_elements.append(elem)
            continue

        bbox = elem["bbox"]
        # マージンを追加して領域を切り出し
        margin = 10
        x1 = max(0, bbox["x"] - margin)
        y1 = max(0, bbox["y"] - margin)
        x2 = min(original_image.width, bbox["x"] + bbox["width"] + margin)
        y2 = min(original_image.height, bbox["y"] + bbox["height"] + margin)

        region_image = original_image.crop((x1, y1, x2, y2))

        best_result = elem
        best_confidence = elem["confidence"]

        # 各前処理手法で再OCRを試行
        for method_name, preprocess_func in preprocess_methods:
            try:
                processed = preprocess_func(region_image, detect_rotation_flag=False)

                # 小さい領域はスケールアップ
                h, w = processed.shape[:2]
                if h < 32:
                    scale = 32 / h
                    processed = cv2.resize(
                        processed,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )

                # PSM 7 (単一行) または PSM 8 (単一単語) で再OCR
                for psm in [7, 8, 13]:
                    region_result = execute_ocr(
                        processed, lang, psm=psm, tessdata_dir=tessdata_dir
                    )

                    for new_elem in region_result["elements"]:
                        if new_elem["confidence"] > best_confidence:
                            best_confidence = new_elem["confidence"]
                            best_result = {
                                "id": elem["id"],
                                "text": new_elem["text"],
                                "bbox": bbox,  # 元のbboxを保持
                                "confidence": new_elem["confidence"],
                            }
            except Exception:
                continue

        improved_elements.append(best_result)

        if best_confidence > elem["confidence"]:
            print(
                f"  改善: '{elem['text']}' ({elem['confidence']:.2f}) -> "
                f"'{best_result['text']}' ({best_confidence:.2f})"
            )

    return {"elements": improved_elements, "total_elements": len(improved_elements)}


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
    result1: Dict[str, Any], result2: Dict[str, Any], iou_threshold: float = 0.5
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

    return {"elements": merged_elements, "total_elements": len(merged_elements)}


def correct_ui_text(text: str) -> str:
    """
    UI要素でよく発生するOCR誤認識パターンを修正する。
    ボタンラベル、メニュー項目等に特化。

    Args:
        text: OCRで認識されたテキスト

    Returns:
        修正後のテキスト
    """
    # 英字UI要素の誤認識パターン (1/l/I, 0/O 等)
    ui_corrections = {
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

    result = text
    for wrong, correct in ui_corrections.items():
        if wrong in result:
            result = result.replace(wrong, correct)

    return result


def correct_japanese_text(text: str) -> str:
    """
    日本語OCRの誤認識パターンを修正する。
    カタカナ文脈で漢字として誤認識されやすい文字を補正。

    Args:
        text: OCRで認識されたテキスト

    Returns:
        修正後のテキスト
    """
    # 漢字 → カタカナの誤認識パターン
    # (カタカナ文脈でのみ修正)
    corrections = {
        "一": "ー",  # 漢数字の一 vs 長音符
        "口": "ロ",  # 漢字の口 vs カタカナのロ
        "力": "カ",  # 漢字の力 vs カタカナのカ
        "工": "エ",  # 漢字の工 vs カタカナのエ
        "夕": "タ",  # 漢字の夕 vs カタカナのタ
        "二": "ニ",  # 漢数字の二 vs カタカナのニ
        "八": "ハ",  # 漢数字の八 vs カタカナのハ
    }

    def is_katakana(char: str) -> bool:
        """文字がカタカナかどうか判定"""
        if not char:
            return False
        code = ord(char)
        # カタカナ: U+30A0-U+30FF、半角カタカナ: U+FF65-U+FF9F
        return (0x30A0 <= code <= 0x30FF) or (0xFF65 <= code <= 0xFF9F)

    chars = list(text)
    result = []

    for i, char in enumerate(chars):
        if char in corrections:
            prev_char = chars[i - 1] if i > 0 else ""
            next_char = chars[i + 1] if i < len(chars) - 1 else ""

            # 前後の文字がカタカナなら修正
            if is_katakana(prev_char) or is_katakana(next_char):
                result.append(corrections[char])
            else:
                result.append(char)
        else:
            result.append(char)

    return "".join(result)


def filter_by_confidence(
    elements: List[Dict[str, Any]], min_confidence: float = 0.3
) -> List[Dict[str, Any]]:
    """
    信頼度に基づいて要素をフィルタリングする。
    短いテキストには高い信頼度を要求。

    Args:
        elements: OCR結果の要素リスト
        min_confidence: 最小信頼度 (デフォルト: 0.3)

    Returns:
        フィルタリング後の要素リスト
    """
    filtered = []

    for elem in elements:
        conf = elem["confidence"]
        text = elem["text"]
        text_len = len(text)

        # 短いテキストは高い信頼度を要求
        if text_len <= 1:
            required_conf = min_confidence * 1.5
        elif text_len <= 2:
            required_conf = min_confidence * 1.2
        else:
            required_conf = min_confidence

        if conf >= required_conf:
            filtered.append(elem)
        # 長いテキストは多少信頼度が低くても許容
        elif text_len >= 5 and conf >= min_confidence * 0.8:
            filtered.append(elem)

    return filtered


def convert_to_json(
    ocr_result: Dict[str, Any], source_image: str, resolution: tuple[int, int]
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
        "resolution": {"width": resolution[0], "height": resolution[1]},
        "elements": ocr_result["elements"],
    }

    return json.dumps(output_data, ensure_ascii=False, indent=2)


def main() -> int:
    """
    CLIエントリーポイント。

    Returns:
        終了コード(0: 成功, 1: エラー)
    """
    parser = argparse.ArgumentParser(description="GAGAN - 画面テスト用OCRツール")
    parser.add_argument(
        "images", nargs="+", help="OCRを実行する画像ファイルのパス(複数指定可)"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="出力JSONファイル名(単一ファイル時のみ有効、デフォルト: <入力ファイル名>.ocr.json)",
    )
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="画像前処理をスキップする"
    )
    parser.add_argument(
        "--lang", default="jpn+eng", help="OCR言語設定(デフォルト: jpn+eng)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード(前処理後の画像を保存)"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="軽量モード(二値化なし、スクリーンショット向け)",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="スクリーンショット専用モード(ダークモード自動検出、UI要素スケーリング、UI誤認識補正)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="高精度モード(複数の二値化手法を併用、処理時間約3倍)",
    )
    parser.add_argument(
        "--detect-rotation",
        action="store_true",
        help="回転検出と補正を有効化(斜めの画像に有効)",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        help="白抜き文字モード(暗い背景に白文字がある場合)",
    )
    parser.add_argument(
        "--keep-debug-images",
        action="store_true",
        help="デバッグ画像を削除せず保持する(--debugと併用)",
    )

    # Phase 1: 精度向上オプション
    parser.add_argument(
        "--fast", action="store_true", help="高速モード(精度向上機能をOFF、従来互換)"
    )
    parser.add_argument(
        "--psm",
        type=str,
        default="auto",
        help="Page Segmentation Mode (auto/3/4/6/7/11等、デフォルト: auto)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="最小信頼度フィルタ (0.0-1.0、デフォルト: 0.3)",
    )
    parser.add_argument(
        "--retry-threshold",
        type=float,
        default=0.8,
        help="再OCRを実行する信頼度閾値 (0.0-1.0、デフォルト: 0.8)",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="低信頼度要素の再OCRを無効化",
    )
    parser.add_argument(
        "--no-japanese-correct", action="store_true", help="日本語誤認識修正を無効化"
    )

    # Phase 2: 高精度オプション
    parser.add_argument(
        "--best", action="store_true", help="tessdata_best (高精度訓練データ) を使用"
    )
    parser.add_argument("--tessdata-dir", type=str, help="tessdataディレクトリを指定")
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="形態素解析による後処理を有効化 (要: janome)",
    )

    # Phase 3: 高度なオプション
    parser.add_argument(
        "--super-resolution",
        action="store_true",
        help="超解像によるアップスケール (要: opencv-contrib-python)",
    )
    parser.add_argument(
        "--text-detection",
        action="store_true",
        help="EASTテキスト領域検出 (要: モデルファイル)",
    )

    # 最高精度モード
    parser.add_argument(
        "--max-accuracy",
        action="store_true",
        help="最高精度モード (--best --post-process --aggressive を有効化)",
    )

    args = parser.parse_args()

    # --max-accuracyの展開
    if args.max_accuracy:
        args.best = True
        args.post_process = True
        args.aggressive = True

    # --fastの処理
    if args.fast:
        args.min_confidence = 0.0
        args.no_japanese_correct = True
        if args.psm == "auto":
            args.psm = "3"  # 従来のデフォルト

    # 複数ファイル指定時は-oオプションを無視
    if len(args.images) > 1 and args.output:
        print("警告: 複数ファイル指定時は-oオプションは無視されます", file=sys.stderr)

    success_count = 0
    error_count = 0

    for image_file in args.images:
        debug_image_path = None  # デバッグ用に保存した画像のパス

        try:
            # 画像ファイルの読み込み
            image_path = Path(image_file)
            if not image_path.exists():
                print(
                    f"エラー: 画像ファイルが見つかりません: {image_file}",
                    file=sys.stderr,
                )
                error_count += 1
                continue

            image = Image.open(image_path)
            resolution = (image.width, image.height)

            # PSM設定
            psm = None if args.psm == "auto" else int(args.psm)

            # tessdata_dir設定
            tessdata_dir = args.tessdata_dir
            if args.best and not tessdata_dir:
                # tessdata_bestのデフォルトパス
                tessdata_best_paths = [
                    "/usr/share/tesseract-ocr/5/tessdata_best",
                    "/usr/share/tesseract-ocr/4.00/tessdata_best",
                    "/usr/local/share/tessdata_best",
                ]
                for path in tessdata_best_paths:
                    if Path(path).exists():
                        tessdata_dir = path
                        break
                if not tessdata_dir:
                    print(
                        "警告: tessdata_bestが見つかりません。標準tessdataを使用します",
                        file=sys.stderr,
                    )

            # 超解像による前処理 (Phase 3)
            if args.super_resolution:
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                upscaled = upscale_with_super_resolution(
                    img_bgr, scale=4, model_name="espcn"
                )
                image = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
                print(f"超解像適用: {resolution} -> ({image.width}, {image.height})")

            # テキスト領域検出モード (Phase 3)
            if args.text_detection:
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                text_regions = detect_text_regions_east(img_bgr)

                if text_regions:
                    print(f"テキスト領域検出: {len(text_regions)}領域")

                    # 各領域ごとにOCRを実行
                    all_elements = []
                    element_id = 0

                    for region in text_regions:
                        # 領域を切り出し
                        x, y = region["x"], region["y"]
                        w, h = region["width"], region["height"]

                        # マージンを追加
                        margin = 5
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(image.width, x + w + margin)
                        y2 = min(image.height, y + h + margin)

                        region_image = image.crop((x1, y1, x2, y2))

                        # 前処理
                        if args.light:
                            processed = preprocess_image_light(
                                region_image, args.detect_rotation
                            )
                        else:
                            processed = preprocess_image_adaptive(
                                region_image, args.detect_rotation
                            )

                        # OCR実行 (PSM 7: 単一行)
                        region_result = execute_ocr(
                            processed, args.lang, psm=7, tessdata_dir=tessdata_dir
                        )

                        # 座標を元画像の座標に変換
                        for elem in region_result["elements"]:
                            elem["id"] = element_id
                            elem["bbox"]["x"] += x1
                            elem["bbox"]["y"] += y1
                            all_elements.append(elem)
                            element_id += 1

                    ocr_result = {
                        "elements": all_elements,
                        "total_elements": len(all_elements),
                    }
                else:
                    print("テキスト領域が検出されませんでした。通常モードで処理")
                    # フォールバック: 通常処理
                    processed_image = preprocess_image_adaptive(
                        image, args.detect_rotation
                    )
                    ocr_result = execute_ocr(
                        processed_image, args.lang, psm=psm, tessdata_dir=tessdata_dir
                    )

            # 画像前処理とOCR実行
            elif args.no_preprocessing:
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
                ocr_result = execute_ocr(
                    processed_image, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )

            elif args.screenshot and args.aggressive:
                # スクリーンショット + 高精度モード
                print("スクリーンショット高精度モードで処理中...")

                debug_paths = []
                screenshot_psm = psm if psm is not None else 11

                # 1. スクリーンショット専用前処理
                processed_screenshot = preprocess_image_screenshot(
                    image, args.detect_rotation
                )
                if args.debug:
                    debug_screenshot_path = image_path.with_suffix(".screenshot.png")
                    cv2.imwrite(str(debug_screenshot_path), processed_screenshot)
                    debug_paths.append(debug_screenshot_path)
                    print(
                        f"スクショ前処理済み画像を保存しました: {debug_screenshot_path}"
                    )

                ocr_result_screenshot = execute_ocr(
                    processed_screenshot,
                    args.lang,
                    psm=screenshot_psm,
                    tessdata_dir=tessdata_dir,
                )
                print(f"スクショ前処理: {ocr_result_screenshot['total_elements']}要素")

                # 2. 軽量前処理 (二値化なし)
                processed_light = preprocess_image_light(image, args.detect_rotation)
                if args.debug:
                    debug_light_path = image_path.with_suffix(".light.png")
                    cv2.imwrite(str(debug_light_path), processed_light)
                    debug_paths.append(debug_light_path)
                    print(f"軽量前処理済み画像を保存しました: {debug_light_path}")

                ocr_result_light = execute_ocr(
                    processed_light,
                    args.lang,
                    psm=screenshot_psm,
                    tessdata_dir=tessdata_dir,
                )
                print(f"軽量前処理: {ocr_result_light['total_elements']}要素")

                # 3. 反転処理 (ダークモード対応)
                processed_inverted = preprocess_image_inverted(
                    image, args.detect_rotation
                )
                if args.debug:
                    debug_inverted_path = image_path.with_suffix(".inverted.png")
                    cv2.imwrite(str(debug_inverted_path), processed_inverted)
                    debug_paths.append(debug_inverted_path)
                    print(f"反転処理済み画像を保存しました: {debug_inverted_path}")

                ocr_result_inverted = execute_ocr(
                    processed_inverted,
                    args.lang,
                    psm=screenshot_psm,
                    tessdata_dir=tessdata_dir,
                )
                print(f"反転処理: {ocr_result_inverted['total_elements']}要素")

                # 結果をマージ
                ocr_result = merge_ocr_results(ocr_result_screenshot, ocr_result_light)
                ocr_result = merge_ocr_results(ocr_result, ocr_result_inverted)
                print(f"マージ後: {ocr_result['total_elements']}要素")

                if args.debug:
                    debug_image_path = debug_paths

            elif args.screenshot:
                # スクリーンショット専用モード
                processed_image = preprocess_image_screenshot(
                    image, args.detect_rotation
                )

                # デバッグモード: 前処理後の画像を保存
                if args.debug:
                    debug_image_path = image_path.with_suffix(".screenshot.png")
                    cv2.imwrite(str(debug_image_path), processed_image)
                    print(f"前処理済み画像を保存しました: {debug_image_path}")

                # OCR実行 (PSM 11: スパーステキスト - UI向け)
                screenshot_psm = psm if psm is not None else 11
                ocr_result = execute_ocr(
                    processed_image,
                    args.lang,
                    psm=screenshot_psm,
                    tessdata_dir=tessdata_dir,
                )

            elif args.light:
                # 軽量モード: 二値化なし、スクリーンショット向け
                processed_image = preprocess_image_light(image, args.detect_rotation)

                # デバッグモード: 前処理後の画像を保存
                if args.debug:
                    debug_image_path = image_path.with_suffix(".light.png")
                    cv2.imwrite(str(debug_image_path), processed_image)
                    print(f"前処理済み画像を保存しました: {debug_image_path}")

                # OCR実行
                ocr_result = execute_ocr(
                    processed_image, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )

            elif args.aggressive:
                # 高精度モード: 複数の二値化手法を併用
                print("高精度モードで処理中...")

                debug_paths = []

                # 適応的閾値処理
                processed_adaptive = preprocess_image_adaptive(
                    image, args.detect_rotation
                )
                if args.debug:
                    debug_adaptive_path = image_path.with_suffix(".adaptive.png")
                    cv2.imwrite(str(debug_adaptive_path), processed_adaptive)
                    debug_paths.append(debug_adaptive_path)
                    print(
                        f"適応的閾値処理済み画像を保存しました: {debug_adaptive_path}"
                    )

                ocr_result_adaptive = execute_ocr(
                    processed_adaptive, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )
                print(f"適応的閾値処理: {ocr_result_adaptive['total_elements']}要素")

                # Otsu二値化
                processed_otsu = preprocess_image_otsu(image, args.detect_rotation)
                if args.debug:
                    debug_otsu_path = image_path.with_suffix(".otsu.png")
                    cv2.imwrite(str(debug_otsu_path), processed_otsu)
                    debug_paths.append(debug_otsu_path)
                    print(f"Otsu二値化済み画像を保存しました: {debug_otsu_path}")

                ocr_result_otsu = execute_ocr(
                    processed_otsu, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )
                print(f"Otsu二値化: {ocr_result_otsu['total_elements']}要素")

                # 結果をマージ
                ocr_result = merge_ocr_results(ocr_result_adaptive, ocr_result_otsu)

                # 白抜き文字処理も追加
                processed_inverted = preprocess_image_inverted(
                    image, args.detect_rotation
                )
                if args.debug:
                    debug_inverted_path = image_path.with_suffix(".inverted.png")
                    cv2.imwrite(str(debug_inverted_path), processed_inverted)
                    debug_paths.append(debug_inverted_path)
                    print(f"反転処理済み画像を保存しました: {debug_inverted_path}")

                ocr_result_inverted = execute_ocr(
                    processed_inverted, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )
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
                ocr_result = execute_ocr(
                    processed_image, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )

            else:
                # 通常モード: 適応的閾値処理のみ
                processed_image = preprocess_image_adaptive(image, args.detect_rotation)

                # デバッグモード: 前処理後の画像を保存
                if args.debug:
                    debug_image_path = image_path.with_suffix(".preprocessed.png")
                    cv2.imwrite(str(debug_image_path), processed_image)
                    print(f"前処理済み画像を保存しました: {debug_image_path}")

                # OCR実行
                ocr_result = execute_ocr(
                    processed_image, args.lang, psm=psm, tessdata_dir=tessdata_dir
                )

            # 後処理: 低信頼度要素の再OCR
            if not args.no_retry and args.retry_threshold > 0:
                ocr_result = retry_low_confidence_ocr(
                    image,
                    ocr_result,
                    confidence_threshold=args.retry_threshold,
                    lang=args.lang,
                    tessdata_dir=tessdata_dir,
                )

            # 後処理: UI誤認識補正 (スクリーンショットモード時)
            if args.screenshot:
                for elem in ocr_result["elements"]:
                    elem["text"] = correct_ui_text(elem["text"])

            # 後処理: 日本語誤認識修正
            if not args.no_japanese_correct:
                for elem in ocr_result["elements"]:
                    elem["text"] = correct_japanese_text(elem["text"])

            # 後処理: 信頼度フィルタリング
            if args.min_confidence > 0:
                original_count = len(ocr_result["elements"])
                ocr_result["elements"] = filter_by_confidence(
                    ocr_result["elements"], args.min_confidence
                )
                ocr_result["total_elements"] = len(ocr_result["elements"])
                filtered_count = original_count - ocr_result["total_elements"]
                if filtered_count > 0 and args.debug:
                    print(f"信頼度フィルタ: {filtered_count}要素を除外")

            # 後処理: 形態素解析 (Janome)
            if args.post_process:
                try:
                    from janome.tokenizer import Tokenizer

                    tokenizer = Tokenizer()

                    for elem in ocr_result["elements"]:
                        text = elem["text"]
                        # 形態素解析で文脈を確認し、誤認識を修正
                        tokens = list(tokenizer.tokenize(text))
                        corrected_parts = []
                        for token in tokens:
                            word = token.surface
                            pos = token.part_of_speech.split(",")[0]
                            # 名詞でカタカナっぽいのに漢字が混ざっている場合は補正
                            if pos == "名詞":
                                word = correct_japanese_text(word)
                            corrected_parts.append(word)
                        elem["text"] = "".join(corrected_parts)
                except ImportError:
                    print(
                        "警告: janomeがインストールされていません。--post-processは無視されます",
                        file=sys.stderr,
                    )

            # JSON変換
            json_output = convert_to_json(ocr_result, image_path.name, resolution)

            # 出力ファイル名の決定
            if args.output and len(args.images) == 1:
                output_path = Path(args.output)
            else:
                output_path = image_path.with_suffix(image_path.suffix + ".ocr.json")

            # JSON出力
            output_path.write_text(json_output, encoding="utf-8")
            print(f"OCR結果を保存しました: {output_path}")
            print(f"認識されたテキスト要素数: {ocr_result['total_elements']}")

            success_count += 1

        except Exception as e:
            print(f"エラーが発生しました ({image_file}): {e}", file=sys.stderr)
            error_count += 1

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

    # 複数ファイル処理時のサマリ
    if len(args.images) > 1:
        print(f"\n処理完了: 成功 {success_count}件, エラー {error_count}件")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
