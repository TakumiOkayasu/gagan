"""画像前処理モジュール"""

from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.config import get_high_resolution_threshold
from src.types import BBox

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
    image: np.ndarray, min_height: int = 1000, skip_for_high_res: bool = True
) -> tuple[np.ndarray, float]:
    """画像が小さい場合にアップスケールする。

    Args:
        image: 入力画像
        min_height: この高さ未満の画像をアップスケール
        skip_for_high_res: 高解像度画像でアップスケールをスキップするか

    Returns:
        (処理後画像, スケール倍率)
    """
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0
    # 高解像度画像ではアップスケールをスキップ
    if skip_for_high_res and h >= get_high_resolution_threshold():
        return image, 1.0
    if h < min_height:
        scale = min_height / h
        return upscale_image(image, scale), scale
    return image, 1.0


def upscale_small_ui_elements(
    image: np.ndarray, min_text_height: int = 32, skip_for_high_res: bool = True
) -> tuple[np.ndarray, float]:
    """小さいUI要素の認識精度向上のためスケーリングする。

    Args:
        image: 入力画像
        min_text_height: 目標となる最小テキスト高さ
        skip_for_high_res: 高解像度画像でアップスケールをスキップするか

    Returns:
        (処理後画像, スケール倍率)
    """
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, 1.0
    # 高解像度画像ではアップスケールをスキップ
    if skip_for_high_res and h >= get_high_resolution_threshold():
        return image, 1.0
    if h < 100:
        scale = max(2.0, min_text_height / 12)
        return upscale_image(image, scale), scale
    return image, 1.0


def apply_sharpening(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """アンシャープマスクでシャープネスを強化する。"""
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)


def apply_auto_sharpening(image: np.ndarray) -> np.ndarray:
    """画像読み込み直後に自動でシャープ化を適用する。

    リサイズされた画像の文字ボケを軽減するための処理。
    カラー画像・グレースケール画像の両方に対応。
    """
    if image is None or image.size == 0:
        return image

    # カラー画像の場合はLチャンネルのみシャープ化 (色ずれ防止)
    if len(image.shape) == 3 and image.shape[2] >= 3:
        # BGRからLabに変換
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Lチャンネルにシャープ化適用
        l_sharpened = apply_sharpening(l_channel, sigma=1.5)

        # 再結合
        lab_sharpened = cv2.merge([l_sharpened, a_channel, b_channel])
        return cv2.cvtColor(lab_sharpened, cv2.COLOR_LAB2BGR)
    else:
        # グレースケール画像
        return apply_sharpening(image, sigma=1.5)


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
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    return apply_sharpening(denoised)


def preprocess_image_light(
    image: np.ndarray | Image.Image, detect_rotation_flag: bool = False
) -> np.ndarray:
    """軽量前処理を実行する。二値化を行わない。"""
    gray = to_grayscale(image)
    gray = apply_rotation_if_needed(gray, detect_rotation_flag)
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    return apply_sharpening(denoised)


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

    sharpened = apply_sharpening(gray)
    gamma_corrected = cv2.LUT(sharpened, get_gamma_table(1.2))
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

    sharpened = apply_sharpening(gray)
    inverted = cv2.bitwise_not(sharpened)
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
