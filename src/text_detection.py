"""テキスト領域検出モジュール"""

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.preprocessing import upscale_image


def upscale_with_super_resolution(
    image: np.ndarray, scale: int = 4, model_name: str = "espcn"
) -> np.ndarray:
    """超解像によるアップスケール。"""
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        model_paths = [
            Path(__file__).parent.parent
            / "models"
            / f"{model_name.upper()}_x{scale}.pb",
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
            Path(__file__).parent.parent / "models" / "frozen_east_text_detection.pb",
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
