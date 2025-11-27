"""CLIモジュール"""

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_WORKERS, set_high_resolution_threshold
from src.processor import (
    resolve_tessdata_dir,
    run_benchmark_mode,
    run_ocr_mode,
    run_regions_only_mode,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサを構築する。"""
    parser = argparse.ArgumentParser(description="GAGAN - 画面テスト用OCRツール")

    # 基本オプション
    parser.add_argument(
        "images", nargs="+", help="OCRを実行する画像ファイルのパス(複数指定可)"
    )
    parser.add_argument(
        "-o", "--output", help="出力JSONファイル名(単一ファイル時のみ有効)"
    )

    # === 画像処理オプション (OCR不要) ===
    parser.add_argument(
        "--save-sharpened", action="store_true", help="シャープ化画像を出力"
    )
    parser.add_argument(
        "--regions-only",
        action="store_true",
        help="テキスト領域検出のみ (OCRをスキップ)",
    )
    parser.add_argument(
        "--super-resolution", action="store_true", help="超解像によるアップスケール"
    )
    parser.add_argument(
        "--text-detection", action="store_true", help="EASTテキスト領域検出"
    )

    # === 前処理オプション ===
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="画像前処理をスキップする"
    )
    parser.add_argument("--light", action="store_true", help="軽量モード(二値化なし)")
    parser.add_argument(
        "--screenshot", action="store_true", help="スクリーンショット専用モード"
    )
    parser.add_argument("--inverted", action="store_true", help="白抜き文字モード")
    parser.add_argument(
        "--detect-rotation", action="store_true", help="回転検出と補正を有効化"
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="前処理でのアップスケールを無効化 (高解像度画像向け)",
    )

    # === OCRオプション ===
    parser.add_argument(
        "--lang", default="jpn+eng", help="OCR言語設定(デフォルト: jpn+eng)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["tesseract", "paddleocr"],
        default="tesseract",
        help="OCRエンジン (tesseract/paddleocr、デフォルト: tesseract)",
    )
    parser.add_argument(
        "--psm", type=str, default="auto", help="Page Segmentation Mode"
    )
    parser.add_argument("--best", action="store_true", help="tessdata_bestを使用")
    parser.add_argument("--tessdata-dir", type=str, help="tessdataディレクトリを指定")

    # === 精度オプション ===
    parser.add_argument(
        "--aggressive", action="store_true", help="高精度モード(複数手法併用)"
    )
    parser.add_argument("--max-accuracy", action="store_true", help="最高精度モード")
    parser.add_argument(
        "--fast", action="store_true", help="高速モード(精度向上機能をOFF)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.3, help="最小信頼度フィルタ"
    )
    parser.add_argument("--char-retry", action="store_true", help="文字単位再OCRを実行")
    parser.add_argument(
        "--no-japanese-correct", action="store_true", help="日本語誤認識修正を無効化"
    )
    parser.add_argument(
        "--post-process", action="store_true", help="形態素解析による後処理を有効化"
    )

    # === 実行オプション ===
    parser.add_argument("--no-parallel", action="store_true", help="並列処理を無効化")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"並列処理のワーカー数 (デフォルト: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="ベンチマークモード (TesseractとPaddleOCRを比較)",
    )

    # === デバッグオプション ===
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード(前処理後の画像を保存)"
    )
    parser.add_argument(
        "--keep-debug-images",
        action="store_true",
        help="デバッグ画像を削除せず保持する",
    )

    return parser


def main() -> int:
    """CLIエントリーポイント。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    # --max-accuracyの展開
    if args.max_accuracy:
        args.best = True
        args.post_process = True
        args.aggressive = True
        args.char_retry = True

    # --fastの処理
    if args.fast:
        args.min_confidence = 0.0
        args.no_japanese_correct = True
        if args.psm == "auto":
            args.psm = "3"

    # --no-upscaleの処理 (閾値を0に設定して全画像でアップスケールをスキップ)
    if args.no_upscale:
        set_high_resolution_threshold(0)

    # 信頼度閾値のバリデーション
    if not (0.0 <= args.min_confidence <= 1.0):
        print(
            f"警告: min-confidence {args.min_confidence} は範囲外です (0.0-1.0)。"
            "0.3に設定します",
            file=sys.stderr,
        )
        args.min_confidence = 0.3

    # 複数ファイル警告
    if len(args.images) > 1 and args.output:
        print("警告: 複数ファイル指定時は-oオプションは無視されます", file=sys.stderr)

    # ワーカー数のバリデーション
    if args.workers < 1:
        print(
            f"警告: workers {args.workers} は無効です。1に設定します",
            file=sys.stderr,
        )
        args.workers = 1

    tessdata_dir = resolve_tessdata_dir(args)
    parallel = not args.no_parallel
    workers = args.workers

    # 存在するファイルのみ抽出
    valid_image_paths: list[Path] = []
    error_count = 0
    for image_file in args.images:
        image_path = Path(image_file)
        if not image_path.exists():
            print(
                f"エラー: 画像ファイルが見つかりません: {image_file}", file=sys.stderr
            )
            error_count += 1
        else:
            valid_image_paths.append(image_path)

    # 領域検出のみモード
    if args.regions_only:
        error_count += run_regions_only_mode(args, valid_image_paths)
        return error_count

    # ベンチマークモード
    if args.benchmark:
        error_count += run_benchmark_mode(args, valid_image_paths, tessdata_dir)
        return 0 if error_count == 0 else 1

    # OCRモード
    success_count, ocr_errors = run_ocr_mode(
        args, valid_image_paths, tessdata_dir, parallel, workers
    )
    error_count += ocr_errors

    return 0 if error_count == 0 else 1
