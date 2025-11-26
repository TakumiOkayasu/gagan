# GAGAN (画眼)

スクリーンショットからテキストを抽出し、座標情報とともにJSON形式で出力するOCRツール。

## インストール

### Tesseract OCRのインストール

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
```

### Pythonパッケージのインストール

```bash
uv pip install -r requirements.txt
```

## 使い方

### 基本的な使用

```bash
uv run python gagan.py screenshot.png
```

デフォルトで `screenshot.png.ocr.json` が生成される。

### オプション

```bash
# 出力ファイル名を指定
uv run python gagan.py screenshot.png -o result.json

# 言語指定(デフォルト: jpn+eng)
uv run python gagan.py screenshot.png --lang jpn+eng

# 前処理をスキップ
uv run python gagan.py screenshot.png --no-preprocessing

# 軽量モード(二値化なし)
uv run python gagan.py screenshot.png --light

# スクリーンショット専用モード(推奨)
uv run python gagan.py screenshot.png --screenshot

# スクリーンショット + 最高精度モード(ダーク/ライト両対応)
uv run python gagan.py screenshot.png --screenshot --max-accuracy

# 高精度モード(薄いグレー文字・白抜き文字の認識向上、処理時間約3倍)
uv run python gagan.py screenshot.png --aggressive

# 白抜き文字モード(暗い背景に白文字がある場合)
uv run python gagan.py screenshot.png --inverted

# 回転検出と補正(画像が斜めになっている場合)
uv run python gagan.py screenshot.png --detect-rotation

# デバッグモード(前処理後の画像を一時保存、処理後自動削除)
uv run python gagan.py screenshot.png --debug

# オプション組み合わせ例
uv run python gagan.py screenshot.png --aggressive --detect-rotation --debug
```

### 精度向上オプション (v2.0新機能)

```bash
# 高速モード(従来互換、精度向上機能OFF)
uv run python gagan.py screenshot.png --fast

# 最高精度モード(--best --post-process --aggressive を有効化)
uv run python gagan.py screenshot.png --max-accuracy

# PSM (Page Segmentation Mode) 指定
uv run python gagan.py screenshot.png --psm auto   # 自動選択(デフォルト)
uv run python gagan.py screenshot.png --psm 6     # Single block(UI向け)
uv run python gagan.py screenshot.png --psm 11    # Sparse text(疎なテキスト)

# 信頼度フィルタ(低信頼度の結果を除外)
uv run python gagan.py screenshot.png --min-confidence 0.5

# 低信頼度要素の再OCR閾値を変更(デフォルト: 0.8)
uv run python gagan.py screenshot.png --retry-threshold 0.9

# 低信頼度要素の再OCRを無効化
uv run python gagan.py screenshot.png --no-retry

# 誤認識しやすい文字の文字単位再OCR (占、上、浴など)
uv run python gagan.py screenshot.png --char-retry

# 日本語誤認識修正を無効化
uv run python gagan.py screenshot.png --no-japanese-correct

# tessdata_best(高精度訓練データ)を使用
uv run python gagan.py screenshot.png --best

# 形態素解析による後処理(要: janome)
uv run python gagan.py screenshot.png --post-process

# 超解像によるアップスケール(要: opencv-contrib-python)
uv run python gagan.py screenshot.png --super-resolution

# EASTテキスト領域検出(要: モデルファイル)
uv run python gagan.py screenshot.png --text-detection
```

### 並列処理オプション

```bash
# 並列処理を有効化 (複数ファイル・aggressive・再OCRを並列実行)
uv run python gagan.py *.png --parallel

# ワーカー数を指定 (デフォルト: CPUコア数、最大8)
uv run python gagan.py *.png --parallel --workers 4

# 高精度モードと並列処理の組み合わせ
uv run python gagan.py *.png --screenshot --aggressive --parallel
```

**並列処理の対象:**
- 複数ファイルの同時処理 (ProcessPoolExecutor)
- aggressive/screenshot_aggressiveモードでの複数前処理の同時実行 (ThreadPoolExecutor)
- 低信頼度要素の再OCR (ThreadPoolExecutor)
- 文字単位再OCR (ThreadPoolExecutor)

### tessdata_bestのインストール

高精度訓練データを使用するには、別途ダウンロードが必要:

```bash
# ダウンロード
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn.traineddata
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# インストール
sudo mkdir -p /usr/share/tesseract-ocr/5/tessdata_best && mv jpn.traineddata eng.traineddata /usr/share/tesseract-ocr/5/tessdata_best/
```

### MLモデルのインストール (超解像・テキスト検出)

`--super-resolution` や `--text-detection` オプションを使用するには、MLモデルが必要:

```bash
# セットアップスクリプトで一括ダウンロード (推奨)
./setup_models.sh
```

<details>
<summary>手動インストール</summary>

**超解像モデル (ESPCN_x4.pb)**

```bash
wget https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb
mkdir -p models && mv ESPCN_x4.pb models/
```

**EASTテキスト検出モデル (frozen_east_text_detection.pb)**

```bash
wget https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb
mkdir -p models && mv frozen_east_text_detection.pb models/
```

</details>

### 処理モード

**通常モード(デフォルト)**
- 適応的閾値処理を使用
- 高速(約1-2秒)
- 小さな画像は自動でアップスケール(高さ1000px以上)

**軽量モード(`--light`)**
- 二値化を行わず、グレースケール+軽いノイズ除去のみ
- 文字のつぶれが少なく、ボタンの縁も保持

**スクリーンショット専用モード(`--screenshot`) - 推奨**
- ダークモード自動検出と反転処理
- 小さいUI要素 (ボタンラベル等) の自動スケーリング
- PSM 11 (スパーステキスト) をデフォルト使用
- UI特有の誤認識補正 (OK, Cancel, File等)
- `--aggressive` と併用で複数前処理をマージ

**高精度モード(`--aggressive`)**
- 適応的閾値処理、Otsu二値化、反転処理の3つを併用
- 処理時間約3倍(約3-4秒)
- 薄いグレー文字や低コントラスト文字の認識に有効
- 白抜き文字(暗い背景に白文字)も同時に検出
- 3つの結果を統合し、信頼度の高い方を採用

**白抜き文字モード(`--inverted`)**
- 暗い背景に白文字がある場合に使用
- 画像を反転してから処理
- ダークモードのUIなどに有効

**回転補正(`--detect-rotation`)**
- 画像の傾きを自動検出して補正
- スマホで斜めに撮影した画面などに有効
- 0.5度以上の傾きがある場合に補正

**低信頼度要素の再OCR (デフォルト有効)**
- 信頼度0.8未満の要素に対して自動的に再OCRを実行
- 該当要素の領域を切り出し、4種類の前処理 (adaptive, otsu, inverted, light) で再試行
- 複数のPSM (7: 単一行, 8: 単一単語, 13: Raw line) で再試行
- 最も信頼度の高い結果を採用
- `--retry-threshold` で閾値を変更可能 (例: 0.9でより厳密に再OCR)
- `--no-retry` で無効化可能 (処理速度優先時)

**文字単位再OCR (`--char-retry`)**
- 誤認識しやすい文字 (占、上、浴、甲、丘、士、充、民、音) を含む要素を検出
- 該当領域を4倍に拡大し、シャープネス強化後に再OCR
- 複数の前処理 (シャープ、適応的閾値、Otsu) と複数PSMで試行
- 疑わしい文字が減少し、信頼度が同等以上の結果を採用
- `--max-accuracy` に含まれる

### OCR失敗ケースと対処法

| 失敗ケース | 対処法 | オプション |
|---------|--------|---------|
| スクリーンショット全般 | スクショ専用モードを使用 | `--screenshot` |
| ダークモードのスクショ | スクショモードで自動検出 | `--screenshot` |
| 薄いグレーの文字が認識できない | 高精度モードを使用 | `--aggressive` |
| 白抜き文字が認識できない | 白抜き文字モードまたは高精度モード | `--inverted` または `--aggressive` |
| 画像が斜めになっている | 回転検出を有効化 | `--detect-rotation` |
| 小さな文字が潰れる | 自動でアップスケール(1000px未満の画像) | (自動適用) |
| 太文字が潰れる | モルフォロジー処理を最適化済み | (自動適用) |
| カタカナが漢字として誤認識 | 日本語誤認識修正(デフォルトON) | (自動適用) |
| UIラベルの誤認識 (0K→OK等) | スクショモードで自動補正 | `--screenshot` |
| 低解像度画像の認識精度が悪い | 超解像を使用 | `--super-resolution` |
| 疎なテキストが検出されない | テキスト領域検出を使用 | `--text-detection` |
| 全体的に精度が低い | 最高精度モードを使用 | `--max-accuracy` |
| スクショで最高精度が必要 | スクショ+最高精度モード | `--screenshot --max-accuracy` |
| 信頼度の低い結果が多い | 再OCR閾値を上げる | `--retry-threshold 0.9` |
| 「占」「上」等の誤認識 | 文字単位再OCRを使用 | `--char-retry` |
| 処理が遅い | 高速モードを使用 | `--fast` |
| 処理が遅い (再OCR無効化) | 再OCRをスキップ | `--no-retry` |

### よく使うフラグの組み合わせ

#### スクリーンショットのOCR (推奨)

```bash
# 基本 - ほとんどのスクショはこれでOK
uv run python gagan.py screenshot.png --screenshot

# 最高精度 - 複雑なUIや小さい文字がある場合
uv run python gagan.py screenshot.png --screenshot --max-accuracy

# ダークモードのスクショ - 自動検出されるが、明示的に指定も可能
uv run python gagan.py screenshot.png --screenshot --inverted
```

#### 製造業・業務システムのUI

```bash
# 専門用語や製品コードが多い場合
uv run python gagan.py screenshot.png --screenshot --max-accuracy --char-retry

# 薄いグレー文字がある場合
uv run python gagan.py screenshot.png --screenshot --aggressive
```

#### 処理速度優先

```bash
# 高速処理 (精度向上機能OFF)
uv run python gagan.py screenshot.png --fast

# 再OCRを無効化して高速化
uv run python gagan.py screenshot.png --screenshot --no-retry
```

#### 複数ファイルの一括処理

```bash
# ワイルドカードで複数ファイル
uv run python gagan.py screenshots/*.png --screenshot

# 最高精度で一括処理
uv run python gagan.py *.png --screenshot --max-accuracy

# 並列処理で高速化 (大量ファイル向け)
uv run python gagan.py *.png --screenshot --parallel

# 最高精度 + 並列処理 (精度と速度のバランス)
uv run python gagan.py *.png --screenshot --max-accuracy --parallel --workers 4
```

#### デバッグ・トラブルシューティング

```bash
# 前処理後の画像を確認
uv run python gagan.py screenshot.png --screenshot --debug --keep-debug-images

# 信頼度の低い結果も含めて確認
uv run python gagan.py screenshot.png --screenshot --min-confidence 0.1
```

## 出力形式

```json
{
  "timestamp": "2025-11-20T11:51:48",
  "source_image": "screenshot.png",
  "resolution": {
    "width": 3840,
    "height": 2160
  },
  "elements": [
    {
      "id": 0,
      "text": "認識されたテキスト",
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 30
      },
      "confidence": 0.95
    }
  ]
}
```
