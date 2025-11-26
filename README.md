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

# 軽量モード(スクリーンショット向け、二値化なし、推奨)
uv run python gagan.py screenshot.png --light

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

### tessdata_bestのインストール

高精度訓練データを使用するには、別途ダウンロードが必要:

```bash
# ダウンロード
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn.traineddata
wget https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# インストール
sudo mkdir -p /usr/share/tesseract-ocr/5/tessdata_best
sudo mv jpn.traineddata eng.traineddata /usr/share/tesseract-ocr/5/tessdata_best/
```

### 超解像モデルのインストール

超解像機能を使用するには、ESPCNモデルが必要:

```bash
# ダウンロード
wget https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb

# インストール
mkdir -p models
mv ESPCN_x4.pb models/
```

### EASTテキスト検出モデルのインストール

テキスト領域検出機能を使用するには:

```bash
# ダウンロード
wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
tar -xzf frozen_east_text_detection.tar.gz

# インストール
mkdir -p models
mv frozen_east_text_detection.pb models/
```

### 処理モード

**通常モード(デフォルト)**
- 適応的閾値処理を使用
- 高速(約1-2秒)
- 小さな画像は自動でアップスケール(高さ1000px以上)

**軽量モード(`--light`) - スクリーンショット推奨**
- 二値化を行わず、グレースケール+軽いノイズ除去のみ
- 文字のつぶれが少なく、ボタンの縁も保持
- クリアなスクリーンショットに最適

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

### OCR失敗ケースと対処法

| 失敗ケース | 対処法 | オプション |
|---------|--------|---------|
| 薄いグレーの文字が認識できない | 高精度モードを使用 | `--aggressive` |
| 白抜き文字が認識できない | 白抜き文字モードまたは高精度モード | `--inverted` または `--aggressive` |
| 画像が斜めになっている | 回転検出を有効化 | `--detect-rotation` |
| 小さな文字が潰れる | 自動でアップスケール(1000px未満の画像) | (自動適用) |
| 太文字が潰れる | モルフォロジー処理を最適化済み | (自動適用) |
| カタカナが漢字として誤認識 | 日本語誤認識修正(デフォルトON) | (自動適用) |
| 低解像度画像の認識精度が悪い | 超解像を使用 | `--super-resolution` |
| 疎なテキストが検出されない | テキスト領域検出を使用 | `--text-detection` |
| 全体的に精度が低い | 最高精度モードを使用 | `--max-accuracy` |
| 処理が遅い | 高速モードを使用 | `--fast` |

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
