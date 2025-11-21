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

### 処理モード

**通常モード(デフォルト)**
- 適応的閾値処理を使用
- 高速(約1-2秒)
- 一般的な画面テストに最適
- 小さな画像は自動でアップスケール(高さ1000px以上)

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
