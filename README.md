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

# 高精度モード(薄いグレー文字の認識向上、処理時間約2倍)
uv run python gagan.py screenshot.png --aggressive

# デバッグモード(前処理後の画像を一時保存、処理後自動削除)
uv run python gagan.py screenshot.png --debug
```

### 処理モード

**通常モード(デフォルト)**
- 適応的閾値処理を使用
- 高速(約1-2秒)
- 一般的な画面テストに最適

**高精度モード(`--aggressive`)**
- 適応的閾値処理とOtsu二値化を併用
- 処理時間約2倍
- 薄いグレー文字や低コントラスト文字の認識に有効
- 2つの結果を統合し、信頼度の高い方を採用

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
