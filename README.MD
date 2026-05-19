Launch:

1) Python **3.10** (required for degan/TensorFlow + PaddleOCR + YOLO):

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Create `input_videos` and add your video files.

3) Run the pipeline:

```bash
python main.py ^
  --cropper-model ./src/YOLO_CROPPER.pt ^
  --price-model ./src/YOLO_PRICE.pt ^
  --videos ./input_videos ^
  --output ./results ^
  --conf 0.25 ^
  --zone-conf 0.30 ^
  --product-db ./products.csv ^
  --db-min-score 30
```

Optional: **DE-GAN deblur** once per price-tag crop (before zone YOLO + OCR):

```bash
python main.py ... --use-degan
```

Optional: disable per-zone LANCZOS upscale (A/B test):

```bash
python main.py ... --no-zone-upscale
```

| Аргумент | Описание |
| --- | --- |
| `--cropper-model` | Путь к YOLO модели для детектирования ценников на кадре |
| `--price-model` | Путь к YOLO модели для сегментации зон внутри кропа |
| `--videos` | Папка с видео |
| `--output` | Папка результатов |
| `--conf` | Порог уверенности YOLO_CROPPER |
| `--zone-conf` | Порог уверенности YOLO_PRICE |
| `--product-db` | CSV база товаров (fullname, code) |
| `--db-min-score` | Минимальный rapidfuzz score для матча |
| `--use-degan` | DE-GAN deblur кропа ценника один раз перед OCR |
| `--no-zone-upscale` | Отключить LANCZOS-апскейл мелких zone-кропов перед OCR |
