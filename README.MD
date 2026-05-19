Launch:
1) pip install -r requirements.txt
2) create input_videos directory and add video you want to proceed
3) python main.py \\
  --cropper-model ./src/YOLO_CROPPER.pt \\
  --price-model ./src/YOLO_PRICE.pt \\
  --videos ./input_videos \\
  --output ./results \\
  --conf 0.25 \\
  --zone-conf 0.30 \\
  --product-db ./products.csv \\
  --db-min-score 30

Аргумент,Описание
--cropper-model,Путь к YOLO модели для детектирования/обрезки целевых зон.
--price-model,Путь к YOLO модели для распознавания зон ценника.
--videos,Путь к папке с исходными видеофайлами для обработки.
--output,"Директория, куда будут сохранены результаты работы пайплайна."
--conf,Порог уверенности (Confidence threshold) для основной модели детектирования.
--zone-conf,Порог уверенности для определения зон интереса.
--product-db,Путь к CSV-файлу с базой данных продуктов для сопоставления.
--db-min-score,Минимальный score (порог совпадения) для валидации продукта по базе данных.

