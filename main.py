#!/usr/bin/env python3
"""
PriceTag Pipeline v5 — Двухэтапный YOLO + зональный OCR
=========================================================
Пайплайн:
  1. YOLO_CROPPER.pt   — детекция ценников на кадре видео  → кроп ценника (BGR)
  2. YOLO_PRICE.pt     — сегментация зон внутри кропа      → zone-кропы по классам
  3. PaddleOCR         — OCR каждой текстовой зоны (не QR/barcode)
  4. pyzbar / WeChatQR — декодирование zone_barcode / zone_qr_code
  5. ProductDB (опц.)  — фаззи-матчинг product_name → code → barcode
  6. CSV               — одна строка на уникальный ценник с координатами

Важно: координаты bbox кропа и сам кроп хранятся вместе в одном dict
и НИКОГДА не разлучаются вплоть до записи в CSV.

Модели:
  --cropper-model  YOLO_CROPPER.pt   (детекция ценников на видеокадре)
  --price-model    YOLO_PRICE.pt     (сегментация зон внутри кропа ценника)
"""

# ─── stdlib ────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ─── third-party (проверяем наличие) ───────────────────────────────────────
try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("❌ Установите: pip install opencv-contrib-python-headless numpy")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌ Установите: pip install ultralytics")

try:
    from PIL import Image
    import imagehash
except ImportError:
    sys.exit("❌ Установите: pip install imagehash Pillow")

try:
    from paddleocr import PaddleOCR
except ImportError:
    sys.exit(
        "❌ Установите: pip install -r requirements.txt  "
        "(paddleocr>=3.0, paddlepaddle, langchain-text-splitters, shapely, pyclipper)"
    )

try:
    import pyzbar.pyzbar as pyzbar
except ImportError:
    pyzbar = None
    logging.warning("⚠️  pyzbar не найден — 1D-штрихкоды не будут декодированы. "
                    "pip install pyzbar")

try:
    from cv2 import QRCodeDetector
    _WECHAT_OK = hasattr(cv2, 'wechat_qrcode')
except Exception:
    _WECHAT_OK = False

# rapidfuzz — быстрый нечёткий поиск
try:
    from rapidfuzz import fuzz, process as rfprocess
except ImportError:
    sys.exit("❌ Установите: pip install rapidfuzz")

# scikit-learn — TF-IDF индекс для предфильтрации 600k строк
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.sparse as sp
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logging.warning("⚠️  scikit-learn не найден — TF-IDF предфильтрация отключена. "
                    "pip install scikit-learn scipy")

import pandas as pd
from dateutil import parser as dateutil_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════════════════
DEBUG_DIR = 'results/debug'
CSV_FIELDS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
    "x_min", "y_min", "x_max", "y_max",
]

PRICE_RE = re.compile(
    r"(?<!\d)"                           # не цифра перед
    r"(\d{1,6})"                         # рублёвая часть
    r"[.,\s]?"
    r"(\d{2})?"                          # копейки (опц.)
    r"(?!\d)",                           # не цифра после
)

DATE_RE = re.compile(
    r"\b(\d{2}[./-]\d{2}[./-]\d{2,4})"
    r"(?:\s+(\d{2}:\d{2}(?::\d{2})?))?",
)

BARCODE_RE = re.compile(r"\b(\d{8,14})\b")

SKU_RE = re.compile(r"\b(?:арт\.?|sku|артикул)[\s:]*(\d{4,10})\b", re.I)
CODE_RE = re.compile(r"\b(?:код|zone|зона)[\s:]*([A-Za-zА-Яа-я0-9\-]{2,10})\b", re.I)

QR_KEYS = {
    "barcode": "qr_code_barcode", "b": "qr_code_barcode",
    "price1": "price1_qr",        "p1": "price1_qr",
    "price2": "price2_qr",        "p2": "price2_qr",
    "price3": "price3_qr",        "p3": "price3_qr",
    "price4": "price4_qr",        "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count", "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price", "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count", "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price", "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr",  "aP": "action_price_qr",
    "actionCode":  "action_code_qr",   "aC": "action_code_qr",
}

TAG_COLOR_MAP = {
    # BGR-диапазоны доминирующих цветов ценников Ленты
    "yellow":  ((20, 100, 100), (35, 255, 255)),   # жёлтый (HSV)
    "red":     ((0,  120, 70),  (10, 255, 255)),
    "green":   ((40, 50,  70),  (90, 255, 255)),
    "blue":    ((100,50,  70),  (130,255, 255)),
    "white":   ((0,  0,  200),  (180, 30, 255)),
}


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCT DB — быстрый матчинг по 600k наименований
# ═══════════════════════════════════════════════════════════════════════════

class ProductDB:
    """
    Загружает CSV с полями `fullname` и `code`.
    Матчинг: TF-IDF cosine для топ-K кандидатов → rapidfuzz token_set_ratio для финального ранжирования.
    Время одного запроса: ~10-50 мс при 600k строк на CPU.
    """

    def __init__(self, csv_path: str, top_k: int = 30, min_score: int = 60):
        self.min_score = min_score
        self.top_k = top_k
        log.info(f"Загрузка БД товаров: {csv_path}")
        t0 = time.time()

        df = pd.read_csv(csv_path,
                         dtype=str, 
                         low_memory=False, 
                         encoding='cp1251', 
                         sep=';')
        df.dropna(subset=["fullname"], inplace=True)
        df["fullname"] = df["fullname"].str.strip()
        df["code"] = df.get("code", pd.Series(dtype=str)).fillna("").str.strip()
        self._names: List[str] = df["fullname"].tolist()
        self._codes: List[str] = df["code"].tolist()

        # Нормализованные имена для rapidfuzz
        self._names_norm = [self._normalize(n) for n in self._names]

        # TF-IDF индекс (char n-gram 2-4, субслово устойчиво к опечаткам)
        if _SKLEARN_OK:
            log.info("  Строим TF-IDF индекс…")
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                min_df=1, sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(self._names_norm)
            log.info(f"  Индекс готов ({self._matrix.shape[0]} строк, {time.time()-t0:.1f}s)")
        else:
            self._vectorizer = None
            self._matrix = None
            log.warning("  TF-IDF недоступен — используется линейный перебор rapidfuzz (медленно)")

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-zа-яё0-9\s]", " ", text.lower()).strip()

    def match(self, query: str) -> Tuple[Optional[str], Optional[str], int]:
        """
        Возвращает (fullname, code, score).
        score ∈ [0, 100].  Если ниже min_score — возвращает (None, None, 0).
        """
        if not query or not query.strip():
            return None, None, 0

        q_norm = self._normalize(query)

        if self._vectorizer is not None and self._matrix is not None:
            # ── Шаг 1: TF-IDF — отбираем top_k кандидатов ──
            q_vec = self._vectorizer.transform([q_norm])
            scores_cos = cosine_similarity(q_vec, self._matrix).ravel()
            # Берём топ-K индексов без полной сортировки
            if len(scores_cos) <= self.top_k:
                candidates_idx = list(range(len(scores_cos)))
            else:
                candidates_idx = np.argpartition(scores_cos, -self.top_k)[-self.top_k:].tolist()
        else:
            # Без sklearn — берём все (медленно, только для малых БД)
            candidates_idx = list(range(len(self._names_norm)))

        # ── Шаг 2: rapidfuzz token_set_ratio на кандидатах ──
        cands_text = [self._names_norm[i] for i in candidates_idx]
        best = rfprocess.extractOne(
            q_norm, cands_text,
            scorer=fuzz.token_set_ratio,
            score_cutoff=self.min_score,
        )
        if best is None:
            return None, None, 0

        best_text, best_score, local_idx = best
        global_idx = candidates_idx[local_idx]
        return self._names[global_idx], self._codes[global_idx], int(best_score)


# ═══════════════════════════════════════════════════════════════════════════
# ZONE SEGMENTOR — YOLO_PRICE.pt сегментирует зоны внутри кропа ценника
# ═══════════════════════════════════════════════════════════════════════════

# Маппинг класс-имя → поле CSV
# Зоны, которые идут в OCR (текст):
ZONE_TO_FIELD: Dict[str, str] = {
    "zone_product_name":     "product_name",
    "zone_price_default":    "price_default",
    "zone_price_card":       "price_card",
    "zone_price_discount":   "discount_amount", #размер скидки %
    # "zone_discount_amount":  "price_discount",
    # "zone_datetime":         "print_datetime",
    # "zone_special_symbols":  "special_symbols",
    # "zone_wholesale_table":  "who",   # таблица оптом → additional_info
}
# Зоны, которые НЕ идут в OCR, а декодируются специальным образом:
ZONE_DECODE_QR      = "zone_qr_code"
ZONE_DECODE_BARCODE = "zone_barcode"
# Класс-контейнер всего ценника (не несёт данных, только bbox ценника):
ZONE_PRICE_TAG      = "price_tag"

# Зоны, для которых нужен числовой парсинг цены (не raw-текст):
PRICE_ZONES = {"zone_price_default", "zone_price_card", "zone_price_discount",
               "zone_discount_amount"}


class ZoneSegmentor:
    """
    Запускает YOLO_PRICE.pt на кропе ценника.
    Возвращает список детектированных зон:
        [{"class": "zone_price_default", "bbox": (x1,y1,x2,y2), "conf": 0.87}, ...]
    bbox — координаты относительно кропа ценника.
    """

    def __init__(self, model_path: str, conf: float = 0.25):
        log.info(f"🏷️  Загрузка YOLO_PRICE: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        # Получаем маппинг id→name из модели
        self._names: Dict[int, str] = self.model.names  # {0: 'price_tag', 1: 'zone_barcode', ...}
        log.info(f"   Классы зон: {list(self._names.values())}")

    def segment(self, crop_bgr: np.ndarray) -> List[Dict]:
        """
        Принимает BGR numpy-кроп ценника.
        Возвращает список зон, отсортированных по убыванию conf.
        """
        results = self.model.predict(crop_bgr, conf=self.conf, verbose=False)
        zones: List[Dict] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self._names.get(cls_id, f"cls_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                # Клампинг координат в границы кропа
                h, w = crop_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    zones.append({
                        "class":  cls_name,
                        "bbox":   (x1, y1, x2, y2),
                        "conf":   conf_val,
                    })
        zones.sort(key=lambda z: z["conf"], reverse=True)
        return zones


# ═══════════════════════════════════════════════════════════════════════════
# DE-GAN deblur (опционально, один раз на кроп ценника)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from degan import DEGAN as _DEGAN

    _DEGAN_OK = True
except ImportError:
    _DEGAN = None  # type: ignore
    _DEGAN_OK = False


class DeganEnhancer:
    """Обёртка над degan: только deblur, BGR in/out."""

    def __init__(self):
        if not _DEGAN_OK:
            raise RuntimeError(
                "degan не установлен. Python 3.10: pip install -r requirements.txt"
            )
        log.info("🧹 Загрузка DE-GAN (deblur)…")
        self._model = _DEGAN(bin_weights=None, wat_weights=None)
        log.info("✅ DE-GAN готов")

    def deblur_bgr(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out = self._model.deblur(gray)
        out_u8 = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)


# ═══════════════════════════════════════════════════════════════════════════
# OCR-МОДУЛЬ (работает на numpy-массивах, без записи на диск)
# ═══════════════════════════════════════════════════════════════════════════

class OCRModule:
    """
    Двухэтапный OCR:
      1. ZoneSegmentor (YOLO_PRICE) → zone-кропы внутри ценника
      2. PaddleOCR на каждом текстовом zone-кропе
      3. pyzbar / WechatQR на zone_barcode / zone_qr_code
    """

    def __init__(self, zone_segmentor: ZoneSegmentor, zone_upscale: bool = True):
        self.segmentor = zone_segmentor
        self._zone_upscale = zone_upscale
        log.info("🔤 Инициализация PaddleOCR (PP-OCRv5, ru+en)…")
        # lang=ru + PP-OCRv5 → eslav_PP-OCRv5_mobile_rec (русский + английский + укр./бел.).
        # use_sr/ESRGAN в PaddleOCR 3.x нет (Unknown argument) — апскейл в _upscale_zone().
        self._ocr = PaddleOCR(
            lang="ru",
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
        # WechatQR
        if _WECHAT_OK:
            try:
                self._wechat = cv2.wechat_qrcode.WeChatQRCode()
            except Exception:
                self._wechat = None
        else:
            self._wechat = None
        log.info("✅ OCR готов")

    # ── апскейл мелких зон (замена use_sr/ESRGAN, недоступных в PaddleOCR 3.x) ─

    @staticmethod
    def _upscale_zone(bgr: np.ndarray, min_side: int = 128, scale_cap: float = 4.0) -> np.ndarray:
        """LANCZOS4-апскейл мелких кропов перед OCR (аналог SR для ценников)."""
        h, w = bgr.shape[:2]
        if h >= min_side and w >= min_side:
            return bgr
        scale = min(scale_cap, max(min_side / h, min_side / w))
        return cv2.resize(
            bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # ── предобработка зоны (не используется: бинаризация конфликтует с PP-OCRv5) ─

    @staticmethod
    def _preprocess_zone(bgr: np.ndarray) -> np.ndarray:
        """Резкость + адаптивный порог. Не вызывается — оставлено для экспериментов."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        # Для маленьких зон увеличиваем до минимального размера
        h, w = sharp.shape
        if h < 32 or w < 32:
            scale = max(32 / h, 32 / w)
            sharp = cv2.resize(
                sharp,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        binary = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        k = np.ones((1, 1), np.uint8)
        cleaned = cv2.dilate(cv2.erode(binary, k, iterations=1), k, iterations=1)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    # ── OCR одной зоны → список строк текста ──────────────────────────────

    def _ocr_zone(self, zone_bgr: np.ndarray) -> List[str]:
        try:
            if self._zone_upscale:
                zone_bgr = self._upscale_zone(zone_bgr)

            cv2.imwrite("debug_ocr_input.jpg", zone_bgr)

            result = self._ocr.predict(zone_bgr)

            lines = []

            if not result:
                return lines

            for item in result:
                rec_texts = item.get("rec_texts", [])
                rec_scores = item.get("rec_scores", [])

                for txt, score in zip(rec_texts, rec_scores):
                    print("TEXT:", txt, score)

                    if txt and score > 0.2:
                        lines.append(txt)

            return lines

        except Exception as e:
            print("OCR ERROR:", repr(e))
            return []
    # ── парсинг числовых значений цен из текста зоны ──────────────────────

    @staticmethod
    def _extract_price(lines: List[str]) -> str:
        """Берём первое валидное числовое значение (цена рублей[.копеек])."""
        full = " ".join(lines)
        m = PRICE_RE.search(full)
        if m:
            rub = m.group(1)
            kop = m.group(2) or "00"
            return f"{rub}.{kop}"
        return " ".join(lines)  # fallback — весь текст

    # ── декодирование QR ──────────────────────────────────────────────────

    def _decode_qr(self, bgr: np.ndarray) -> Dict[str, str]:
        result: Dict[str, str] = {}
        payloads: List[str] = []

        if self._wechat:
            try:
                texts, _ = self._wechat.detectAndDecode(bgr)
                payloads.extend([t for t in texts if t])
            except Exception:
                pass
        if not payloads:
            det = cv2.QRCodeDetector()
            data, _, _ = det.detectAndDecode(bgr)
            if data:
                payloads.append(data)

        for payload in payloads:
            parts = re.split(r"[&|;]", payload)
            for part in parts:
                for sep in ("=", ":"):
                    if sep in part:
                        k, _, v = part.partition(sep)
                        k, v = k.strip(), v.strip()
                        if k in QR_KEYS:
                            result[QR_KEYS[k]] = v
                        break
        return result

    # ── декодирование 1D-штрихкода ────────────────────────────────────────

    @staticmethod
    def _decode_barcode(bgr: np.ndarray) -> str:
        if pyzbar is None:
            return ""
        codes = pyzbar.decode(bgr)
        for c in codes:
            if c.type in ("EAN13", "EAN8", "CODE128", "CODE39", "UPC_A"):
                return c.data.decode("utf-8", errors="replace")
        return ""

    # ── цвет всего ценника (по полному кропу) ────────────────────────────

    @staticmethod
    def _dominant_color(bgr: np.ndarray) -> str:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        for color_name, (lo, hi) in TAG_COLOR_MAP.items():
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask = cv2.inRange(hsv, lo_arr, hi_arr)
            if cv2.countNonZero(mask) > bgr.shape[0] * bgr.shape[1] * 0.15:
                return color_name
        return "unknown"

    # ── главный метод ──────────────────────────────────────────────────────

    def recognize(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {f: "" for f in CSV_FIELDS}

        timestamp = int(time.time() * 1000)
        debug_prefix = f"{DEBUG_DIR}/crop_{timestamp}"

        cv2.imwrite(f"{debug_prefix}_raw.jpg", crop_bgr)

        try:
            out["color"] = self._dominant_color(crop_bgr)
        except:
            pass

        try:
            zones = self.segmentor.segment(crop_bgr)
        except Exception as e:
            print("SEGMENT ERROR:", e)
            zones = []

        debug_img = crop_bgr.copy()

        # fallback если YOLO ничего не дал
        if not zones:
            print("NO ZONES -> FULL OCR")

            lines = self._ocr_zone(crop_bgr)

            print("FULL OCR:", lines)

            full = " ".join(lines)

            out["additional_info"] = full[:300]

            alpha = [
                l for l in lines
                if re.search(r"[А-Яа-яA-Za-z]{3,}", l)
            ]

            if alpha:
                out["product_name"] = max(alpha, key=len)

            m = BARCODE_RE.search(full)
            if m:
                out["barcode"] = m.group(1)

            cv2.imwrite(f"{debug_prefix}_fallback.jpg", debug_img)

            return out

        for i, zone in enumerate(zones):
            try:
                cls = str(zone.get("class", "unknown"))


                bbox = zone.get("bbox", [])

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox]

                h, w = crop_bgr.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                zone_crop = crop_bgr[y1:y2, x1:x2]

                if zone_crop.size == 0:
                    continue

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.imwrite(
                    f"{debug_prefix}_{i}_{cls}.jpg",
                    zone_crop
                )

                print("OCR CALL:", cls)

                lines = self._ocr_zone(zone_crop)

                print("OCR RESULT:", lines)

                if not lines:
                    continue

                text = " ".join(lines).strip()

                if not text:
                    continue

                csv_field = ZONE_TO_FIELD.get(cls)
                print("ZONE CLASS:", cls)
                print("TEXT:", text)
                print("CSV FIELD:", csv_field)
                # если нет маппинга — складываем в additional_info
                if csv_field is None:
                    if cls not in ("price_tag", "zone_container"):  # исключаем шум
                        out["additional_info"] += f" {text}"
                    continue

                if cls in PRICE_ZONES:
                    out[csv_field] = self._extract_price(lines)

                elif cls == "zone_datetime":
                    out[csv_field] = text

                elif cls == "zone_wholesale_table":
                    out[csv_field] = " | ".join(lines)

                else:
                    out[csv_field] = text

            except Exception as e:
                print("ZONE ERROR:", e)

        cv2.imwrite(f"{debug_prefix}_zones.jpg", debug_img)
        print(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ПАЙПЛАЙН
# ═══════════════════════════════════════════════════════════════════════════

class PriceTagPipeline:
    def __init__(
        self,
        model_path: str,          # YOLO_CROPPER.pt — детекция ценников на кадре
        price_model_path: str,    # YOLO_PRICE.pt  — сегментация зон внутри кропа
        video_dir: str,
        output_dir: str,
        conf: float = 0.25,
        zone_conf: float = 0.25,
        min_height: int = 20,
        frame_step: int = 3,
        exclude_left_pct: float = 5.5,
        central_zone_pct: float = 30.0,
        camera_matrix=None,
        dist_coeffs=None,
        alpha: float = 1.0,
        undistort_before_yolo: bool = False,
        hash_threshold: int = 10,
        product_db: Optional[ProductDB] = None,
        save_crops: bool = True,
        use_degan: bool = False,
        zone_upscale: bool = True,
    ):
        base = Path(__file__).resolve().parent
        self.model = YOLO(
            str(model_path if Path(model_path).is_absolute() else base / model_path)
        )
        self.video_dir = Path(
            video_dir if Path(video_dir).is_absolute() else base / video_dir
        )
        self.output_dir = Path(
            output_dir if Path(output_dir).is_absolute() else base / output_dir
        )

        self.conf = conf
        self.min_height = min_height
        self.frame_step = frame_step
        self.exclude_left_pct = exclude_left_pct
        self.central_zone_pct = central_zone_pct
        self.undistort_before_yolo = undistort_before_yolo
        self.hash_threshold = hash_threshold
        self.save_crops = save_crops
        self.product_db = product_db
        self.use_degan = use_degan
        self.zone_upscale = zone_upscale
        self.degan: Optional[DeganEnhancer] = None
        if use_degan:
            self.degan = DeganEnhancer()

        # Параметры дисторсии
        self.camera_matrix = None
        self.dist_coeffs = None
        self.alpha = alpha
        self.undistort_maps = None

        if camera_matrix is not None and dist_coeffs is not None and undistort_before_yolo:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
            log.info("🔧 Undistort включён: применяется к полному кадру ДО YOLO")

        # Создаём директории
        (self.output_dir / "best_crops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "filtered_crops").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug_videos").mkdir(parents=True, exist_ok=True)
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

        self.tracker_yaml_path = self._create_custom_tracker()

        # ── Двухэтапный OCR ──
        # Сначала ZoneSegmentor (YOLO_PRICE), затем OCRModule использует его внутри
        price_model_full = str(
            price_model_path if Path(price_model_path).is_absolute()
            else base / price_model_path
        )
        zone_segmentor = ZoneSegmentor(price_model_full, conf=zone_conf)
        self.ocr = OCRModule(zone_segmentor, zone_upscale=zone_upscale)

    # ── настройка трекера ──────────────────────────────────────────────────

    def _create_custom_tracker(self) -> str:
        tracker_path = self.output_dir / "custom_tracker.yaml"
        config_str = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.5\n"
            "track_low_thresh: 0.1\n"
            "new_track_thresh: 0.6\n"
            "track_buffer: 120\n"
            "match_thresh: 0.8\n"
            "gmc_method: sparseOptFlow\n"
            "fuse_score: True\n"
            "mot20: False\n"
        )
        tracker_path.write_text(config_str, encoding="utf-8")
        log.info(f"⚙️  Создан конфиг трекера: {tracker_path}")
        return str(tracker_path)

    # ── undistort ──────────────────────────────────────────────────────────

    def _init_undistort_maps(self, frame_shape):
        h, w = frame_shape[:2]
        new_cam, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), self.alpha, (w, h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_cam, (w, h), cv2.CV_16SC2
        )
        return map1, map2

    # ── геометрические фильтры ─────────────────────────────────────────────

    def _is_in_central_zone(self, x1, y1, x2, y2, w, h) -> bool:
        cy = (y1 + y2) / 2
        margin = h * (self.central_zone_pct / 100) / 2
        return margin <= cy <= h - margin

    def _score_crop(self, crop_bgr: np.ndarray, in_central: bool) -> float:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape
        if ch < self.min_height:
            return 0.0
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_n = min(1.0, sharp / 800.0)
        scale_n = min(1.0, ch / 100.0)
        central_bonus = 0.15 if in_central else 0.0
        return 0.60 * sharp_n + 0.25 * scale_n + central_bonus

    # ── обработка одного видео ─────────────────────────────────────────────

    def process_video(self, video_path: Path):
        log.info(f"🎬 Обработка: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning(f"⛔ Не удалось открыть {video_path}")
            return {}, {}

        if (
            self.undistort_before_yolo
            and self.camera_matrix is not None
            and self.undistort_maps is None
        ):
            ret, test_frame = cap.read()
            if ret:
                self.undistort_maps = self._init_undistort_maps(test_frame.shape)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                log.info("✅ Карты undistort предвычислены")

        tracks: Dict[int, Dict] = {}
        track_status: Dict[int, Dict] = {}

        out = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % self.frame_step != 0:
                continue

            h, w, _ = frame.shape

            if self.undistort_maps is not None:
                map1, map2 = self.undistort_maps
                frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
                h, w, _ = frame.shape

            cutoff_x = w * (self.exclude_left_pct / 100.0)
            timestamp_ms = int((frame_idx / fps) * 1000)

            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                verbose=False,
                tracker=self.tracker_yaml_path,
            )
            debug_frame = frame.copy()

            if results[0].boxes is not None and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0])
                    conf_val = float(box.conf[0])
                    center_x = (x1 + x2) / 2

                    is_filtered = center_x <= cutoff_x
                    in_central = self._is_in_central_zone(x1, y1, x2, y2, w, h)

                    if track_id not in track_status:
                        track_status[track_id] = {"ever_filtered": False}
                    if is_filtered:
                        track_status[track_id]["ever_filtered"] = True

                    # ── КРОП из того же кадра, на котором детектила YOLO ──
                    # Координаты bbox и кроп формируются в одном месте —
                    # гарантируем соответствие crop ↔ (x_min, y_min, x_max, y_max)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    score = self._score_crop(crop, in_central)
                    cur_val = score * conf_val

                    if (
                        track_id not in tracks
                        or cur_val > tracks[track_id]["score"] * tracks[track_id]["conf"]
                    ):
                        tracks[track_id] = {
                            "score": score,
                            "crop": crop.copy(),          # BGR numpy-кроп в памяти
                            "vid": video_path.stem,
                            "frame": frame_idx,
                            "conf": conf_val,
                            "timestamp_ms": timestamp_ms,
                            # ── ключевые координаты — не теряем связь с кропом ──
                            "bbox": (x1, y1, x2, y2),
                            "in_central": in_central,
                            "is_filtered": is_filtered,
                        }

                    # Отрисовка дебага
                    if track_status[track_id]["ever_filtered"]:
                        color, label = (0, 0, 255), f"ID:{track_id} [REJ]"
                    elif is_filtered:
                        color, label = (0, 0, 255), f"ID:{track_id} [X]"
                    else:
                        color, label = (0, 255, 0), f"ID:{track_id}" + (" ★" if in_central else "")

                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        debug_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )

            # Дебаг-оверлей
            cv2.line(debug_frame, (int(cutoff_x), 0), (int(cutoff_x), h), (255, 0, 255), 2)
            cv2.putText(
                debug_frame, f"Left {self.exclude_left_pct}%",
                (int(cutoff_x) + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2,
            )
            cy1 = int(h * (1 - self.central_zone_pct / 100) / 2)
            cy2 = int(h * (1 + self.central_zone_pct / 100) / 2)
            cv2.line(debug_frame, (0, cy1), (w, cy1), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(debug_frame, (0, cy2), (w, cy2), (255, 255, 0), 1, cv2.LINE_AA)

            rotated = cv2.rotate(debug_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if out is None:
                debug_path = (
                    self.output_dir / "debug_videos" / f"{video_path.stem}_debug.mp4"
                )
                out = cv2.VideoWriter(
                    str(debug_path), fourcc, fps, (rotated.shape[1], rotated.shape[0])
                )
            out.write(rotated)

        cap.release()
        if out:
            out.release()

        return tracks, track_status

    # ── дедупликация + OCR + запись CSV ────────────────────────────────────

    def run(self):
        video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
        videos = [f for ext in video_exts for f in self.video_dir.glob(ext)]
        if not videos:
            log.warning("📂 Видео не найдены.")
            return

        # ── Шаг 1: собираем лучшие кропы по всем видео ──
        all_tracks: Dict[int, Dict] = {}
        all_statuses: Dict[int, Dict] = {}

        for vid in videos:
            self.undistort_maps = None
            vid_tracks, vid_statuses = self.process_video(vid)
            for tid in vid_tracks:
                cur = vid_tracks[tid]
                if (
                    tid not in all_tracks
                    or cur["score"] * cur["conf"]
                    > all_tracks[tid]["score"] * all_tracks[tid]["conf"]
                ):
                    all_tracks[tid] = cur
                if tid not in all_statuses:
                    all_statuses[tid] = vid_statuses[tid]
                elif vid_statuses[tid]["ever_filtered"]:
                    all_statuses[tid]["ever_filtered"] = True

        log.info(f"📊 До дедупликации найдено {len(all_tracks)} треков.")

        # ── Шаг 2: дедупликация pHash ──
        sorted_tracks = sorted(
            all_tracks.items(),
            key=lambda item: item[1]["score"] * item[1]["conf"],
            reverse=True,
        )

        seen_hashes: List[Any] = []
        duplicates_removed = 0
        unique_tracks: List[Tuple[int, Dict]] = []

        for track_id, best_det in sorted_tracks:
            crop_rgb = cv2.cvtColor(best_det["crop"], cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(crop_rgb)
            current_hash = imagehash.phash(pil_img)

            is_dup = any(
                (current_hash - sh) <= self.hash_threshold for sh in seen_hashes
            )
            if is_dup:
                duplicates_removed += 1
                continue

            seen_hashes.append(current_hash)
            unique_tracks.append((track_id, best_det))

        log.info(
            f"🛡️  Дубликатов удалено: {duplicates_removed}. "
            f"Уникальных кропов: {len(unique_tracks)}"
        )

        # ── Шаг 3: OCR + матчинг БД + запись CSV ──
        csv_rows: List[Dict] = []
        ocr_errors = 0

        for i, (track_id, best_det) in enumerate(unique_tracks, 1):
            is_rejected = all_statuses[track_id]["ever_filtered"]
            x1, y1, x2, y2 = best_det["bbox"]

            # Имя файла для сохранения кропа (если нужно)
            base_name = (
                f"{best_det['vid']}_f{best_det['frame']:04d}_track{track_id:03d}"
            )

            # Сохранение кропа (опционально)
            if self.save_crops:
                rotated_crop = cv2.rotate(best_det["crop"], cv2.ROTATE_90_COUNTERCLOCKWISE)
                if is_rejected:
                    cv2.imwrite(
                        str(self.output_dir / "filtered_crops" / f"{base_name}_REJECTED.png"),
                        rotated_crop,
                    )
                else:
                    cv2.imwrite(
                        str(self.output_dir / "best_crops" / f"{base_name}.png"),
                        rotated_crop,
                    )

            # Отклонённые треки пропускаем для OCR
            if is_rejected:
                continue

            # ── OCR: кроп → словарь полей ──
            log.info(f"  [{i}/{len(unique_tracks)}] OCR track_id={track_id} "
                     f"bbox=({x1},{y1},{x2},{y2})")
            try:
                rotated_crop = cv2.rotate(best_det['crop'], cv2.ROTATE_90_COUNTERCLOCKWISE)
                if self.degan is not None:
                    rotated_crop = self.degan.deblur_bgr(rotated_crop)
                    cv2.imwrite(
                        str(Path(DEBUG_DIR) / f"{base_name}_degan.jpg"),
                        rotated_crop,
                    )
                ocr_fields = self.ocr.recognize(rotated_crop)

            except Exception as e:
                log.error(f"  OCR failed (track {track_id}): {e}")
                ocr_fields = {f: "" for f in CSV_FIELDS}
                ocr_errors += 1

            # ── Матчинг БД товаров → barcode ──
            if self.product_db is not None:
                product_name = ocr_fields.get("product_name", "")
                matched_name, matched_code, match_score = self.product_db.match(product_name)
                if matched_name:
                    log.debug(
                        f"  DB match: '{product_name}' → '{matched_name}' "
                        f"(code={matched_code}, score={match_score})"
                    )
                    ocr_fields["barcode"] = matched_code or ocr_fields.get("barcode", "")
                else:
                    log.debug(f"  DB: нет совпадений для '{product_name}'")

            # ── Заполнение мета-полей (координаты, время, файл) ──
            # Эти поля пишутся ПОСЛЕ OCR и НЕ перезаписываются OCR
            row: Dict[str, Any] = {f: "" for f in CSV_FIELDS}
            row.update(ocr_fields)

            # Мета — жёстко привязаны к конкретному кропу
            row["filename"] = best_det["vid"]
            row["frame_timestamp"] = best_det["timestamp_ms"]
            row["x_min"] = x1
            row["y_min"] = y1
            row["x_max"] = x2
            row["y_max"] = y2

            # Поля, которых нет на ценнике → "нет"
            # (пустая строка = "не распознан", "нет" = "отсутствует на ценнике")
            # Простая эвристика: если поле пустое и это не координата/мета — ""
            # (пользователь может уточнить правило)

            csv_rows.append(row)

        # ── Запись CSV ──
        if csv_rows:
            csv_path = self.output_dir / "results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(csv_rows)
            log.info(f"📄 CSV сохранён: {csv_path} ({len(csv_rows)} строк)")
        else:
            log.warning("⚠️  Нет строк для записи в CSV.")

        if ocr_errors:
            log.warning(f"⚠️  OCR завершился с ошибкой для {ocr_errors} кропов.")

        log.info("✅ Готово.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_matrix_string(s: str) -> np.ndarray:
    try:
        return np.array(json.loads(s), dtype=np.float32)
    except Exception:
        import ast
        return np.array(ast.literal_eval(s), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="PriceTag Pipeline v5 — видео → двухэтапный YOLO → OCR → CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Основные параметры
    parser.add_argument("--cropper-model", default="./YOLO_CROPPER.pt",
                        help="Путь к YOLO_CROPPER.pt (детекция ценников на кадре)")
    parser.add_argument("--price-model",   default="./YOLO_PRICE.pt",
                        help="Путь к YOLO_PRICE.pt (сегментация зон внутри кропа)")
    parser.add_argument("--videos",        default="./input_videos",    help="Папка с видео")
    parser.add_argument("--output",        default="./results",         help="Папка для результатов")
    parser.add_argument("--conf",          type=float, default=0.25,    help="Порог уверенности YOLO_CROPPER")
    parser.add_argument("--zone-conf",     type=float, default=0.25,    help="Порог уверенности YOLO_PRICE (зоны)")
    parser.add_argument("--min-height",    type=int,   default=20,      help="Мин. высота кропа (px)")
    parser.add_argument("--frame-step",    type=int,   default=3,       help="Шаг кадров (1=каждый)")
    parser.add_argument("--exclude-left-pct", type=float, default=5.5,
                        help="Отсекать ценники в левых N%% кадра")
    parser.add_argument("--central-zone-pct", type=float, default=30.0,
                        help="Центральная зона по вертикали (%%)")
    parser.add_argument("--hash-threshold",   type=int,   default=10,
                        help="Строгость pHash-дедупликации (0-64)")
    parser.add_argument("--no-save-crops", action="store_true",
                        help="Не сохранять кропы на диск (только CSV)")
    parser.add_argument("--use-degan", action="store_true",
                        help="DE-GAN deblur кропа ценника один раз перед YOLO_PRICE/OCR")
    parser.add_argument("--no-zone-upscale", action="store_true",
                        help="Отключить LANCZOS-апскейл мелких zone-кропов перед OCR")

    # База товаров
    parser.add_argument(
        "--product-db",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "CSV-файл базы товаров (поля: fullname, code). "
            "Если указан, распознанное наименование фаззи-матчится "
            "с fullname, а соответствующий code пишется в колонку barcode."
        ),
    )
    parser.add_argument("--db-top-k",    type=int, default=30,
                        help="Топ-K кандидатов TF-IDF для финального rapidfuzz-ранжирования")
    parser.add_argument("--db-min-score", type=int, default=60,
                        help="Минимальный score rapidfuzz [0-100] для записи матча")

    # Параметры дисторсии
    parser.add_argument("--camera-matrix",   type=str,   default=None)
    parser.add_argument("--dist-coeffs",     type=str,   default=None)
    parser.add_argument("--alpha",           type=float, default=1.0)
    parser.add_argument("--camera-params",   type=str,   default=None,
                        help="JSON-файл с ключами 'matrix' и 'distortion'")
    parser.add_argument("--undistort-before-yolo", action="store_true",
                        help="Применять undistort к полному кадру ДО YOLO")

    args = parser.parse_args()

    # Параметры камеры
    camera_matrix = None
    dist_coeffs = None
    if args.camera_params and Path(args.camera_params).exists():
        with open(args.camera_params, "r", encoding="utf-8") as f:
            params = json.load(f)
        camera_matrix = np.array(params["matrix"],     dtype=np.float32)
        dist_coeffs   = np.array(params["distortion"], dtype=np.float32)
        log.info(f"📁 Параметры камеры из {args.camera_params}")
    elif args.camera_matrix and args.dist_coeffs:
        camera_matrix = parse_matrix_string(args.camera_matrix)
        dist_coeffs   = np.array(json.loads(args.dist_coeffs), dtype=np.float32)
        log.info("📝 Параметры камеры из CLI")

    # База товаров
    product_db = None
    if args.product_db:
        db_path = Path(args.product_db)
        if not db_path.exists():
            log.error(f"❌ Файл БД товаров не найден: {db_path}")
            sys.exit(1)
        product_db = ProductDB(
            str(db_path),
            top_k=args.db_top_k,
            min_score=args.db_min_score,
        )

    if args.use_degan and not _DEGAN_OK:
        log.error(
            "❌ --use-degan: пакет degan недоступен. "
            "Используйте Python 3.10 и pip install -r requirements.txt"
        )
        sys.exit(1)

    PriceTagPipeline(
        model_path=args.cropper_model,
        price_model_path=args.price_model,
        video_dir=args.videos,
        output_dir=args.output,
        conf=args.conf,
        zone_conf=args.zone_conf,
        min_height=args.min_height,
        frame_step=args.frame_step,
        exclude_left_pct=args.exclude_left_pct,
        central_zone_pct=args.central_zone_pct,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        alpha=args.alpha,
        undistort_before_yolo=args.undistort_before_yolo,
        hash_threshold=args.hash_threshold,
        product_db=product_db,
        save_crops=not args.no_save_crops,
        use_degan=args.use_degan,
        zone_upscale=not args.no_zone_upscale,
    ).run()


if __name__ == "__main__":
    main()
