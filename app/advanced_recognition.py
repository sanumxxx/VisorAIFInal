"""
Усовершенствованный модуль распознавания изображений с использованием YOLOv8 и EasyOCR.
Обеспечивает высокую точность распознавания различных объектов с выделением контуров,
а также распознавание текста на русском и английском языках.
Включает функционал для сохранения метаданных в XML формате и поиск изображений.
"""

import os
import sys
import json
import cv2
import numpy as np
import logging
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import re
from datetime import datetime
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedRecognizer:
    """
    Класс для распознавания объектов и текста на изображениях с использованием YOLOv8 и EasyOCR.
    Поддерживает выделение объектов с помощью масок и контуров.
    Включает функционал для сохранения метаданных изображений и поиска.
    """

    def __init__(self,
                 model_size='m',
                 confidence_threshold=0.25,
                 device=None,
                 models_path=None,
                 special_classes=None,
                 ocr_languages=None,
                 ocr_enabled=True,
                 use_masks=True,
                 use_image_enhancement=True,
                 mask_threshold=0.5,
                 mask_refinement=True):
        """
        Инициализация распознавателя.

        Args:
            model_size (str): Размер модели YOLOv8 ('n', 's', 'm', 'l', 'x')
            confidence_threshold (float): Порог уверенности для обнаружения (0.0-1.0)
            device (str): Устройство для вычислений ('cpu', 'cuda', '0')
            models_path (str): Путь для хранения моделей
            special_classes (list): Список классов, требующих особой обработки
            ocr_languages (list): Список языков для OCR (['ru', 'en'] по умолчанию)
            ocr_enabled (bool): Включить распознавание текста
            use_masks (bool): Использовать маски вместо прямоугольников
            use_image_enhancement (bool): Использовать улучшение изображения перед распознаванием
            mask_threshold (float): Пороговое значение для бинаризации масок (0.0-1.0)
            mask_refinement (bool): Применять дополнительное уточнение масок
        """
        self.confidence_threshold = confidence_threshold
        self.models_path = models_path or os.path.join(os.getcwd(), "models")
        self.device = device or 'cuda:0' if self._cuda_available() else 'cpu'
        self.model_size = model_size
        self.special_classes = special_classes or ['cat', 'dog', 'person']
        self.ocr_languages = ocr_languages or ['ru', 'en']
        self.ocr_enabled = ocr_enabled
        self.use_masks = use_masks
        self.use_image_enhancement = use_image_enhancement
        self.mask_threshold = mask_threshold
        self.mask_refinement = mask_refinement

        # Создаем директорию для моделей
        os.makedirs(self.models_path, exist_ok=True)

        # Настройка окружения для моделей
        os.environ["EASYOCR_MODULE_PATH"] = os.path.join(self.models_path, "easyocr")
        os.makedirs(os.environ["EASYOCR_MODULE_PATH"], exist_ok=True)

        logger.info(f"Инициализация модели YOLOv8{model_size} на устройстве {self.device}")
        logger.info(f"Режим выделения: {'маски объектов' if use_masks else 'прямоугольники'}")
        logger.info(f"Улучшение изображения: {'включено' if use_image_enhancement else 'выключено'}")
        logger.info(f"Уточнение масок: {'включено' if mask_refinement else 'выключено'}")

        # Установка ultralytics, если не установлена
        try:
            import ultralytics
            from ultralytics import YOLO
            logger.info(f"Используется установленный ultralytics версии {ultralytics.__version__}")
        except ImportError:
            logger.warning("Библиотека ultralytics не установлена. Попытка установки...")
            self._install_ultralytics()

            # Повторная попытка импорта
            try:
                import ultralytics
                from ultralytics import YOLO
                logger.info(f"Ultralytics успешно установлен, версия {ultralytics.__version__}")
            except ImportError:
                logger.error("Не удалось установить ultralytics. Пожалуйста, установите вручную: pip install ultralytics")
                raise ImportError("Не удалось установить необходимую библиотеку ultralytics")

        # Инициализация модели YOLOv8 для сегментации
        try:
            self._initialize_model()
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели: {e}")
            raise RuntimeError(f"Не удалось инициализировать модель: {e}")

        # Инициализация OCR, если включено
        if self.ocr_enabled:
            try:
                self._initialize_ocr()
            except Exception as e:
                logger.error(f"Ошибка при инициализации OCR: {e}")
                logger.warning("Распознавание текста будет отключено")
                self.ocr_enabled = False

    def _cuda_available(self):
        """Проверка доступности CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _install_ultralytics(self):
        """Установка библиотеки ultralytics"""
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            return True
        except subprocess.CalledProcessError:
            return False

    def _install_easyocr(self):
        """Установка библиотеки easyocr"""
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr"])
            return True
        except subprocess.CalledProcessError:
            return False

    def _initialize_model(self):
        """Инициализация модели YOLOv8 для сегментации"""
        from ultralytics import YOLO

        # Выбор модели сегментации (добавляем -seg к имени модели)
        if self.use_masks:
            model_name = f"yolov8{self.model_size}-seg"
        else:
            model_name = f"yolov8{self.model_size}"

        # Путь к локальной модели
        local_model_path = os.path.join(self.models_path, f"{model_name}.pt")

        # Проверяем наличие локальной модели
        if os.path.exists(local_model_path):
            logger.info(f"Загрузка локальной модели {model_name} из {local_model_path}")
            self.model = YOLO(local_model_path)
        else:
            # Загрузка модели из интернета
            logger.info(f"Загрузка модели {model_name} из репозитория ultralytics")
            self.model = YOLO(model_name)

            # Сохранение модели локально
            try:
                import shutil
                model_file = self.model.ckpt_path
                logger.info(f"Сохранение модели локально в {local_model_path}")
                shutil.copy2(model_file, local_model_path)
                logger.info("Модель успешно сохранена локально")
            except Exception as save_err:
                logger.warning(f"Не удалось сохранить модель локально: {save_err}")

        # Установка устройства
        self.model.to(self.device)

    def _initialize_ocr(self):
        """Инициализация модели распознавания текста"""
        try:
            import easyocr
            logger.info(f"Инициализация EasyOCR для языков: {', '.join(self.ocr_languages)}")

            # Создаем reader с указанными языками
            gpu = self.device != 'cpu'
            self.reader = easyocr.Reader(self.ocr_languages, gpu=gpu, model_storage_directory=os.environ["EASYOCR_MODULE_PATH"])
            logger.info("EasyOCR успешно инициализирован")

        except ImportError:
            logger.warning("Библиотека easyocr не установлена. Попытка установки...")
            self._install_easyocr()

            # Повторная попытка импорта
            try:
                import easyocr
                self.reader = easyocr.Reader(self.ocr_languages, gpu=(self.device != 'cpu'), model_storage_directory=os.environ["EASYOCR_MODULE_PATH"])
                logger.info("EasyOCR успешно инициализирован")
            except ImportError:
                logger.error("Не удалось установить easyocr. Пожалуйста, установите вручную: pip install easyocr")
                raise ImportError("Не удалось установить необходимую библиотеку easyocr")
            except Exception as e:
                logger.error(f"Ошибка при инициализации EasyOCR: {e}")
                raise RuntimeError(f"Не удалось инициализировать EasyOCR: {e}")

    def _enhance_image(self, image_path):
        """
        Улучшение качества изображения перед распознаванием.

        Args:
            image_path (str): Путь к изображению

        Returns:
            ndarray: Улучшенное изображение
            str: Путь к временному сохраненному изображению
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Не удалось прочитать изображение: {image_path}")
            return None, image_path

        # Применение методов улучшения изображения
        try:
            # Конвертация в LAB цветовое пространство для улучшения контраста
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # Объединение каналов
            enhanced_lab = cv2.merge((cl, a, b))

            # Конвертация обратно в BGR
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Увеличение резкости
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

            # Сохранение улучшенного изображения во временный файл
            temp_path = f"{os.path.splitext(image_path)[0]}_enhanced{os.path.splitext(image_path)[1]}"
            cv2.imwrite(temp_path, enhanced_image)

            logger.info(f"Изображение улучшено и сохранено в: {temp_path}")
            return enhanced_image, temp_path

        except Exception as e:
            logger.error(f"Ошибка при улучшении изображения: {e}")
            return image, image_path

    def _refine_mask(self, mask, orig_shape):
        """
        Улучшение качества маски для более точного выделения объекта.

        Args:
            mask (ndarray): Исходная маска
            orig_shape (tuple): Размеры исходного изображения (высота, ширина)

        Returns:
            ndarray: Улучшенная маска
            list: Контур маски
        """
        height, width = orig_shape

        # Преобразование маски в формат 8-бит для обработки
        if mask.max() <= 1.0:
            binary_mask = (mask > self.mask_threshold).astype(np.uint8) * 255
        else:
            binary_mask = (mask > self.mask_threshold * 255).astype(np.uint8)

        # Удаление шума и заполнение дырок в маске
        if self.mask_refinement:
            # Применение морфологических операций
            kernel = np.ones((5, 5), np.uint8)

            # Закрытие (closing) - удаляет маленькие черные области
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Открытие (opening) - удаляет маленькие белые области
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Дилатация для немного расширения маски
            binary_mask = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Поиск контуров на бинарной маске
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,  # Только внешние контуры
            cv2.CHAIN_APPROX_TC89_L1  # Более точный алгоритм аппроксимации
        )

        if not contours:
            return binary_mask, []

        # Получение только крупных контуров (отфильтровывание мелких артефактов)
        min_contour_area = 50  # Минимальная площадь в пикселях
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

        if not valid_contours:
            return binary_mask, []

        # Сортировка контуров по площади (от большего к меньшему)
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        # Берем самый большой контур
        largest_contour = sorted_contours[0]

        # Создаем маску только с самым большим контуром
        refined_mask = np.zeros_like(binary_mask)
        cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)

        # Применяем аппроксимацию контура для сглаживания
        # Параметр epsilon определяет точность аппроксимации:
        # меньшее значение = больше точек = более точный контур
        epsilon = 0.0005 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Нормализуем координаты контура (от 0 до 1)
        normalized_contour = approx_contour.reshape(-1, 2).astype(float)
        normalized_contour[:, 0] /= width
        normalized_contour[:, 1] /= height

        return refined_mask, normalized_contour.tolist()

    def _process_mask(self, mask, orig_shape, bbox=None):
        """
        Обработка маски с учетом bbox для лучшего выравнивания по объекту.

        Args:
            mask (ndarray): Исходная маска
            orig_shape (tuple): Размеры изображения (высота, ширина)
            bbox (list): Координаты ограничивающей рамки [x1, y1, x2, y2] в диапазоне 0-1

        Returns:
            list: Нормализованный контур
        """
        height, width = orig_shape

        # Улучшение маски
        refined_mask, contour = self._refine_mask(mask, orig_shape)

        if not contour and bbox:
            # Если контур не найден, но есть bbox, создаем контур из bbox
            x1, y1, x2, y2 = bbox
            # Преобразуем нормализованные координаты в пиксельные
            x1_px, y1_px = int(x1 * width), int(y1 * height)
            x2_px, y2_px = int(x2 * width), int(y2 * height)

            # Создаем прямоугольный контур
            rect_contour = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]
            return rect_contour

        # Если есть bbox, проверяем и корректируем контур, чтобы он соответствовал bbox
        if contour and bbox:
            x1, y1, x2, y2 = bbox

            # Проверяем, не выходит ли контур за пределы bbox
            # Если выходит, обрезаем его или расширяем bbox

            # Находим минимальные и максимальные координаты контура
            min_x = min(point[0] for point in contour)
            min_y = min(point[1] for point in contour)
            max_x = max(point[0] for point in contour)
            max_y = max(point[1] for point in contour)

            # Если контур существенно меньше bbox, возможно это ошибка сегментации
            # В этом случае расширяем контур до границ bbox
            contour_area = (max_x - min_x) * (max_y - min_y)
            bbox_area = (x2 - x1) * (y2 - y1)

            if contour_area < 0.5 * bbox_area:
                # Контур слишком мал относительно bbox, расширяем его
                expanded_contour = []
                for point in contour:
                    # Масштабируем точки контура для лучшего соответствия bbox
                    scaled_x = x1 + (point[0] - min_x) * (x2 - x1) / (max_x - min_x)
                    scaled_y = y1 + (point[1] - min_y) * (y2 - y1) / (max_y - min_y)
                    expanded_contour.append([scaled_x, scaled_y])
                return expanded_contour

        return contour

    def recognize(self, image_path, include_raw=False, enhanced_detection=True, detect_text=True):
        """
        Распознавание объектов и текста на изображении.

        Args:
            image_path (str): Путь к изображению
            include_raw (bool): Включать ли исходные результаты в ответ
            enhanced_detection (bool): Использовать ли улучшенное обнаружение
            detect_text (bool): Распознавать ли текст на изображении

        Returns:
            dict: Результаты распознавания
        """
        logger.info(f"Распознавание изображения: {image_path}")

        # Проверка существования файла
        if not os.path.exists(image_path):
            return {'error': f"Файл не найден: {image_path}"}

        # Предобработка изображения для улучшения качества
        original_image_path = image_path
        processed_image = None

        if self.use_image_enhancement:
            processed_image, temp_image_path = self._enhance_image(image_path)
            if processed_image is not None:
                image_path = temp_image_path

        try:
            # Запуск распознавания с помощью YOLOv8
            results = self.model(image_path, conf=self.confidence_threshold, verbose=False)

            # Структура для возврата результатов
            structured_result = {
                'objects': [],
                'labels': [],
                'text_blocks': [],
                'all_text': "",
                'processing_info': {
                    'model': f"YOLOv8{self.model_size}" + ("-seg" if self.use_masks else ""),
                    'device': self.device,
                    'confidence_threshold': self.confidence_threshold,
                    'image_enhanced': self.use_image_enhancement,
                    'mask_refinement': self.mask_refinement,
                    'ocr_enabled': self.ocr_enabled and detect_text,
                    'ocr_languages': self.ocr_languages if self.ocr_enabled and detect_text else [],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            # Добавляем исходные результаты, если требуется
            if include_raw:
                structured_result['raw_results'] = results

            # Обработка результатов распознавания объектов
            unique_classes = set()

            for result in results:
                # Получаем размеры изображения
                height, width = result.orig_shape

                # Обработка обнаруженных объектов
                for i, (box, cls_id, conf) in enumerate(zip(result.boxes.xyxy,
                                                        result.boxes.cls,
                                                        result.boxes.conf)):

                    # Получение класса и уверенности
                    cls_name = result.names[int(cls_id)]
                    confidence = float(conf)

                    # Получение координат границ
                    x1, y1, x2, y2 = box.cpu().numpy()

                    # Нормализация координат к диапазону 0-1
                    bbox = [
                        float(x1) / width,
                        float(y1) / height,
                        float(x2) / width,
                        float(y2) / height
                    ]

                    # Создание объекта с базовой информацией
                    obj = {
                        'class': cls_name,
                        'confidence': float(confidence),
                        'bbox': bbox,
                        'mask_applied': False
                    }

                    # Если используются маски и они доступны в результатах, добавляем их
                    if self.use_masks and hasattr(result, 'masks') and result.masks is not None:
                        try:
                            # Получаем маску для текущего объекта, если она есть
                            if i < len(result.masks):
                                mask = result.masks[i].data[0].cpu().numpy()

                                # Обработка маски для получения уточненного контура
                                contour = self._process_mask(mask, (height, width), bbox)

                                if contour:
                                    obj['contour'] = contour
                                    obj['mask_applied'] = True

                                    # Если включено отладочное сохранение масок
                                    if include_raw:
                                        # Сохраняем маску для отладки
                                        obj['mask_info'] = {
                                            'refined': True,
                                            'contour_points': len(contour)
                                        }
                        except Exception as mask_err:
                            logger.warning(f"Ошибка при обработке маски для объекта {cls_name}: {mask_err}")

                    # Добавление объекта в результаты
                    structured_result['objects'].append(obj)

                    # Добавление класса в уникальные
                    unique_classes.add(cls_name)

            # Добавление уникальных классов как меток
            for cls_name in unique_classes:
                structured_result['labels'].append(cls_name)

            # Улучшенное распознавание для специальных классов
            if enhanced_detection and any(cls in unique_classes for cls in self.special_classes):
                # Применяем дополнительное распознавание для повышения точности
                enhanced_result = self._enhance_detection(image_path, structured_result)
                structured_result = enhanced_result

            # Распознавание текста, если включено
            if self.ocr_enabled and detect_text:
                text_result = self._recognize_text(image_path)
                structured_result['text_blocks'] = text_result['text_blocks']
                structured_result['all_text'] = text_result['all_text']

            logger.info(f"Распознано объектов: {len(structured_result['objects'])}")
            if self.ocr_enabled and detect_text:
                logger.info(f"Распознано текстовых блоков: {len(structured_result['text_blocks'])}")

            # Если было использовано временное изображение, удаляем его
            if self.use_image_enhancement and original_image_path != image_path:
                try:
                    os.remove(image_path)
                    logger.info(f"Временное изображение удалено: {image_path}")
                except Exception as rm_err:
                    logger.warning(f"Не удалось удалить временное изображение: {rm_err}")

            # Добавляем путь к исходному изображению
            structured_result['original_image'] = original_image_path

            return structured_result

        except Exception as e:
            logger.error(f"Ошибка при распознавании: {e}")

            # Если было использовано временное изображение, удаляем его при ошибке
            if self.use_image_enhancement and original_image_path != image_path:
                try:
                    os.remove(image_path)
                except Exception:
                    pass

            return {'error': str(e)}

    def _enhance_detection(self, image_path, base_result):
        """
        Улучшенное распознавание для специальных классов.

        Args:
            image_path (str): Путь к изображению
            base_result (dict): Базовые результаты распознавания

        Returns:
            dict: Улучшенные результаты распознавания
        """
        # Проверяем, есть ли специальные классы в результатах
        special_objects = [obj for obj in base_result['objects']
                          if obj['class'] in self.special_classes]

        if not special_objects:
            return base_result

        # Для каждого специального объекта
        for obj in special_objects:
            # Помечаем объект как улучшенный
            obj['enhanced'] = True

            # Если маска не была применена, но нужна для специального класса
            if not obj.get('mask_applied', False) and self.use_masks:
                # Загружаем изображение
                image = cv2.imread(image_path)
                if image is None:
                    continue

                height, width = image.shape[:2]

                # Получаем координаты bbox в пикселях
                bbox = obj['bbox']
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)

                # Создаем маску на основе bbox
                mask = np.zeros((height, width), dtype=np.uint8)

                # Для человека используем эллипс вместо прямоугольника
                if obj['class'] == 'person':
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                # Для животных используем прямоугольник с закругленными углами
                elif obj['class'] in ['cat', 'dog']:
                    # Закругление углов
                    corner_radius = min((x2 - x1) // 4, (y2 - y1) // 4)

                    # Рисуем прямоугольник с закругленными углами (эмуляция)
                    cv2.rectangle(mask, (x1 + corner_radius, y1), (x2 - corner_radius, y1 + corner_radius), 255, -1)
                    cv2.rectangle(mask, (x1, y1 + corner_radius), (x2, y2 - corner_radius), 255, -1)
                    cv2.rectangle(mask, (x1 + corner_radius, y2 - corner_radius), (x2 - corner_radius, y2), 255, -1)

                    # Добавляем закругленные углы
                    cv2.circle(mask, (x1 + corner_radius, y1 + corner_radius), corner_radius, 255, -1)
                    cv2.circle(mask, (x2 - corner_radius, y1 + corner_radius), corner_radius, 255, -1)
                    cv2.circle(mask, (x1 + corner_radius, y2 - corner_radius), corner_radius, 255, -1)
                    cv2.circle(mask, (x2 - corner_radius, y2 - corner_radius), corner_radius, 255, -1)
                else:
                    # Для других объектов просто используем прямоугольник
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                # Применяем обработку маски
                contour = self._process_mask(mask, (height, width), bbox)

                if contour:
                    obj['contour'] = contour
                    obj['mask_applied'] = True
                    obj['mask_method'] = 'bbox_based'

            # Расширяем границы для лучшего охвата животного
            if obj['class'] in ['cat', 'dog']:
                # Расширение границ для животных
                bbox = obj['bbox']

                # Расширяем границы на 15% во все стороны для животных
                width_obj = bbox[2] - bbox[0]
                height_obj = bbox[3] - bbox[1]

                bbox[0] = max(0, bbox[0] - width_obj * 0.15)
                bbox[1] = max(0, bbox[1] - height_obj * 0.15)
                bbox[2] = min(1.0, bbox[2] + width_obj * 0.15)
                bbox[3] = min(1.0, bbox[3] + height_obj * 0.15)

                obj['bbox'] = bbox

                # Если есть контур, также расширяем его
                if 'contour' in obj:
                    # Находим центр контура
                    contour_points = np.array(obj['contour'])
                    center_x = np.mean(contour_points[:, 0])
                    center_y = np.mean(contour_points[:, 1])

                    # Расширяем контур от центра
                    expanded_contour = []
                    for point in obj['contour']:
                        # Вектор от центра к точке
                        vec_x = point[0] - center_x
                        vec_y = point[1] - center_y

                        # Расширяем на 15%
                        new_x = center_x + vec_x * 1.15
                        new_y = center_y + vec_y * 1.15

                        # Ограничиваем значения от 0 до 1
                        new_x = max(0, min(1, new_x))
                        new_y = max(0, min(1, new_y))

                        expanded_contour.append([new_x, new_y])

                    obj['contour'] = expanded_contour
                    obj['contour_expanded'] = True

        # Для кошек и собак можно применить дополнительное распознавание породы
        # Это можно реализовать через отдельную модель для классификации

        return base_result

    # -------- УЛУЧШЕННЫЕ МЕТОДЫ ОБРАБОТКИ ТЕКСТА --------

    def _process_recognized_text(self, text):
        """
        Обрабатывает распознанный текст: удаляет лишние символы, исправляет опечатки.

        Args:
            text (str): Исходный распознанный текст

        Returns:
            str: Обработанный текст
        """
        if not text:
            return ""

        # Предварительная обработка
        processed_text = text.strip()

        # Удаление лишних символов
        processed_text = self._remove_extra_symbols(processed_text)

        # Нормализация пробелов
        processed_text = self._normalize_spaces(processed_text)

        # Исправление распространенных ошибок распознавания
        processed_text = self._fix_common_ocr_errors(processed_text)

        # Коррекция опечаток на основе словаря (если доступно)
        processed_text = self._spelling_correction(processed_text)

        # Нормализация регистра
        processed_text = self._normalize_case(processed_text)

        return processed_text

    def _remove_extra_symbols(self, text):
        """
        Удаляет лишние и некорректные символы из текста.

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст без лишних символов
        """
        # Удаление непечатаемых символов
        text = ''.join(char for char in text if char.isprintable())

        # Удаление повторяющихся знаков пунктуации
        text = re.sub(r'([.!?,;:])\1+', r'\1', text)

        # Удаление символов, которые часто являются ошибками распознавания
        # (специфические символы, которые редко используются в обычном тексте)
        text = re.sub(r'[\\|@#$%^&*<>{}[\]~`]', ' ', text)

        # Сохранение важных символов: точка, запятая, дефис, апостроф и т.д.
        # но удаление случайных символов между буквами
        clean_text = ""
        prev_is_alpha = False

        for i, char in enumerate(text):
            if char.isalnum() or char.isspace():
                clean_text += char
                prev_is_alpha = char.isalnum()
            elif char in '.,;:!?-\'\"()' and (prev_is_alpha or i == 0 or text[i-1].isspace()):
                # Сохраняем знаки пунктуации только в правильном контексте
                clean_text += char
                prev_is_alpha = False
            else:
                # Заменяем другие символы пробелом если они не рядом с пробелом
                if i > 0 and not text[i-1].isspace() and i < len(text)-1 and not text[i+1].isspace():
                    clean_text += ' '
                prev_is_alpha = False

        return clean_text

    def _normalize_spaces(self, text):
        """
        Нормализует пробелы в тексте.

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст с нормализованными пробелами
        """
        # Заменяем повторяющиеся пробелы на один
        text = re.sub(r'\s+', ' ', text)

        # Удаляем пробелы перед знаками пунктуации
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)

        # Добавляем пробелы после знаков пунктуации, если за ними следует буква или цифра
        text = re.sub(r'([.,;:!?])([a-zA-Zа-яА-Я0-9])', r'\1 \2', text)

        # Удаляем пробелы после открывающей скобки и перед закрывающей
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        return text.strip()

    def _fix_common_ocr_errors(self, text):
        """
        Исправляет распространенные ошибки OCR.

        Args:
            text (str): Исходный текст

        Returns:
            str: Исправленный текст
        """
        # Словарь распространенных замен (ошибка: правильное)
        common_errors = {
            # Латинские буквы вместо кириллических и наоборот
            'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'b': 'в',
            # Цифры вместо букв и наоборот
            '0': 'O', 'O': 'О', 'о': 'o', '1': 'I', 'I': 'І', '5': 'S', '8': 'B',
            # Символы, которые часто путают
            'rn': 'm', 'vv': 'w', 'ТТ': 'П', 'п': 'n', 'и': 'n',
            # Другие распространенные ошибки
            'l': 'I', ']': 'l', '[': 'l', '|': 'I',
        }

        # Применяем замены только для отдельных символов или коротких комбинаций
        # Это позволяет избежать неправильных замен в словах
        for error, correct in common_errors.items():
            # Заменяем символы только если они стоят отдельно или в начале/конце слова
            pattern = r'(\b|(?<=\s))' + re.escape(error) + r'(\b|(?=\s))'
            text = re.sub(pattern, correct, text)

        # Дополнительные специфические исправления
        # Например, исправление распространенных ошибок в названиях

        return text

    def _spelling_correction(self, text):
        """
        Применяет коррекцию орфографии с использованием словаря.
        Использует базовый алгоритм расстояния Левенштейна.

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст с исправленными словами
        """
        try:
            # Если установлен модуль для проверки орфографии, используем его
            from spellchecker import SpellChecker

            # Определяем язык текста (русский или английский)
            if any(ord(char) > 127 for char in text):
                # Текст содержит не ASCII символы, вероятно русский
                lang = 'ru'
            else:
                # Вероятно английский
                lang = 'en'

            try:
                spell = SpellChecker(language=lang)

                # Разбиваем текст на слова и исправляем каждое
                words = text.split()
                corrected_words = []

                for word in words:
                    # Исправляем только слова длиной более 3 символов
                    if len(word) > 3 and word.isalpha():
                        corrected = spell.correction(word)
                        if corrected:
                            corrected_words.append(corrected)
                        else:
                            corrected_words.append(word)
                    else:
                        corrected_words.append(word)

                return ' '.join(corrected_words)

            except Exception as e:
                logger.warning(f"Ошибка при проверке орфографии: {e}")
                return text

        except ImportError:
            # Если модуль не установлен, реализуем простую проверку
            # на основе расстояния Левенштейна для часто используемых слов

            # Простой словарь часто используемых слов
            common_words = {
                'книга': ['кпига', 'книrа', 'кннга', 'kнига', 'книгa'],
                'человек': ['челoвек', 'человеk', 'чeловек', 'человек'],
                'магазин': ['мaгазин', 'магaзин', 'магазuн', 'мaгaзин'],
                'телефон': ['телефoн', 'телeфон', 'телефoн', 'тeлефон'],
                'компьютер': ['компютер', 'kомпьютер', 'компьюmер', 'компьтер'],
                'book': ['bo0k', 'b00k', 'boоk', 'b0ok'],
                'phone': ['рhone', 'ph0ne', 'phonе', 'рhоnе'],
                'store': ['st0re', 'storе', 'stor', 'st0rе'],
                'computer': ['computеr', 'сomputer', 'c0mputer', 'computеr']
            }

            # Заменяем известные ошибки
            words = text.split()
            corrected_words = []

            for word in words:
                corrected = word

                # Проверяем, есть ли слово в словаре известных ошибок
                for correct, errors in common_words.items():
                    if word.lower() in errors:
                        corrected = correct
                        break

                corrected_words.append(corrected)

            return ' '.join(corrected_words)

    def _normalize_case(self, text):
        """
        Нормализует регистр текста.

        Args:
            text (str): Исходный текст

        Returns:
            str: Текст с нормализованным регистром
        """
        if not text:
            return ""

        # Если весь текст в верхнем регистре, преобразуем в нижний
        if text.isupper():
            return text.lower()

        # Проверяем наличие случайных заглавных букв в середине слов
        words = text.split()
        normalized_words = []

        for word in words:
            if len(word) <= 1:
                normalized_words.append(word)
                continue

            # Если в середине слова есть заглавные буквы, нормализуем
            if any(c.isupper() for c in word[1:]):
                # Сохраняем первую букву в исходном регистре
                first_char = word[0]
                rest = word[1:].lower()
                normalized_words.append(first_char + rest)
            else:
                normalized_words.append(word)

        # Собираем текст обратно
        normalized_text = ' '.join(normalized_words)

        # Делаем первую букву предложения заглавной
        if normalized_text and normalized_text[0].isalpha():
            normalized_text = normalized_text[0].upper() + normalized_text[1:]

        return normalized_text

    def _recognize_text(self, image_path):
        """
        Распознавание текста на изображении с помощью EasyOCR.

        Args:
            image_path (str): Путь к изображению

        Returns:
            dict: Словарь с распознанным текстом и его местоположением
        """
        logger.info(f"Распознавание текста на изображении: {image_path}")

        result = {
            'text_blocks': [],
            'all_text': ""
        }

        try:
            # Чтение изображения для получения размеров
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Не удалось загрузить изображение: {image_path}")
                return result

            height, width = image.shape[:2]

            # Распознавание текста с помощью EasyOCR
            detection_result = self.reader.readtext(image_path)

            all_texts = []

            for detection in detection_result:
                # Формат EasyOCR: [bbox, text, confidence]
                bbox, text, confidence = detection

                # Обработка распознанного текста
                processed_text = self._process_recognized_text(text)

                # Координаты в формате [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Преобразуем в нормализованные координаты [x1, y1, x3, y3]
                x_vals = [p[0] for p in bbox]
                y_vals = [p[1] for p in bbox]

                normalized_bbox = [
                    min(x_vals) / width,
                    min(y_vals) / height,
                    max(x_vals) / width,
                    max(y_vals) / height
                ]

                # Создаем нормализованный контур для текстового блока
                normalized_contour = []
                for point in bbox:
                    normalized_contour.append([point[0] / width, point[1] / height])

                # Добавляем текстовый блок с обработанным текстом
                result['text_blocks'].append({
                    'text': processed_text,
                    'original_text': text,  # Сохраняем оригинальный текст
                    'confidence': confidence,
                    'bbox': normalized_bbox,
                    'contour': normalized_contour
                })

                # Добавляем в общий текст
                if processed_text:
                    all_texts.append(processed_text)

            # Объединяем все тексты в один и применяем дополнительную обработку
            if all_texts:
                combined_text = " ".join(all_texts)
                # Финальная обработка объединенного текста
                result['all_text'] = self._normalize_spaces(combined_text)

            return result
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста: {e}")
            return result

    def visualize_results(self, image_path, results, output_path=None):
        """
        Визуализация результатов распознавания на изображении.

        Args:
            image_path (str): Путь к исходному изображению
            results (dict): Результаты распознавания
            output_path (str, optional): Путь для сохранения визуализации

        Returns:
            str: Путь к сохраненному изображению с визуализацией
        """
        if output_path is None:
            dir_name = os.path.dirname(image_path)
            base_name = os.path.basename(image_path)
            output_path = os.path.join(dir_name, f"visualized_{base_name}")

        # Проверка на ошибки в результатах
        if 'error' in results:
            logger.error(f"Невозможно визуализировать результаты с ошибкой: {results['error']}")
            return None

        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Не удалось загрузить изображение для визуализации: {image_path}")
                return None

            # Получаем размеры изображения
            height, width, _ = image.shape

            # Создаем маску для наложения на изображение
            overlay = image.copy()
            mask_overlay = np.zeros_like(image)

            # Визуализация обнаруженных объектов
            for obj in results.get('objects', []):
                try:
                    # Выбор цвета в зависимости от класса
                    if obj['class'] in self.special_classes:
                        color = (0, 255, 0)  # Зеленый для специальных классов
                    else:
                        color = (0, 165, 255)  # Оранжевый для остальных

                    # Если у объекта есть контур, используем его
                    if 'contour' in obj:
                        # Преобразуем нормализованные координаты в пиксели
                        contour_points = []
                        for point in obj['contour']:
                            x = int(point[0] * width)
                            y = int(point[1] * height)
                            contour_points.append([x, y])

                        contour = np.array(contour_points, dtype=np.int32)

                        # Рисуем контур на исходном изображении
                        cv2.polylines(image, [contour], True, color, 2)

                        # Создаем маску для полупрозрачной заливки
                        cv2.fillPoly(mask_overlay, [contour], color)

                        # Находим верхнюю точку для подписи
                        top_point = min(contour_points, key=lambda p: p[1])
                        label_x, label_y = top_point
                    else:
                        # Получаем координаты bbox в пикселях
                        bbox = obj['bbox']
                        x1 = int(bbox[0] * width)
                        y1 = int(bbox[1] * height)
                        x2 = int(bbox[2] * width)
                        y2 = int(bbox[3] * height)

                        # Рисуем прямоугольник
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                        # Создаем маску для полупрозрачной заливки
                        cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), color, -1)

                        # Координаты для подписи
                        label_x, label_y = x1, y1

                    # Рисуем метку с уверенностью
                    label = f"{obj['class']} ({obj['confidence']:.2f})"

                    # Фон для текста с увеличенным padding
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_bg_x1 = max(0, label_x - 2)
                    text_bg_y1 = max(0, label_y - 30)
                    text_bg_x2 = min(width, label_x + text_size[0] + 4)
                    text_bg_y2 = min(height, label_y)

                    cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

                    # Текст (белый для лучшей видимости)
                    cv2.putText(image, label, (label_x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as vis_err:
                    logger.warning(f"Ошибка при визуализации объекта: {vis_err}")

            # Визуализация распознанного текста
            for text_block in results.get('text_blocks', []):
                try:
                    # Цвет для текстовых блоков - красный
                    color = (0, 0, 255)

                    # Если у текстового блока есть контур, используем его
                    if 'contour' in text_block:
                        # Преобразуем нормализованные координаты в пиксели
                        contour_points = []
                        for point in text_block['contour']:
                            x = int(point[0] * width)
                            y = int(point[1] * height)
                            contour_points.append([x, y])

                        contour = np.array(contour_points, dtype=np.int32)

                        # Рисуем контур
                        cv2.polylines(image, [contour], True, color, 2)

                        # Находим нижнюю точку для подписи
                        bottom_right = max(contour_points, key=lambda p: (p[1], p[0]))
                        text_x, text_y = bottom_right
                    else:
                        # Получаем координаты bbox в пикселях
                        bbox = text_block['bbox']
                        x1 = int(bbox[0] * width)
                        y1 = int(bbox[1] * height)
                        x2 = int(bbox[2] * width)
                        y2 = int(bbox[3] * height)

                        # Рисуем прямоугольник
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                        # Координаты для подписи
                        text_x, text_y = x1, y2

                    # Получаем текст для отображения
                    display_text = text_block.get('text', '')  # Используем обработанный текст

                    # Ограничиваем длину для отображения
                    if len(display_text) > 20:
                        display_text = display_text[:17] + "..."

                    # Фон для текста
                    text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + 25), color, -1)

                    # Текст
                    cv2.putText(image, display_text, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                except Exception as text_vis_err:
                    logger.warning(f"Ошибка при визуализации текстового блока: {text_vis_err}")

            # Накладываем полупрозрачную маску на основное изображение
            alpha = 0.3  # Уровень прозрачности
            beta = 1.0 - alpha
            gamma = 0.0
            image = cv2.addWeighted(mask_overlay, alpha, image, beta, gamma)

            # Добавляем информацию о модели в левый нижний угол
            model_info = f"YOLOv8{self.model_size}-seg | {datetime.now().strftime('%Y-%m-%d')}"
            cv2.putText(image, model_info, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(image, model_info, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Сохраняем изображение
            cv2.imwrite(output_path, image)
            logger.info(f"Визуализация сохранена в: {output_path}")

            return output_path
        except Exception as e:
            logger.error(f"Ошибка при визуализации результатов: {e}")
            return None

    # -------- МЕТОДЫ ДЛЯ РАБОТЫ С МЕТАДАННЫМИ --------

    def generate_metadata(self, image_path, results):
        """
        Генерирует метаданные для изображения на основе результатов распознавания.

        Args:
            image_path (str): Путь к изображению
            results (dict): Результаты распознавания

        Returns:
            dict: Метаданные изображения
        """
        try:
            # Получаем только имя файла без пути
            image_name = os.path.basename(image_path)

            # Список уникальных объектов
            object_names = [obj['class'] for obj in results.get('objects', [])]
            unique_objects = list(dict.fromkeys(object_names))  # удаляем дубликаты с сохранением порядка

            # Получаем распознанный текст
            recognized_text = results.get('all_text', "").strip()

            # Формируем метаданные
            metadata = {
                'name': image_name,
                'objects': unique_objects,
                'text': recognized_text
            }

            logger.info(f"Метаданные созданы для {image_name}")
            return metadata

        except Exception as e:
            logger.error(f"Ошибка при создании метаданных: {e}")
            return None

    def save_metadata_to_xml(self, metadata_list, output_file):
        """
        Сохраняет список метаданных в XML-файл.

        Args:
            metadata_list (list): Список словарей с метаданными
            output_file (str): Путь для сохранения XML-файла

        Returns:
            bool: True если сохранение прошло успешно
        """
        try:
            # Создаем корневой элемент
            root = ET.Element("metadata")

            # Добавляем каждое изображение в метаданные
            for metadata in metadata_list:
                if not metadata or 'name' not in metadata:
                    continue

                # Создаем элемент для изображения
                image_elem = ET.SubElement(root, "image")
                image_elem.set("name", metadata['name'])

                # Добавляем объекты
                if 'objects' in metadata and metadata['objects']:
                    obj_elem = ET.SubElement(image_elem, "object")
                    obj_elem.text = " ".join(metadata['objects'])

                # Добавляем текст
                if 'text' in metadata and metadata['text']:
                    text_elem = ET.SubElement(image_elem, "text")
                    text_elem.text = metadata['text']

            # Преобразуем элемент в строку XML
            rough_string = ET.tostring(root, 'utf-8')

            # Делаем XML более читабельным с отступами
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")

            # Сохраняем в файл
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

            logger.info(f"Метаданные сохранены в файл: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении метаданных: {e}")
            return False

    def read_metadata_from_xml(self, input_file):
        """
        Читает метаданные из XML-файла.

        Args:
            input_file (str): Путь к XML-файлу

        Returns:
            list: Список словарей с метаданными или None при ошибке
        """
        try:
            if not os.path.exists(input_file):
                logger.error(f"Файл не найден: {input_file}")
                return None

            # Парсим XML-файл
            tree = ET.parse(input_file)
            root = tree.getroot()

            metadata_list = []

            # Обрабатываем каждый элемент image
            for image_elem in root.findall("image"):
                metadata = {
                    'name': image_elem.get("name"),
                    'objects': [],
                    'text': ""
                }

                # Получаем объекты
                obj_elem = image_elem.find("object")
                if obj_elem is not None and obj_elem.text:
                    metadata['objects'] = obj_elem.text.split()

                # Получаем текст
                text_elem = image_elem.find("text")
                if text_elem is not None and text_elem.text:
                    metadata['text'] = text_elem.text

                metadata_list.append(metadata)

            logger.info(f"Прочитано {len(metadata_list)} записей метаданных из {input_file}")
            return metadata_list

        except Exception as e:
            logger.error(f"Ошибка при чтении метаданных: {e}")
            return None

    def merge_metadata(self, metadata_file, new_metadata_list, output_file=None):
        """
        Объединяет существующие метаданные с новыми.

        Args:
            metadata_file (str): Путь к существующему файлу метаданных
            new_metadata_list (list): Список новых метаданных для добавления
            output_file (str, optional): Путь для сохранения результата.
                                        Если не указан, используется metadata_file

        Returns:
            bool: True если объединение выполнено успешно
        """
        try:
            # Определяем выходной файл
            if output_file is None:
                output_file = metadata_file

            # Читаем существующие метаданные, если файл существует
            existing_metadata = []
            if os.path.exists(metadata_file):
                existing_metadata = self.read_metadata_from_xml(metadata_file) or []

            # Создаем словарь для быстрого поиска по имени
            metadata_dict = {item['name']: item for item in existing_metadata}

            # Добавляем или обновляем метаданные
            for new_item in new_metadata_list:
                if not new_item or 'name' not in new_item:
                    continue

                # Если запись уже существует, обновляем её
                if new_item['name'] in metadata_dict:
                    metadata_dict[new_item['name']] = new_item
                else:
                    # Иначе добавляем новую запись
                    metadata_dict[new_item['name']] = new_item

            # Преобразуем обратно в список
            merged_metadata = list(metadata_dict.values())

            # Сохраняем результат
            return self.save_metadata_to_xml(merged_metadata, output_file)

        except Exception as e:
            logger.error(f"Ошибка при объединении метаданных: {e}")
            return False

    # -------- ФУНКЦИИ ПОИСКА ПО МЕТАДАННЫМ --------

    def search_images_by_metadata(self, query, metadata_file=None, results_limit=None):
        """
        Поиск изображений по метаданным (объектам и тексту).

        Args:
            query (str): Поисковый запрос
            metadata_file (str, optional): Путь к файлу метаданных
            results_limit (int, optional): Ограничение количества результатов

        Returns:
            list: Список найденных изображений с релевантностью
        """
        try:
            # Если не указан файл метаданных, используем путь по умолчанию
            if metadata_file is None:
                metadata_file = "metadata.xml"

            # Проверяем существование файла метаданных
            if not os.path.exists(metadata_file):
                logger.error(f"Файл метаданных не найден: {metadata_file}")
                return []

            # Загружаем метаданные
            metadata_list = self.read_metadata_from_xml(metadata_file)
            if not metadata_list:
                logger.warning("Метаданные пусты или повреждены")
                return []

            # Нормализуем запрос для поиска
            normalized_query = query.lower().strip()
            query_terms = normalized_query.split()

            # Результаты поиска в формате: (имя_изображения, релевантность, причина)
            search_results = []

            for metadata in metadata_list:
                image_name = metadata.get('name', '')
                objects = metadata.get('objects', [])
                text = metadata.get('text', '').lower()

                # Счетчик совпадений для оценки релевантности
                relevance_score = 0
                match_reasons = []

                # Поиск по объектам
                for term in query_terms:
                    # Проверяем, содержится ли термин в списке объектов
                    for obj in objects:
                        if term.lower() == obj.lower():
                            relevance_score += 3  # Полное совпадение объекта имеет высокий вес
                            match_reasons.append(f"Объект: {obj}")
                            break
                        elif term.lower() in obj.lower():
                            relevance_score += 1  # Частичное совпадение объекта
                            match_reasons.append(f"Частичное совпадение с объектом: {obj}")
                            break

                # Поиск по тексту
                for term in query_terms:
                    if not term:
                        continue

                    # Проверяем полное совпадение слова в тексте
                    text_words = text.split()
                    if term.lower() in text_words:
                        relevance_score += 2  # Полное совпадение слова в тексте
                        match_reasons.append(f"Текст содержит слово: {term}")
                    # Проверяем частичное совпадение в тексте
                    elif term.lower() in text:
                        relevance_score += 1  # Частичное совпадение в тексте
                        match_reasons.append(f"Текст содержит часть: {term}")

                # Если есть совпадения, добавляем в результаты
                if relevance_score > 0:
                    search_results.append({
                        'image_name': image_name,
                        'relevance': relevance_score,
                        'match_reasons': match_reasons,
                        'metadata': metadata
                    })

            # Сортируем результаты по релевантности (от большей к меньшей)
            search_results.sort(key=lambda x: x['relevance'], reverse=True)

            # Ограничиваем количество результатов, если задано
            if results_limit and len(search_results) > results_limit:
                search_results = search_results[:results_limit]

            return search_results

        except Exception as e:
            logger.error(f"Ошибка при поиске изображений: {e}")
            return []

    def find_similar_images(self, image_path, metadata_file=None, results_limit=10):
        """
        Находит изображения, похожие на заданное, используя метаданные.

        Args:
            image_path (str): Путь к образцу изображения
            metadata_file (str, optional): Путь к файлу метаданных
            results_limit (int): Максимальное количество результатов

        Returns:
            list: Список похожих изображений
        """
        try:
            # Сначала анализируем исходное изображение
            results = self.recognize(image_path)

            if 'error' in results:
                logger.error(f"Ошибка при распознавании исходного изображения: {results['error']}")
                return []

            # Формируем поисковый запрос из объектов и текста
            query_terms = []

            # Добавляем классы объектов
            for obj in results.get('objects', []):
                if 'class' in obj and obj['class'] not in query_terms:
                    query_terms.append(obj['class'])

            # Добавляем ключевые слова из текста (первые 5 слов)
            if 'all_text' in results and results['all_text']:
                text_words = results['all_text'].split()
                # Фильтруем короткие слова и добавляем до 5 длинных слов
                important_words = [word for word in text_words if len(word) > 3][:5]
                query_terms.extend(important_words)

            # Если запрос пустой, возвращаем пустой результат
            if not query_terms:
                logger.warning("Не удалось извлечь термины для поиска из изображения")
                return []

            # Формируем запрос из уникальных терминов
            query = " ".join(set(query_terms))
            logger.info(f"Поиск похожих изображений с запросом: {query}")

            # Выполняем поиск по метаданным
            return self.search_images_by_metadata(query, metadata_file, results_limit)

        except Exception as e:
            logger.error(f"Ошибка при поиске похожих изображений: {e}")
            return []

    def batch_process(self, image_dir, output_dir=None, extensions=None, metadata_file=None):
        """
        Пакетная обработка всех изображений в директории.

        Args:
            image_dir (str): Путь к директории с изображениями
            output_dir (str, optional): Путь для сохранения результатов
            extensions (list, optional): Список расширений файлов для обработки
            metadata_file (str, optional): Путь к файлу метаданных

        Returns:
            dict: Результаты обработки для каждого изображения
        """
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            return {'error': f"Директория не найдена: {image_dir}"}

        if output_dir is None:
            output_dir = os.path.join(image_dir, "results")

        os.makedirs(output_dir, exist_ok=True)

        extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        # Получаем список файлов изображений
        image_files = []
        for ext in extensions:
            image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
            image_files.extend(list(Path(image_dir).glob(f"*{ext.upper()}")))

        if not image_files:
            return {'error': f"В директории {image_dir} не найдено изображений с расширениями {extensions}"}

        logger.info(f"Найдено {len(image_files)} изображений для обработки")

        # Результаты обработки и метаданные
        results = {}
        all_metadata = []

        # Обработка каждого изображения
        for image_path in image_files:
            image_path_str = str(image_path)
            base_name = os.path.basename(image_path_str)

            # Распознавание
            result = self.recognize(image_path_str)

            # Визуализация
            vis_output_path = os.path.join(output_dir, f"vis_{base_name}")
            self.visualize_results(image_path_str, result, vis_output_path)

            # Сохранение результатов в JSON
            json_output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Генерация метаданных, если нет ошибок
            if 'error' not in result:
                metadata = self.generate_metadata(image_path_str, result)
                if metadata:
                    all_metadata.append(metadata)

            # Сохраняем результаты
            results[base_name] = {
                'recognition': result,
                'visualization': vis_output_path,
                'json': json_output_path
            }

        # Сохранение метаданных, если их нужно сохранить
        if metadata_file is not None and all_metadata:
            metadata_output = metadata_file
            if os.path.dirname(metadata_output) == "":
                # Если указано только имя файла без пути, сохраняем в output_dir
                metadata_output = os.path.join(output_dir, metadata_file)

            # Если существующий файл метаданных, объединяем с ним
            if os.path.exists(metadata_output):
                self.merge_metadata(metadata_output, all_metadata)
            else:
                # Иначе создаем новый файл
                self.save_metadata_to_xml(all_metadata, metadata_output)

            logger.info(f"Метаданные для {len(all_metadata)} изображений сохранены в: {metadata_output}")
        elif all_metadata:
            # Если файл не указан, но метаданные есть, сохраняем их в output_dir
            metadata_output = os.path.join(output_dir, "metadata.xml")
            self.save_metadata_to_xml(all_metadata, metadata_output)
            logger.info(f"Метаданные сохранены в: {metadata_output}")

        logger.info(f"Обработка завершена для {len(results)} изображений")
        return results


# Пример использования
if __name__ == "__main__":
    import argparse

    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Распознавание объектов и текста на изображениях с помощью YOLOv8')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению или директории для анализа')
    parser.add_argument('--batch', action='store_true', help='Обработать все изображения в директории')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения результатов')
    parser.add_argument('--model', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'], help='Размер модели YOLOv8')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности (0.0-1.0)')
    parser.add_argument('--no-masks', action='store_true', help='Отключить использование масок')
    parser.add_argument('--no-text', action='store_true', help='Отключить распознавание текста')
    parser.add_argument('--metadata', type=str, default='metadata.xml', help='Путь к файлу метаданных')
    parser.add_argument('--search', type=str, help='Поиск изображений по ключевым словам')
    parser.add_argument('--find-similar', type=str, help='Поиск похожих на указанное изображение')

    args = parser.parse_args()

    # Создаем экземпляр распознавателя
    recognizer = AdvancedRecognizer(
        model_size=args.model,
        confidence_threshold=args.conf,
        use_masks=not args.no_masks,
        ocr_enabled=not args.no_text
    )

    # Если указан поиск
    if args.search:
        results = recognizer.search_images_by_metadata(args.search, args.metadata)
        print(f"\nРезультаты поиска по запросу: '{args.search}'")
        print(f"Найдено изображений: {len(results)}")

        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['image_name']} (релевантность: {result['relevance']})")
            print(f"   Объекты: {', '.join(result['metadata']['objects'])}")
            print(f"   Текст: {result['metadata']['text']}")
            print(f"   Причины совпадения: {', '.join(result['match_reasons'])}")

    # Если указан поиск похожих изображений
    elif args.find_similar:
        results = recognizer.find_similar_images(args.find_similar, args.metadata)
        print(f"\nПохожие изображения для: '{args.find_similar}'")
        print(f"Найдено изображений: {len(results)}")

        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['image_name']} (релевантность: {result['relevance']})")
            print(f"   Объекты: {', '.join(result['metadata']['objects'])}")
            print(f"   Текст: {result['metadata']['text']}")
            print(f"   Причины совпадения: {', '.join(result['match_reasons'])}")

    # Если указана пакетная обработка
    elif args.batch:
        if not os.path.isdir(args.image):
            print(f"ОШИБКА: {args.image} не является директорией")
            sys.exit(1)

        print(f"Начало пакетной обработки изображений в: {args.image}")
        results = recognizer.batch_process(args.image, args.output, metadata_file=args.metadata)
        print(f"Обработка завершена, обработано {len(results)} изображений")

    # Обработка одного изображения
    else:
        if not os.path.isfile(args.image):
            print(f"ОШИБКА: Файл {args.image} не найден")
            sys.exit(1)

        print(f"Обработка изображения: {args.image}")
        results = recognizer.recognize(args.image)

        if 'error' in results:
            print(f"Ошибка при распознавании: {results['error']}")
            sys.exit(1)

        print(f"\nОбнаружено объектов: {len(results['objects'])}")
        for i, obj in enumerate(results['objects']):
            print(f"  {i+1}. {obj['class']} (уверенность: {obj['confidence']:.2f})")

        if not args.no_text and results.get('all_text'):
            print(f"\nРаспознанный текст: {results['all_text']}")

        # Визуализация и сохранение результатов
        output_path = args.output or f"recognized_{os.path.basename(args.image)}"
        vis_path = recognizer.visualize_results(args.image, results, output_path)

        if vis_path:
            print(f"\nРезультат сохранен в: {vis_path}")

        # Сохранение метаданных
        metadata = recognizer.generate_metadata(args.image, results)
        if metadata:
            if os.path.exists(args.metadata):
                recognizer.merge_metadata(args.metadata, [metadata])
            else:
                recognizer.save_metadata_to_xml([metadata], args.metadata)
            print(f"Метаданные сохранены в: {args.metadata}")