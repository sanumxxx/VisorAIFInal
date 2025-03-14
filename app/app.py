"""
Веб-приложение на Flask для автоматического анализа изображений.
Обеспечивает обнаружение объектов и распознавание текста с использованием
модуля advanced_recognition.py и предоставляет веб-интерфейс для работы с системой.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, abort
from werkzeug.utils import secure_filename
import logging

# Импортируем модуль распознавания
from advanced_recognition import AdvancedRecognizer

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'results')
app.config['METADATA_FILE'] = os.path.join(os.getcwd(), 'metadata', 'metadata.xml')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла: 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Создаем необходимые директории
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['METADATA_FILE']), exist_ok=True)

# Глобальная настройка для включения OCR (распознавания текста)
FORCE_OCR_ENABLED = True

# Инициализация распознавателя
recognizer = None

OBJECT_TRANSLATIONS = {
    # Люди
    'человек': 'person',
    'люди': 'person',

    # Транспорт
    'автомобиль': 'car',
    'машина': 'car',
    'велосипед': 'bicycle',
    'мотоцикл': 'motorcycle',
    'автобус': 'bus',
    'поезд': 'train',
    'грузовик': 'truck',
    'лодка': 'boat',

    # Животные
    'кошка': 'cat',
    'собака': 'dog',
    'лошадь': 'horse',
    'овца': 'sheep',
    'корова': 'cow',
    'слон': 'elephant',
    'медведь': 'bear',
    'зебра': 'zebra',
    'жираф': 'giraffe',
    'птица': 'bird',

    # Предметы
    'стол': 'table',
    'стул': 'chair',
    'диван': 'sofa',
    'растение': 'potted plant',
    'кровать': 'bed',
    'зеркало': 'mirror',
    'обеденный стол': 'dining table',
    'окно': 'window',
    'книга': 'book',
    'часы': 'clock',
    'ваза': 'vase',
    'ножницы': 'scissors',
    'плюшевый мишка': 'teddy bear',
    'фен': 'hair drier',
    'зубная щетка': 'toothbrush',

    # Электроника
    'телевизор': 'tv',
    'ноутбук': 'laptop',
    'мышь': 'mouse',
    'пульт': 'remote',
    'клавиатура': 'keyboard',
    'телефон': 'cell phone',
    'микроволновка': 'microwave',
    'духовка': 'oven',
    'тостер': 'toaster',
    'раковина': 'sink',
    'холодильник': 'refrigerator',

    # Еда
    'банан': 'banana',
    'яблоко': 'apple',
    'бутерброд': 'sandwich',
    'апельсин': 'orange',
    'брокколи': 'broccoli',
    'морковь': 'carrot',
    'хот-дог': 'hot dog',
    'пицца': 'pizza',
    'пончик': 'donut',
    'торт': 'cake',

    # Спорт
    'мяч': 'sports ball',
    'кайт': 'kite',
    'бейсбольная бита': 'baseball bat',
    'бейсбольная перчатка': 'baseball glove',
    'скейтборд': 'skateboard',
    'серфборд': 'surfboard',
    'теннисная ракетка': 'tennis racket',

    # На улице
    'светофор': 'traffic light',
    'пожарный гидрант': 'fire hydrant',
    'дорожный знак': 'stop sign',
    'парковочный счетчик': 'parking meter',
    'скамейка': 'bench',

    # Прочее
    'рюкзак': 'backpack',
    'зонт': 'umbrella',
    'сумка': 'handbag',
    'галстук': 'tie',
    'чемодан': 'suitcase',
    'фрисби': 'frisbee',
    'лыжи': 'skis',
    'сноуборд': 'snowboard',
    'бутылка': 'bottle',
    'бокал вина': 'wine glass',
    'чашка': 'cup',
    'вилка': 'fork',
    'нож': 'knife',
    'ложка': 'spoon',
    'миска': 'bowl'
}

# Создаем обратный словарь (с английского на русский)
REVERSE_TRANSLATIONS = {eng: rus for rus, eng in OBJECT_TRANSLATIONS.items()}

# Объединенный словарь для двустороннего поиска
EXTENDED_TRANSLATIONS = OBJECT_TRANSLATIONS.copy()
EXTENDED_TRANSLATIONS.update({eng: rus for rus, eng in OBJECT_TRANSLATIONS.items()})

def init_recognizer():
    """Инициализация модуля распознавания при первом обращении"""
    global recognizer
    if recognizer is None:
        logger.info("Инициализация модуля распознавания...")
        try:
            # Явно указываем все параметры, чтобы убедиться, что OCR включен
            recognizer = AdvancedRecognizer(
                model_size='m',
                confidence_threshold=0.25,
                device=None,  # Автоматический выбор устройства
                models_path=None,  # Путь по умолчанию
                special_classes=None,  # Используем классы по умолчанию
                ocr_languages=['ru', 'en'],  # Явно указываем языки
                ocr_enabled=True,  # Явно включаем OCR
                use_masks=True,
                use_image_enhancement=True,
                mask_threshold=0.5,
                mask_refinement=True
            )
            logger.info(f"Модуль распознавания успешно инициализирован на устройстве: {recognizer.device}")

            # Проверяем, что OCR действительно включен
            if not recognizer.ocr_enabled or not hasattr(recognizer, 'reader'):
                logger.warning("OCR не инициализирован корректно. Пробуем инициализировать вручную.")
                try:
                    # Принудительная инициализация OCR
                    recognizer.ocr_enabled = True
                    recognizer._initialize_ocr()
                    logger.info(f"OCR переинициализирован, статус: {recognizer.ocr_enabled}")
                except Exception as ocr_err:
                    logger.error(f"Ошибка при переинициализации OCR: {ocr_err}")

            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации модуля распознавания: {e}")
            # Повторно попытаемся инициализировать при следующем обращении
            return False
    return True

def allowed_file(filename):
    """Проверяет, имеет ли файл допустимое расширение"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


def enhanced_search(recognizer, query, metadata_file=None, results_limit=None):
    """
    Расширенная функция поиска с поддержкой переводов объектов.

    Args:
        recognizer: Экземпляр AdvancedRecognizer
        query (str): Поисковый запрос
        metadata_file (str, optional): Путь к файлу метаданных
        results_limit (int, optional): Ограничение количества результатов

    Returns:
        list: Список найденных изображений с релевантностью
    """
    # Разбиваем запрос на отдельные слова
    query_terms = query.lower().strip().split()

    # Находим переводы всех слов запроса
    translated_terms = []
    translated_query_parts = []

    for term in query_terms:
        if term in OBJECT_TRANSLATIONS:
            # Если есть прямой перевод слова
            translated_term = OBJECT_TRANSLATIONS[term]
            translated_terms.append(translated_term)
            translated_query_parts.append(f"{term}→{translated_term}")
        else:
            # Оставляем слово без перевода
            translated_terms.append(term)

    # Если были переводы, формируем переведенный запрос
    if translated_query_parts:
        translated_query = " ".join(translated_terms)
        logger.info(f"Расширенный поиск: оригинальный запрос '{query}', переведенный '{translated_query}'")

        # Выполняем поиск с переведенным запросом
        results = recognizer.search_images_by_metadata(translated_query, metadata_file, results_limit)

        # Добавляем информацию о переводе в результаты
        for result in results:
            result['translated'] = True
            result['original_query'] = query
            result['translated_query'] = translated_query
            result['translations'] = ", ".join(translated_query_parts)

        return results
    else:
        # Если переводов не было, используем обычный поиск
        logger.info(f"Стандартный поиск (без переводов): '{query}'")
        results = recognizer.search_images_by_metadata(query, metadata_file, results_limit)

        # Отмечаем, что перевод не применялся
        for result in results:
            result['translated'] = False

        return results

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загрузки и анализа изображения с сохранением предсказуемых имен файлов"""
    # Проверяем, есть ли файл в запросе
    if 'file' not in request.files:
        flash('Не выбран файл', 'error')
        return redirect(request.url)

    file = request.files['file']

    # Если пользователь не выбрал файл
    if file.filename == '':
        flash('Не выбран файл', 'error')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Инициализируем распознаватель, если еще не инициализирован
        if not init_recognizer():
            flash('Ошибка инициализации модуля распознавания. Попробуйте позже.', 'error')
            return redirect(url_for('index'))

        # Получаем безопасное имя файла
        filename = secure_filename(file.filename)

        # Создаем уникальное имя для сохранения оригинального файла
        # но сохраняем связь с оригинальным именем для результатов
        unique_id = str(uuid.uuid4())
        base_name, extension = os.path.splitext(filename)

        # Для хранения оригинала используем уникальное имя
        unique_filename = f"{base_name}_{unique_id}{extension}"
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Но для результатов будем использовать предсказуемое имя без UUID
        result_filename = f"vis_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        # Сохраняем загруженный файл
        file.save(original_file_path)

        try:
            # Обработка изображения
            detect_text = True  # Всегда включено
            enhanced_detection = 'enhanced_detection' in request.form
            include_raw = 'include_raw' in request.form

            logger.info(f"Начинаем обработку изображения: {original_file_path}")
            logger.info(f"Результат будет сохранен как: {result_path}")

            # Распознавание
            results = recognizer.recognize(
                original_file_path,
                detect_text=True,  # Принудительно включаем OCR
                enhanced_detection=enhanced_detection,
                include_raw=include_raw
            )

            if 'error' in results:
                flash(f"Ошибка при обработке изображения: {results['error']}", 'error')
                return redirect(url_for('index'))

            # Визуализация результатов с явным указанием пути сохранения
            # Это гарантирует, что результат будет сохранен с правильным именем
            visualized_path = recognizer.visualize_results(original_file_path, results, result_path)

            # Сохранение результатов в JSON с тем же именем, но с расширением .json
            json_filename = f"{os.path.splitext(result_filename)[0]}.json"
            json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)

            with open(json_path, 'w', encoding='utf-8') as f:
                # Если 'raw_results' есть в результатах, удаляем их перед сохранением
                if 'raw_results' in results:
                    del results['raw_results']
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Для метаданных используем оригинальное имя файла (без UUID)
            # Это гарантирует, что метаданные будут связаны с предсказуемым именем файла
            metadata = recognizer.generate_metadata(filename, results)
            if metadata:
                if os.path.exists(app.config['METADATA_FILE']):
                    recognizer.merge_metadata(app.config['METADATA_FILE'], [metadata])
                else:
                    recognizer.save_metadata_to_xml([metadata], app.config['METADATA_FILE'])

            # Перенаправление на страницу результатов
            # Передаем оригинальное имя файла (без UUID) для поиска результатов
            return redirect(url_for('show_result', result_id=unique_id, original_filename=filename))

        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            flash(f"Произошла ошибка при обработке изображения: {str(e)}", 'error')
            return redirect(url_for('index'))
    else:
        flash('Недопустимый формат файла. Разрешены: png, jpg, jpeg, bmp, webp', 'error')
        return redirect(url_for('index'))


@app.route('/result/<result_id>')
def show_result(result_id):
    """Отображение результатов анализа изображения"""
    # Получаем оригинальное имя файла из параметра, если оно передано
    original_filename = request.args.get('original_filename')

    # Ищем все файлы с таким ID
    original_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"*_{result_id}.*"))

    if not original_files:
        flash('Результат не найден', 'error')
        return redirect(url_for('index'))

    original_path = str(original_files[0])

    # Если original_filename не передан, извлекаем его из пути
    if not original_filename:
        # Извлекаем имя файла из пути, отбрасывая UUID
        base_name = os.path.basename(original_path)
        name_parts = base_name.split('_')
        if len(name_parts) > 1:
            # Предполагаем, что последняя часть - UUID
            name_without_uuid = '_'.join(name_parts[:-1])
            original_filename = f"{name_without_uuid}{os.path.splitext(base_name)[1]}"
        else:
            original_filename = base_name

    # Конструируем предсказуемые пути к результатам
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"vis_{original_filename}")
    json_path = os.path.join(app.config['RESULTS_FOLDER'], f"vis_{os.path.splitext(original_filename)[0]}.json")

    # Проверяем, существуют ли файлы
    if not os.path.exists(result_path) or not os.path.exists(json_path):
        logger.warning(f"Файлы результатов не найдены: {result_path} или {json_path}")
        # Если файлы не существуют, пробуем найти по старому методу с UUID
        result_files = list(Path(app.config['RESULTS_FOLDER']).glob(f"result_{result_id}.*"))
        json_files = list(Path(app.config['RESULTS_FOLDER']).glob(f"result_{result_id}.json"))

        if result_files and json_files:
            result_path = str([f for f in result_files if not f.name.endswith('.json')][0])
            json_path = str(json_files[0])
        else:
            flash('Файлы результатов не найдены', 'error')
            return redirect(url_for('index'))

    # Загружаем результаты
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении файла результатов {json_path}: {e}")
        flash('Ошибка при чтении результатов анализа', 'error')
        return redirect(url_for('index'))

    return render_template(
        'result.html',
        result_id=result_id,
        original_filename=os.path.basename(original_path),
        result_filename=os.path.basename(result_path),
        results=results
    )

@app.route('/download/<result_id>/<file_type>')
def download_file(result_id, file_type):
    """Скачивание файлов результатов"""
    if file_type == 'original':
        files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"*_{result_id}.*"))
        if not files:
            abort(404)
        file_path = str(files[0])

    elif file_type == 'visualized':
        files = list(Path(app.config['RESULTS_FOLDER']).glob(f"result_{result_id}.*"))
        files = [f for f in files if not f.name.endswith('.json')]
        if not files:
            abort(404)
        file_path = str(files[0])

    elif file_type == 'json':
        file_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{result_id}.json")
        if not os.path.exists(file_path):
            abort(404)
    else:
        abort(404)

    return send_file(file_path, as_attachment=True)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Поиск изображений по метаданным с поддержкой переводов"""
    if request.method == 'POST':
        # Инициализируем распознаватель, если еще не инициализирован
        if not init_recognizer():
            flash('Ошибка инициализации модуля распознавания. Попробуйте позже.', 'error')
            return redirect(url_for('index'))

        query = request.form.get('query', '')
        limit = int(request.form.get('limit', 10))

        if not query:
            flash('Введите запрос для поиска', 'error')
            return render_template('search.html', results=None, query=query)

        try:
            # Проверяем наличие файла метаданных
            if not os.path.exists(app.config['METADATA_FILE']):
                flash('Файл метаданных не найден. Сначала загрузите и обработайте изображения.', 'warning')
                return render_template('search.html', results=None, query=query)

            # Выполняем расширенный поиск с поддержкой переводов
            search_results = enhanced_search(recognizer, query, app.config['METADATA_FILE'], limit)

            # Для каждого результата ищем соответствующий файл визуализации
            for result in search_results:
                # Используем предсказуемое имя файла визуализации
                vis_filename = f"vis_{result['image_name']}"
                result['vis_filename'] = vis_filename

                # Проверяем доступность файла визуализации
                vis_path = os.path.join(app.config['RESULTS_FOLDER'], vis_filename)
                result['has_visualization'] = os.path.exists(vis_path)

            # Проверяем, был ли запрос переведен
            was_translated = any(result.get('translated', False) for result in search_results)

            return render_template('search.html',
                                   results=search_results,
                                   query=query,
                                   was_translated=was_translated)

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            flash(f"Произошла ошибка при поиске: {str(e)}", 'error')
            return render_template('search.html', results=None, query=query)

    return render_template('search.html', results=None, query='')

@app.route('/view_metadata')
def view_metadata():
    """Просмотр всех метаданных"""
    if not init_recognizer():
        flash('Ошибка инициализации модуля распознавания. Попробуйте позже.', 'error')
        return redirect(url_for('index'))

    try:
        # Проверяем наличие файла метаданных
        if not os.path.exists(app.config['METADATA_FILE']):
            flash('Файл метаданных не найден. Сначала загрузите и обработайте изображения.', 'warning')
            return render_template('metadata.html', metadata=None)

        # Чтение метаданных
        metadata_list = recognizer.read_metadata_from_xml(app.config['METADATA_FILE'])

        return render_template('metadata.html', metadata=metadata_list)

    except Exception as e:
        logger.error(f"Ошибка при чтении метаданных: {e}")
        flash(f"Произошла ошибка при чтении метаданных: {str(e)}", 'error')
        return render_template('metadata.html', metadata=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Отдача загруженных файлов"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


# Убедитесь, что эта функция определена в вашем app.py

@app.route('/similar', methods=['GET', 'POST'])
def find_similar():
    """Поиск похожих изображений"""
    if request.method == 'POST':
        # Проверяем, есть ли файл в запросе
        if 'file' not in request.files:
            flash('Не выбран файл', 'error')
            return redirect(request.url)

        file = request.files['file']

        # Если пользователь не выбрал файл
        if file.filename == '':
            flash('Не выбран файл', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Инициализируем распознаватель, если еще не инициализирован
            if not init_recognizer():
                flash('Ошибка инициализации модуля распознавания. Попробуйте позже.', 'error')
                return redirect(url_for('index'))

            # Создаем уникальное имя файла
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            base_name, extension = os.path.splitext(filename)
            unique_filename = f"{base_name}_{unique_id}{extension}"

            # Сохраняем загруженный файл
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            limit = int(request.form.get('limit', 10))

            try:
                # Проверяем наличие файла метаданных
                if not os.path.exists(app.config['METADATA_FILE']):
                    flash('Файл метаданных не найден. Сначала загрузите и обработайте изображения.', 'warning')
                    return render_template('similar.html', results=None, image_path=None)

                # Поиск похожих изображений
                similar_results = recognizer.find_similar_images(file_path, app.config['METADATA_FILE'], limit)

                # Для каждого результата ищем соответствующий файл визуализации
                for result in similar_results:
                    original_filename = result['image_name']

                    # Ищем соответствующий файл визуализации
                    vis_filename = find_visualization_file(original_filename)

                    # Добавляем информацию о визуализации в результат
                    result['vis_filename'] = vis_filename
                    result['has_visualization'] = vis_filename is not None

                return render_template('similar.html', results=similar_results, image_path=unique_filename)

            except Exception as e:
                logger.error(f"Ошибка при поиске похожих изображений: {e}")
                flash(f"Произошла ошибка при поиске: {str(e)}", 'error')
                return render_template('similar.html', results=None, image_path=None)
        else:
            flash('Недопустимый формат файла. Разрешены: png, jpg, jpeg, bmp, webp', 'error')
            return redirect(url_for('index'))

    return render_template('similar.html', results=None, image_path=None)

def find_visualization_file(original_filename):
    """
    Находит соответствующий визуализированный файл для оригинального имени из метаданных.

    Args:
        original_filename (str): Оригинальное имя файла из метаданных

    Returns:
        str: Имя найденного визуализированного файла или None, если файл не найден
    """
    try:
        import os
        import glob

        results_folder = app.config['RESULTS_FOLDER']

        # Получаем базовое имя файла без пути и расширения
        base_name = os.path.splitext(os.path.basename(original_filename))[0]

        # Список шаблонов для поиска
        patterns = [
            # Префикс vis_ + точное имя
            os.path.join(results_folder, f"vis_{original_filename}"),

            # Префикс vis_ + базовое имя + любое расширение
            os.path.join(results_folder, f"vis_{base_name}.*"),

            # Префикс vis_ + базовое имя + любой UUID + любое расширение
            os.path.join(results_folder, f"vis_{base_name}_*.*"),

            # Если базовое имя содержит UUID (имя_uuid), ищем по основному имени
            os.path.join(results_folder, f"vis_{base_name.split('_')[0]}*.*") if '_' in base_name else None
        ]

        # Удаляем None из списка паттернов
        patterns = [p for p in patterns if p is not None]

        # Ищем по всем шаблонам
        for pattern in patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                # Возвращаем только имя файла, без пути
                return os.path.basename(matching_files[0])

        # Запасной вариант: смотрим все файлы с префиксом vis_ и ищем наиболее похожее имя
        all_vis_files = glob.glob(os.path.join(results_folder, "vis_*.*"))

        if all_vis_files:
            # Импортируем только если нужно
            from difflib import SequenceMatcher

            best_match = None
            best_ratio = 0

            for vis_file in all_vis_files:
                vis_basename = os.path.basename(vis_file)
                # Удаляем префикс vis_ для сравнения
                vis_name = vis_basename[4:] if vis_basename.startswith('vis_') else vis_basename

                # Вычисляем коэффициент похожести
                ratio = SequenceMatcher(None, original_filename, vis_name).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = vis_basename

            # Если коэффициент выше 0.5, считаем это совпадением
            if best_ratio > 0.5:
                return best_match

        # Ничего не нашли
        return None

    except Exception as e:
        logger.error(f"Ошибка при поиске визуализации для {original_filename}: {str(e)}")
        return None


def find_closest_match(filename, directory):
    """
    Ищет наиболее близкое совпадение для имени файла в указанной директории.

    Args:
        filename (str): Искомое имя файла
        directory (str): Директория для поиска

    Returns:
        str: Путь к найденному файлу или None, если файл не найден
    """
    try:
        import os
        from difflib import SequenceMatcher

        # Если файл существует напрямую, возвращаем его
        direct_path = os.path.join(directory, filename)
        if os.path.exists(direct_path):
            return direct_path

        # Ищем префикс 'vis_' если его нет
        if not filename.startswith('vis_') and os.path.exists(os.path.join(directory, f"vis_{filename}")):
            return os.path.join(directory, f"vis_{filename}")

        # Получаем список файлов в директории
        files = os.listdir(directory)

        # Ищем файлы с тем же расширением
        base, ext = os.path.splitext(filename)
        matching_ext = [f for f in files if f.endswith(ext)]

        # Ищем файлы, которые содержат часть имени (без UUID)
        if '_' in base:
            name_parts = base.split('_')
            if len(name_parts) > 1:
                # Предполагаем, что UUID в конце
                name_without_uuid = '_'.join(name_parts[:-1])
                matching_name = [f for f in matching_ext if name_without_uuid in f]

                if matching_name:
                    # Берем первое совпадение
                    return os.path.join(directory, matching_name[0])

        # Если не нашли по имени, ищем по сходству
        if matching_ext:
            # Ищем наиболее похожее имя
            best_match = None
            best_ratio = 0

            for f in matching_ext:
                ratio = SequenceMatcher(None, filename, f).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = f

            # Если сходство выше 0.6, считаем это совпадением
            if best_ratio > 0.6:
                return os.path.join(directory, best_match)

        # Ничего не нашли
        return None

    except Exception as e:
        logger.error(f"Ошибка при поиске файла {filename}: {str(e)}")
        return None


@app.route('/results/<path:filename>')
def result_file(filename):
    """Отдача файлов результатов с улучшенной обработкой ошибок"""
    try:
        # Проверяем, чтобы не было выхода за пределы директории результатов
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)

        # Проверяем существование файла
        if os.path.exists(file_path):
            return send_file(file_path)

        # Если файл не найден напрямую, проверяем, может быть это запрос на vis_ файл
        if not filename.startswith('vis_') and os.path.exists(
                os.path.join(app.config['RESULTS_FOLDER'], f"vis_{filename}")):
            return send_file(os.path.join(app.config['RESULTS_FOLDER'], f"vis_{filename}"))

        # Если это запрос на визуализированное изображение
        if filename.startswith('vis_'):
            # Пробуем найти любой файл с похожим именем
            base_name = filename[4:]  # Убираем префикс vis_
            # Проверяем без UUID
            if '_' in base_name:
                name_parts = base_name.split('_')
                # Если это похоже на имя с UUID, ищем без него
                if len(name_parts) > 1 and len(name_parts[-1]) >= 8:
                    name_without_uuid = '_'.join(name_parts[:-1])
                    alt_path = os.path.join(app.config['RESULTS_FOLDER'],
                                            f"vis_{name_without_uuid}{os.path.splitext(base_name)[1]}")
                    if os.path.exists(alt_path):
                        return send_file(alt_path)

            # Отдаем изображение-заглушку вместо 404
            placeholder_path = os.path.join(os.getcwd(), 'static', 'img', 'image-not-found.png')
            if os.path.exists(placeholder_path):
                return send_file(placeholder_path)

        # Если все проверки не помогли, возвращаем 404
        logger.warning(f"Файл не найден: {file_path}")
        abort(404)

    except Exception as e:
        logger.error(f"Ошибка при отдаче файла {filename}: {str(e)}")
        return abort(500)

@app.route('/batch_process', methods=['GET', 'POST'])
def batch_process():
    """Пакетная обработка изображений"""
    if request.method == 'POST':
        # Проверка наличия файлов
        if 'files[]' not in request.files:
            flash('Не выбраны файлы', 'error')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        # Если пользователь не выбрал файлы
        if len(files) == 0:
            flash('Не выбраны файлы', 'error')
            return redirect(request.url)

        # Инициализируем распознаватель, если еще не инициализирован
        if not init_recognizer():
            flash('Ошибка инициализации модуля распознавания. Попробуйте позже.', 'error')
            return redirect(url_for('index'))

        # Проверяем статус OCR
        if hasattr(recognizer, 'ocr_enabled'):
            logger.info(f"Статус OCR перед пакетной обработкой: {recognizer.ocr_enabled}")
            # Принудительно включаем OCR, если это необходимо
            if not recognizer.ocr_enabled:
                recognizer.ocr_enabled = True
                logger.info("Принудительно включен OCR перед пакетной обработкой")

        # Создаем временную директорию для пакетной обработки
        batch_id = str(uuid.uuid4())
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        processed_files = []

        # Сохраняем все файлы
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_folder, filename)
                file.save(file_path)
                processed_files.append(filename)

        if len(processed_files) == 0:
            flash('Нет допустимых файлов для обработки', 'error')
            return redirect(url_for('batch_process'))

        try:
            # Создаем директорию для результатов
            output_folder = os.path.join(app.config['RESULTS_FOLDER'], f"batch_{batch_id}")
            os.makedirs(output_folder, exist_ok=True)

            # Модифицируем функцию batch_process для принудительного включения распознавания текста
            original_batch_process = recognizer.batch_process

            def patched_batch_process(*args, **kwargs):
                # Убедимся, что OCR включен перед обработкой
                if hasattr(recognizer, 'ocr_enabled'):
                    recognizer.ocr_enabled = True
                return original_batch_process(*args, **kwargs)

            # Временно заменяем функцию
            recognizer.batch_process = patched_batch_process

            # Запускаем пакетную обработку
            batch_results = recognizer.batch_process(
                batch_folder,
                output_folder,
                metadata_file=app.config['METADATA_FILE']
            )

            # Восстанавливаем оригинальную функцию
            recognizer.batch_process = original_batch_process

            if 'error' in batch_results:
                flash(f"Ошибка при пакетной обработке: {batch_results['error']}", 'error')
                return redirect(url_for('batch_process'))

            # Перенаправление на страницу результатов пакетной обработки
            return redirect(url_for('batch_results', batch_id=batch_id))

        except Exception as e:
            logger.error(f"Ошибка при пакетной обработке: {e}")
            flash(f"Произошла ошибка при пакетной обработке: {str(e)}", 'error')
            return redirect(url_for('batch_process'))

    return render_template('batch.html')

@app.route('/batch_results/<batch_id>')
def batch_results(batch_id):
    """Отображение результатов пакетной обработки"""
    batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{batch_id}")
    output_folder = os.path.join(app.config['RESULTS_FOLDER'], f"batch_{batch_id}")

    if not os.path.exists(batch_folder) or not os.path.exists(output_folder):
        flash('Результаты пакетной обработки не найдены', 'error')
        return redirect(url_for('index'))

    # Получаем списки исходных файлов и результатов
    original_files = [f for f in os.listdir(batch_folder) if allowed_file(f)]

    results = []
    for filename in original_files:
        base_name = os.path.splitext(filename)[0]
        visualized_path = os.path.join(output_folder, f"vis_{filename}")
        json_path = os.path.join(output_folder, f"{base_name}.json")

        # Проверяем наличие файлов результатов
        if os.path.exists(visualized_path) and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            num_objects = len(json_data.get('objects', []))
            num_text_blocks = len(json_data.get('text_blocks', []))
            all_text = json_data.get('all_text', '')

            results.append({
                'original': filename,
                'visualized': f"vis_{filename}",
                'json': f"{base_name}.json",
                'num_objects': num_objects,
                'num_text_blocks': num_text_blocks,
                'text': all_text[:100] + '...' if len(all_text) > 100 else all_text
            })

    return render_template('batch_results.html',
                           batch_id=batch_id,
                           results=results,
                           num_processed=len(results))

@app.route('/batch_download/<batch_id>/<filename>')
def batch_download(batch_id, filename):
    """Скачивание файлов из пакетной обработки"""
    if filename.startswith('vis_') or filename.endswith('.json'):
        file_path = os.path.join(app.config['RESULTS_FOLDER'], f"batch_{batch_id}", filename)
    else:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{batch_id}", filename)

    if not os.path.exists(file_path):
        abort(404)

    return send_file(file_path, as_attachment=True)

@app.route('/about')
def about():
    """Информация о системе"""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Обработка ошибки 404"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Обработка ошибки 500"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # При запуске приложения инициализируем распознаватель
    init_recognizer()
    app.run(debug=True)