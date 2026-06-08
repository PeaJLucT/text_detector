from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import io
import base64
from PIL import Image
import tempfile
from text_detector import load_models, detect_and_read

TROCR_MODEL_PATH = "./text_recognition_model/model"

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Глобальные переменные для моделей
yolo_model = None
yolo_model_2 = None
processor = None
trocr_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Конвертирует PIL Image в base64 строку"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
    
    temp_path = None
    try:
        # Сохраняем временный файл
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        with Image.open(temp_path) as img:
            img_width, img_height = img.size
        # Получаем параметр уверенности из запроса (если есть)
        conf = float(request.form.get('conf', 0.5))
        
        # Обрабатываем изображение
        print(f"Обработка изображения: {filename}, уверенность: {conf}")
        finded_images, image_with_boxes, final_text, detected_data = detect_and_read(
            yolo_model, yolo_model_2, processor, trocr_model,
            temp_path,
            conf=conf,
            output_folder=''
        )
        
        # Удаляем временный файл
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        if image_with_boxes is None:
            return jsonify({'error': 'Ошибка при обработке изображения'}), 500
        
        # Если текст пустой, но слова найдены - возможно проблема с распознаванием
        if not final_text.strip() and len(finded_images) > 0:
            final_text = "Текст не распознан. Попробуйте изменить параметр уверенности."
        elif not final_text.strip() and len(finded_images) == 0:
            final_text = "Слова не найдены на изображении."
        
        # Конвертируем изображение с рамками в base64
        image_base64 = image_to_base64(image_with_boxes)
        
        # Разбиваем финальный текст на массив строк/слов для отображения в правой панели React
        lines_list = []
        for item in detected_data:
            raw_box = item["box"] 
            
            # Нормализуем в диапазон 0-1000
            xmin = int((raw_box[0] / img_width) * 1000)
            ymin = int((raw_box[1] / img_height) * 1000)
            xmax = int((raw_box[2] / img_width) * 1000)
            ymax = int((raw_box[3] / img_height) * 1000)
            
            lines_list.append({
                "text": item["text"],
                "boundingBox": [ymin, xmin, ymax, xmax] 
            })


        print(f"Результат: найдено слов - {len(finded_images)}, строк для вывода - {len(lines_list)}")
        
        return jsonify({
            'success': True,
            'text': final_text,
            'image': image_base64,
            'words_count': len(finded_images),
            'lines': lines_list
        })
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_text():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'Текст пуст'}), 400
        
        # Создаем временный файл с текстом
        temp_file = io.BytesIO()
        temp_file.write(text.encode('utf-8'))
        temp_file.seek(0)
        
        return send_file(
            temp_file,
            mimetype='text/plain; charset=utf-8',
            as_attachment=True,
            download_name='recognized_text.txt'
        )
    except Exception as e:
        return jsonify({'error': f'Ошибка при создании файла: {str(e)}'}), 500

if __name__ == '__main__':
    # Проверка наличия модели YOLO
    yolo_model_path = './segmentation best weight/best_1.pt'
    yolo_model_path_2 = './segmentation best weight/best_4.pt'
    if not os.path.exists(yolo_model_path) or not os.path.exists(yolo_model_path_2):
        print("=" * 60)
        print("⚠️  ВНИМАНИЕ: Модель YOLO не найдена!")
        print("=" * 60)
        print(f"Ожидаемый путь: {os.path.abspath(yolo_model_path)}")
        print("\nМодель YOLO (best.pt) НЕ устанавливается через requirements.txt.")
        print("Вам нужно:")
        print("  1. Получить обученную модель YOLO (файл best.pt)")
        print("  2. Поместить её в папку: segmentation best weight/best.pt")
        print("\nПапка 'segmentation best weight' уже создана.")
        print("=" * 60)
        exit(1)

    if not os.path.isdir(TROCR_MODEL_PATH) or not os.path.exists(
        os.path.join(TROCR_MODEL_PATH, "config.json")
    ):
        print("=" * 60)
        print("⚠️  ВНИМАНИЕ: Модель распознавания слов не найдена!")
        print("=" * 60)
        print(f"Ожидаемый путь: {os.path.abspath(TROCR_MODEL_PATH)}")
        print("Нужны файлы: config.json, model.safetensors, tokenizer.json")
        print("=" * 60)
        exit(1)
    
    # Загружаем модели только один раз (не при перезагрузке в debug режиме)
    if yolo_model is None or yolo_model_2 is None:
        print("Загрузка моделей...")
        try:
            yolo_model, yolo_model_2, processor, trocr_model = load_models(
                trocr_model_path=TROCR_MODEL_PATH,
            )
            print("✅ Модели загружены успешно!")
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            print("\nВозможные причины:")
            print("  - Не найдена папка text_recognition_model/model")
            print("  - Отсутствуют config.json или model.safetensors")
            print("  - Модель YOLO не найдена в segmentation best weight/")
            exit(1)
    
    print("Запуск веб-сервера...")
    print("Откройте в браузере: http://localhost:5000")
    # use_reloader=False предотвращает перезагрузку моделей при изменениях кода
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)