from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import io
import base64
from PIL import Image
import tempfile
from full_detector import load_models, detect_and_read

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Глобальные переменные для моделей
yolo_model = None
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
        
        # Получаем параметр уверенности из запроса (если есть)
        conf = float(request.form.get('conf', 0.4))
        
        # Обрабатываем изображение
        print(f"Обработка изображения: {filename}, уверенность: {conf}")
        finded_images, image_with_boxes, final_text = detect_and_read(
            yolo_model, processor, trocr_model,
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
            print("⚠️ Внимание: слова найдены, но текст не распознан")
            final_text = "Текст не распознан. Попробуйте изменить параметр уверенности модели."
        elif not final_text.strip() and len(finded_images) == 0:
            print("⚠️ Внимание: слова не найдены на изображении")
            final_text = "Слова не найдены на изображении. Попробуйте уменьшить параметр уверенности модели."
        
        # Конвертируем изображение с рамками в base64
        image_base64 = image_to_base64(image_with_boxes)
        
        print(f"Результат: найдено слов - {len(finded_images)}, длина текста - {len(final_text)}")
        
        return jsonify({
            'success': True,
            'text': final_text,
            'image': image_base64,
            'words_count': len(finded_images)
        })
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
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
    yolo_model_path = 'weights/best.pt'
    if not os.path.exists(yolo_model_path):
        print("=" * 60)
        print("⚠️  ВНИМАНИЕ: Модель YOLO не найдена!")
        print("=" * 60)
        print(f"Ожидаемый путь: {os.path.abspath(yolo_model_path)}")
        print("\nМодель YOLO (best.pt) НЕ устанавливается через requirements.txt.")
        print("Вам нужно:")
        print("  1. Получить обученную модель YOLO (файл best.pt)")
        print("  2. Поместить её в папку: weights/best.pt")
        print("\nПапка 'weights' уже создана.")
        print("=" * 60)
        exit(1)
    
    # Загружаем модели только один раз (не при перезагрузке в debug режиме)
    if yolo_model is None:
        print("Загрузка моделей...")
        try:
            yolo_model, processor, trocr_model = load_models()
            print("✅ Модели загружены успешно!")
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            print("\nВозможные причины:")
            print("  - Проблема с интернет-соединением (TrOCR загружается с HuggingFace)")
            print("  - Модель YOLO не найдена: weights/best.pt")
            print("  - Проблемы с доступом к HuggingFace")
            print("\nПопробуйте:")
            print("  1. Проверить интернет-соединение")
            print("  2. Убедиться, что модель YOLO находится в weights/best.pt")
            print("  3. Запустить снова (модель TrOCR может быть уже в кэше)")
            exit(1)
    
    print("Запуск веб-сервера...")
    print("Откройте в браузере: http://localhost:5000")
    # use_reloader=False предотвращает перезагрузку моделей при изменениях кода
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)