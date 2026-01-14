const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const processBtn = document.getElementById('processBtn');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const textOutput = document.getElementById('textOutput');
const downloadBtn = document.getElementById('downloadBtn');
const errorMessage = document.getElementById('errorMessage');
const loader = document.getElementById('loader');
const stats = document.getElementById('stats');

let currentText = '';

// Обработка клика по области загрузки
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Обработка выбора файла
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        showError('Пожалуйста, выберите изображение');
    }
});

// Обработка файла
function handleFile(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showError('Пожалуйста, выберите файл изображения');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
        hideError();
    };
    reader.readAsDataURL(file);
}

// Удаление изображения
removeBtn.addEventListener('click', () => {
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    hideError();
});

// Обновление значения уверенности
confSlider.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(1);
});

// Обработка распознавания
processBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        showError('Пожалуйста, выберите изображение');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', confSlider.value);
    
    // Показываем загрузку
    processBtn.disabled = true;
    loader.style.display = 'inline-block';
    processBtn.querySelector('.btn-text').textContent = 'Обработка...';
    hideError();
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Показываем результаты
            resultImage.src = data.image;
            currentText = data.text || 'Текст не распознан';
            textOutput.textContent = currentText;
            stats.textContent = `Найдено слов: ${data.words_count || 0}`;
            resultsSection.style.display = 'block';
            
            // Прокручиваем к результатам
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            showError(data.error || 'Ошибка при обработке изображения');
        }
    } catch (error) {
        showError('Ошибка соединения с сервером: ' + error.message);
    } finally {
        processBtn.disabled = false;
        loader.style.display = 'none';
        processBtn.querySelector('.btn-text').textContent = 'Распознать текст';
    }
});

// Скачивание текста
downloadBtn.addEventListener('click', async () => {
    if (!currentText) {
        showError('Нет текста для скачивания');
        return;
    }
    
    try {
        const response = await fetch('/download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: currentText })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'recognized_text.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const data = await response.json();
            showError(data.error || 'Ошибка при скачивании файла');
        }
    } catch (error) {
        showError('Ошибка при скачивании: ' + error.message);
    }
});

// Показ ошибки
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
}

// Скрытие ошибки
function hideError() {
    errorMessage.style.display = 'none';
}
