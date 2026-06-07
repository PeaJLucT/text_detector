@echo off
chcp 65001 >nul
echo =========================================
echo Запуск интерфейса и нейросети
echo =========================================

:: 1. Проверяем, создано ли виртуальное окружение
if not exist "venv" (
    echo [1/4] Виртуальное окружение не найдено. Создаем venv...
    python -m venv venv
    if errorlevel 1 (
        echo Ошибка: Python не установлен или не добавлен в PATH!
        pause
        exit /b
    )
    
    echo [2/4] Активируем окружение и устанавливаем библиотеки...
    call venv\Scripts\activate
    
    echo Установка PyTorch с поддержкой CUDA...
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    
    echo Установка остальных зависимостей...
    pip install ultralytics, numpy, cv2, transformers, pillow, flask, flask_cors
) else (
    echo [1/2] Окружение venv уже настроено.
    call venv\Scripts\activate
)

:: ЗАПУСК НЕЙРОСЕТИ
:: Замените '' на ту команду, которой реально запускается ваша модель
echo [3/4] Запускаем сервер нейросети
start "Server" cmd /k "python app_v2.py"

:: ЗАПУСК ФРОНТЕНДА
echo [4/4] Запускаем веб-сайт...
start "Web" cmd /c "cd dist && python -m http.server 3000"

:: ОЖИДАНИЕ И ОТКРЫТИЕ БРАУЗЕРА
:: Ждем 3 секунды, чтобы оба сервера успели включиться
echo Открываем браузер...
timeout /t 8 /nobreak >nul
start http://localhost:3000

echo Готово! Если браузер не открылся, перейдите по адресу http://localhost:3000
pause