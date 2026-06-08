@echo off
chcp 65001 >nul
echo =========================================
echo Запуск интерфейса и нейросети
echo =========================================

:: Виртуальное окружение не хранится в репозитории — создаём локально при первом запуске
if not exist "venv" (
    echo [1/5] Виртуальное окружение не найдено. Создаём venv...
    python -m venv venv
    if errorlevel 1 (
        echo Ошибка: Python не установлен или не добавлен в PATH!
        pause
        exit /b
    )
) else (
    echo [1/5] Виртуальное окружение найдено.
)

echo [2/5] Активируем окружение и проверяем библиотеки...
call venv\Scripts\activate

pip show torch >nul 2>&1
if errorlevel 1 (
    echo Установка PyTorch с поддержкой CUDA...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
) else (
    echo PyTorch уже установлен, пропускаем.
)

pip show flask >nul 2>&1
if errorlevel 1 (
    echo Установка остальных зависимостей...
    pip install ultralytics numpy opencv-python transformers pillow flask flask-cors
) else (
    echo Остальные зависимости уже установлены.
)

:: Проверка модели TrOCR (не входит в репозиторий)
if not exist "text_recognition_model\model\config.json" (
    echo.
    echo =========================================
    echo ВНИМАНИЕ: модель TrOCR не найдена!
    echo =========================================
    echo Скачайте архив с Google Drive и распакуйте в:
    echo   text_recognition_model\model\
    echo Подробности — в README.md, раздел «Модели».
    echo =========================================
    pause
    exit /b
)

:: ЗАПУСК НЕЙРОСЕТИ
echo [3/5] Запускаем сервер нейросети...
start "Server" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && python app_v2.py"

:: ЗАПУСК ФРОНТЕНДА
echo [4/5] Запускаем веб-сайт...
start "Web" cmd /c "cd /d %~dp0dist && python -m http.server 3000"

:: ОЖИДАНИЕ И ОТКРЫТИЕ БРАУЗЕРА
echo [5/5] Открываем браузер...
timeout /t 8 /nobreak >nul
start http://localhost:3000

echo.
echo Готово! Если браузер не открылся, перейдите по адресу http://localhost:3000
pause
