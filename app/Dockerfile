FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения (исправлено)
COPY . /app/

# Создание директорий для загрузок и результатов
RUN mkdir -p uploads results metadata models

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]