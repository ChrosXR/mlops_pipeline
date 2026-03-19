# Imagen base ligera
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puerto de la API
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]