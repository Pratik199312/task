# Use Python 3.9 base image
FROM python:3.9

# Set working directory
WORKDIR /app
 
# Copy necessary files
COPY requirements.txt ./
COPY train.py ./ 
COPY serve.py ./
COPY Training_Data.csv ./
 
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000 

CMD ["python train_model.py && uvicorn serve_model:app --host 0.0.0.0 --port 8000"]
