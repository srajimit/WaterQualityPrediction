FROM python:3.10-slim

WORKDIR /app

COPY Requirements.txt /app/Requirements.txt
RUN pip install --no-cache-dir -r Requirements.txt

EXPOSE 5000

CMD ["python", "flaskapp.py"]
