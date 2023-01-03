FROM python:3.9
WORKDIR /
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
CMD ["uvicorn", "main:medic", "--host", "0.0.0.0", "--port", "80"]