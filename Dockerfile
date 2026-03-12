FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "examples/streamlit_batch_predictor.py", "--server.port=7860", "--server.address=0.0.0.0"]