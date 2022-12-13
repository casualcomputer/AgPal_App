FROM python:3.9.13

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8080 

COPY agpal_plus_8d2114c78f0a.json . 
COPY app.py . 
COPY data  ./data/

CMD streamlit run --server.port 8080 --server.enableCORS false app.py