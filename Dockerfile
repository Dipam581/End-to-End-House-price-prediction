FROM python:latest

LABEL Maintainer="Dipam.Ghosh"

WORKDIR /usr/app

COPY . /usr/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV FLASK_APP=app

CMD ["flask", "--app", "app", "run", "--host=0.0.0.0", "--port=8501"]