FROM python:3.9-slim-buster
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update && apt-get -yqq install gcc python3-dev nano procps
RUN pip install -r requirements.txt
COPY . /code/
CMD ["python", "/code/run.py", "serve"]