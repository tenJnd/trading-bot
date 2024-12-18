# build stage
FROM python:3.10.0 as builder

#ARG PYPI_URL
#ENV PYPI_URL=${PYPI_URL}

WORKDIR /app

RUN python -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/tenJnd/notifier.git@main
RUN pip install git+https://github.com/tenJnd/utils.git@main
RUN pip install git+https://github.com/tenJnd/database-tools.git@main
RUN pip install --upgrade git+https://github.com/tenJnd/llm-adapters.git@main


# app stage
FROM python:3.10.0

RUN groupadd -g 999 python && useradd -r -u 999 -g python python

RUN mkdir /app && chown python:python /app

WORKDIR /app

COPY --chown=python:python --from=builder /app/venv ./venv
COPY --chown=python:python src /app/src
RUN mkdir -p /app/trading_data && chown python:python /app/trading_data

USER 999

ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/app"

#CMD ["python", "src/main.py"]
