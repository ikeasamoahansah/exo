FROM ubuntu:22.04

WORKDIR /code

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# install app
COPY . .

# final configuration

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]