FROM cytomineuliege/software-python3-base:v2.8.2-py3.7.6

RUN pip install numpy torchvision joblib sldc

RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD utils.py /app/utils.py
ADD unet.py /app/unet.py
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]