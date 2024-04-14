# FROM python:3.10.9
FROM jupyter/datascience-notebook
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN git clone https://github.com/alper-d/3d_reconstruction_data.git
EXPOSE 8888
CMD jupyter notebook demo.ipynb —-ip=0.0.0.0 —-port=8888 —-allow-root —-no-browser 