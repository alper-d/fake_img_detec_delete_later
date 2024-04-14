# FROM python:3.10.9
FROM jupyter/datascience-notebook
WORKDIR /app
COPY . /app
RUN mkdir demo_data
RUN cp aagfhgtpmv.mp4 demo_data/aagfhgtpmv.mp4 && rm aagfhgtpmv.mp4
RUN cp aagfhgtpmv.pt demo_data/aagfhgtpmv.pt && rm aagfhgtpmv.pt
RUN cp metadata_edited.json demo_data/metadata_edited.json && rm metadata_edited.json
RUN pip install -r requirements.txt
RUN git clone https://github.com/alper-d/3d_reconstruction_data.git
EXPOSE 8888
CMD jupyter notebook demo.ipynb —-ip=0.0.0.0 —-port=8888 —-allow-root —-no-browser 