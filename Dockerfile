# FROM python:3.10.9
FROM jupyter/datascience-notebook

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
#ENV NAME World

# Run Jupyter Notebook when the container launches
CMD jupyter notebook demo.ipynb —-ip=0.0.0.0 —-port=8888 —-allow-root —-no-browser 