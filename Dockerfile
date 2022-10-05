FROM continuumio/miniconda3

RUN pip install mlflow>=1.29.0 \
	&& pip install numpy \
	&& pip install scipy \
	&& pip install pandas \
	&& pip install scikit-learn \
	&& pip install cloudpickle \
