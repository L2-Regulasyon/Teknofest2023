FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN export contver=1.0
RUN pip3 install --upgrade gradio \
seaborn \
scikit-learn \
ipywidgets \
polars \
catboost \
lightgbm \
transformers \
datasets \
sentence-transformers \
