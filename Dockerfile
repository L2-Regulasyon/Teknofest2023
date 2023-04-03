FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN export contver=1.03
RUN pip3 install --upgrade \
scikit-learn \
ipywidgets \
polars \
catboost \
lightgbm \
transformers \
datasets \
sentence-transformers \
zemberek-python \
gradio \
seaborn \
fasttext
