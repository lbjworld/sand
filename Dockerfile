FROM gw000/keras:2.0.6-py2-tf-cpu

RUN pip install pandas pandas-datareader scikit-learn matplotlib jupyter

CMD jupyter notebook --no-browser --ip=0.0.0.0 --port=8888 --allow-root
