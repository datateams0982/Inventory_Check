FROM civisanalytics/datascience-python

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install tzdata
ENV TZ Asia/Taipei

COPY ./inv_check_daily_prediction /inv_check_daily_prediction
RUN pip install -r /inv_check_daily_prediction/requirements.txt

CMD ["python","/inv_check_daily_prediction/main.py"]