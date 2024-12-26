FROM apache/airflow:2.10.3

RUN pip install pyspark==3.5.0 \
    && pip install pandas \
    && pip install minio \
    && pip install pyarrow

WORKDIR /opt/
RUN curl https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
RUN curl https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.540/aws-java-sdk-bundle-1.12.540.jar


USER root
RUN apt-get update && apt-get install -y default-jdk && apt-get autoremove -yqq --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
