import os
import logging
import requests
import zipfile
import pandas as pd
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from pyspark.ml.recommendation import ALS
from minio import Minio
import tarfile

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
S3_BUCKET = "movielens-bucket"

LOCAL_DATA_DIR = "/shared_data"
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"
MODEL_DIR = "als_model"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
EXTRACT_DIR = "/shared_data/data"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

dag = DAG(
    'movielens_processing_pipeline',
    default_args=default_args,
    schedule_interval=None,
    description="Pipeline: Download, split, train model and store in Minio",
    catchup=False,
)

def download_dataset():
    logger.info("Downloading MovieLens dataset...")
    response = requests.get(MOVIELENS_URL)
    if response.status_code == 200:
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        local_zip_path = os.path.join(EXTRACT_DIR, 'ml-latest-small.zip')
        with open(local_zip_path, "wb") as f:
            f.write(response.content)
        logger.info("Dataset downloaded.")
    else:
        logger.error("Failed to download dataset.")
        raise Exception("Failed to download dataset.")

def extract_dataset():
    local_zip_path = os.path.join(EXTRACT_DIR, 'ml-latest-small.zip')
    if not os.path.exists(local_zip_path):
        logger.error("Dataset not found.")
        raise Exception("Dataset not found.")
    with zipfile.ZipFile(local_zip_path, 'r') as z:
        z.extractall(EXTRACT_DIR)
    logger.info("Dataset extracted.")

def split_data():
    ratings_path = os.path.join(EXTRACT_DIR, "ml-latest-small", "ratings.csv")
    if not os.path.exists(ratings_path):
        logger.error("Ratings file not found.")
        raise Exception("Ratings file not found.")
    df = pd.read_csv(ratings_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    train = df[df['timestamp'] < '2015-01-01']
    test = df[df['timestamp'] >= '2015-01-01']
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    train.to_csv(os.path.join(LOCAL_DATA_DIR, TRAIN_FILE), index=False)
    test.to_csv(os.path.join(LOCAL_DATA_DIR, TEST_FILE), index=False)
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")

def upload_to_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    found = client.bucket_exists(S3_BUCKET)
    if not found:
        client.make_bucket(S3_BUCKET)
        logger.info(f"Bucket {S3_BUCKET} created.")
    else:
        logger.info("Bucket already exists.")

    # Загрузка файлов
    for file_name in [TRAIN_FILE, TEST_FILE]:
        file_path = os.path.join(LOCAL_DATA_DIR, file_name)
        if not os.path.exists(file_path):
            logger.error(f"File {file_name} not found.")
            raise Exception(f"File {file_name} not found.")
        client.fput_object(S3_BUCKET, file_name, file_path)
        logger.info(f"Uploaded {file_name} to Minio.")

def train_model():
    # Создание сессии Spark
    spark = (SparkSession.builder
             .appName("MovieLensALS")
             .master("spark://spark-master:7077")
             .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Конфигурация Hadoop для работы с Minio
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3a.endpoint", MINIO_ENDPOINT)
    hadoop_conf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
    hadoop_conf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Схема данных
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", IntegerType(), True)
    ])

    # Чтение тренировочного набора данных из Minio
    train_data_path = f"s3a://{S3_BUCKET}/{TRAIN_FILE}"
    train_df = spark.read.csv(train_data_path, header=True, schema=schema)

    # Обучение модели ALS
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        maxIter=5,
        rank=10
    )
    model = als.fit(train_df)
    logger.info("Model trained.")

    # Сохранение модели локально
    local_model_path = os.path.join(LOCAL_DATA_DIR, MODEL_DIR)
    model.write().overwrite().save(local_model_path)
    logger.info("Model saved locally.")

    # Архивация модели
    tar_path = os.path.join(LOCAL_DATA_DIR, "als_model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_model_path, arcname=os.path.basename(local_model_path))
    logger.info("Model archived.")

    # Загрузка модели в Minio
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    client.fput_object(S3_BUCKET, "als_model.tar.gz", tar_path)
    logger.info("Model uploaded to Minio.")

    spark.stop()

# Определение задач
download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_dataset,
    dag=dag
)

extract_task = PythonOperator(
    task_id='extract_dataset',
    python_callable=extract_dataset,
    dag=dag
)

split_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag
)

upload_task = PythonOperator(
    task_id='upload_to_minio',
    python_callable=upload_to_minio,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Последовательность задач
download_task >> extract_task >> split_task >> upload_task >> train_task