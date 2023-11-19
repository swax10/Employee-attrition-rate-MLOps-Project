from setuptools import setup

setup(
    name='mlflow-tracking-server',
    version='0.1.0',
    packages=['mlflow_tracking_server'],
    python_requires='>=3.6',
    url='https://github.com/mlflow/mlflow.git@master#subdirectory=tracking-server',
)
