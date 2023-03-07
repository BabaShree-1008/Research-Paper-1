# Research-Paper-1

## Project Setup

### Environment

ensure you have conda installed on your pc. create a virtual env for the project by running the following command

```
conda create --name research-1 python=3.10
```

### Dependencies

This project uses `poetry` for dependency management. you can install `poetry` in the env we created in the previous step by running 
```
pip install --upgrade poetry
```

The project dependencies are specified in the `poetry.lock` file. after the installation of poetry, install the project dependencies with the command
```
poetry install
```

## Monitoring

We use `mlflow` to monitor and track all our experiment `runs`. you can start up an mlfow `server` on a new terminal by running the command 
```
mlflow server
```
