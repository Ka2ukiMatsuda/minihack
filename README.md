# minihack

## Setup

### create virtual environment

```bash
pyenv install 3.11.11
pyenv virtualenv 3.11.11 minihack
pyenv local minihack
```

### install dependencies

```bash
pip install fastapi[standard]
```

```bash
pip install -r requirements.txt
```

## Run development server
```bash
fastapi dev main.py
```

## Info

- sentence-transformers
    - [models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

## AWS

```bash
export AWS_PROFILE=<profile>
```

```bash
aws sso login --sso-session <session>
```

```bash
aws configure list
```



```
pip install slack_bolt

python beepboop.py
```
