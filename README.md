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
pip install -r requirements.txt
```

## Run Bot Locally

```bash
VISION_ENDPOINT=<azure-computor-vision-endpoint>
VISION_KEY=<azure-computor-vision-key>
SLACK_BOT_TOKEN=<slack-bot-token>
SLACK_SIGNING_SECRET=<slack-signing-secret>
SLACK_APP_TOKEN=<slack-app-token>
```

```bash
# you should authenticate with aws sso before running the bot
python bot.py
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



