from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from openai import OpenAI

app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
    name="beepboop"
)

# Listen for messages that mention the bot
# @app.event("app_mention")
# def handle_mention(event, say):
#     user_message = event["text"]
#     print(user_message)
#     chat_message = user_message.split(" ")[:-1]
#     chat_message = " ".join(chat_message)
#     say(chat_message)


@app.event("app_mention")
def handle_mention(event, say):
    user_message = event["text"]
    chat_message = " ".join(user_message.split(" "))
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a coding assistant that talks like a pirate.",
        input="Please provide a short answer to the following question: " + chat_message,
    )

    
    bot_reply = response.output_text
    say(bot_reply)

def main():
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))


    handler.start()

if __name__ == "__main__":    
    # app.start(port=int(os.getenv("PORT", 3000)))
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    main()