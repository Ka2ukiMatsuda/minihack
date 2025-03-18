from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os


app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
    name="beepboop"
)

# Listen for messages that mention the bot
@app.event("app_mention")
def handle_mention(event, say):
    user_message = event["text"]
    print(user_message)
    say(user_message)

def main():
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()

if __name__ == "__main__":    
    # app.start(port=int(os.getenv("PORT", 3000)))
    main()