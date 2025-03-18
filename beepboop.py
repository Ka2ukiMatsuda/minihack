from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os


'''
export SLACK_SIGNING_SECRET=4b44c34b67ff0a303bbce5911fe45e80
export SLACK_APP_TOKEN=xapp-1-A08J7S1KRS7-8621297727907-4c05144413831d069febcd918faa3401761124b4859cfe0a33f278cc192dd7eb
export SLACK_BOT_TOKEN=xoxb-8623675139697-8629396339345-vuvKPPKxvQvNs4rHF6fAk6oJ
'''

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