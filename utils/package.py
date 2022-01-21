class Package:
    def __init__(self, sentTo: str, sentFrom: str, consumed: bool, content: any):
        self.sentTo = sentTo
        self.sentFrom = sentFrom
        self.consumed = consumed
        self.content = content

    def json(self):
        return {
            "sentTo": self.sentTo,
            "sentFrom": self.sentFrom,
            "consumed": self.consumed,
            "content": self.content,
        }