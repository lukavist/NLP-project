import re

PATTERNS = {
    "check_balance": [r"\b(balance|check|account|how much)\b"],
    "withdraw": [r"\b(withdraw|get cash|cash out|money)\b"],
    "end": [r"\b(end|done|bye|no thanks|exit|nothing|no|nope)\b"],
    "continue": [r"\b(continue|yes|okay|sure|more|yep)\b"],
}


class ATMChatBot:
    def __init__(self):
        self.state = "Idle"
        self.balance = 2345.56


    def detect_intent(self, user_input):
        user_input = user_input.lower()
        for intent, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_input):
                    return intent
        return "unknown"

    def extract_amount(self, user_input):
        # match first number that is located in the string
        match = re.search(r"(\d+(\.\d{1,2})?)", user_input)
        if match:
            return float(match.group(1))
        else:
            return None

    def run(self):
        print("Bot: Hello! Welcome to Chatbot v1.0! How can I help you today?")

        while self.state != "Exit":
            user_input = input("You: ")

            if self.state == "Idle":
                intent = self.detect_intent(user_input)
                if intent == "check_balance":
                    self.state = "CheckBalance"
                elif intent == "withdraw":
                    self.state = "AskForWithdraw"
                elif intent == "end":
                    print("Bot: Thanks for banking with us. Bye!")
                    break
                else:
                    print("Bot: I can help you check your balance or withdraw money. What would you like to do?")

            if self.state == "CheckBalance":
                print(f"Your current balance is ${self.balance:.2f}.")
                self.state = "EndSession"

            if self.state == "AskForWithdraw":
                amount = self.extract_amount(user_input)
                if amount:
                    if amount <= self.balance:
                        self.balance -= amount
                        print(f"Bot: ${amount:.2f} has been withdrew from your account."
                              f" Your new balance is ${self.balance:.2f}.")
                    else:
                        print(f"Sorry, can't withdraw ${amount:.2f}, since your balance is ${self.balance:.2f}.")
                    self.state = "EndSession"
                else:
                    print(f"Bot: How much would you like to withdraw? E.g (cash out 100)")

            if self.state == "EndSession":
                intent = self.detect_intent(user_input)
                if intent == "continue":
                    self.state = "Idle"
                elif intent == "end":
                    self.state = "Exit"
                    print("Bot: Thanks for banking with us. Bye!")
                else:
                    print("Bot: I can help you check your balance or withdraw money. Do you want to proceed?")


if __name__ == '__main__':
    bot = ATMChatBot()
    bot.run()
