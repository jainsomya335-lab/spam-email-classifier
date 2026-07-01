import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "text": [
    "congratulations! you have won a $4000 Walmart gift card. click here to claim now!",
    "URGENT: Your account has been suspended. Verify your details immediately.",
    "HEY, are we still meeting for lunch tommorow?",
    "Can you send me the Report before end of day?",
    "FREE entry in a weekly competition to win an IPHONE.Text WIN to 80086 now!",
    "Reminder: your dentist appointment is scheduled for Monday at 10am.",
        "You have been selected for a cash prize of $5000. Reply YES to claim.",
        "Don't forget to bring your laptop for the meeting today.",
        "Limited time offer! Buy one get one free on all products. Shop now!",
        "Mom, I'll be home late tonight, don't wait up for dinner.",
        "Claim your free vacation to the Bahamas now, click the link below!",
        "Project deadline has been moved to next Friday, please update your schedule.",
        "WINNER!! As a valued customer you have been selected to receive a reward.",
        "Let's catch up this weekend, it's been a while!",
        "Get cheap loans approved instantly, no credit check required!",
        "Please find attached the invoice for last month's services.",
        "You've been chosen for a free trial of our weight loss pills, act fast!",
        "Happy birthday! Hope you have a wonderful day.",
        "Your package could not be delivered. Click here to reschedule.",
        "Team meeting moved to 3 PM in conference room B.",
        "Earn $5000 a week working from home, no experience needed!",
        "Can you review my code before I push it to GitHub?",
        "Act now! Your subscription is about to expire, renew with 50% discount.",
        "Thanks for your help yesterday, really appreciate it.",
        "Congratulations, you've been pre-approved for a $10,000 loan!",
        "Lunch is on me today, let's go to that new restaurant.",
        "Click here to verify your bank account or it will be locked.",
        "Looking forward to seeing you at the conference next week.",
        "You have an unclaimed inheritance waiting, contact us to process it.",
        "Could you pick up some groceries on your way home?",
   ],
   "label": [
        "spam", "spam", "ham", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham",
   ],
}

df = pd.DataFrame(data)
print(f"Dataset size: {len(df)} messages")
print(df["label"].value_counts(),"\n")


X_train , X_test, Y_train,Y_test = train_test_split(
  df["text"],df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)


vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec,Y_train)
Y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassificaton Report:\n", classification_report(Y_test,Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

sample_messages = [
  "Win a free iphone now, click this link!",
  "Hey, can we reschedule our meeting to 4PM?",

]
sample_vec = vectorizer.transform(sample_messages)
predictions = model.predict(sample_vec)

print("\n--- Custom predictions ---")
for msg, pred in zip(sample_messages, predictions):
    print(f"[{pred.upper()}] {msg}")

