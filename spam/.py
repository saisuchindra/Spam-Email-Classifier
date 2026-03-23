# LG-9 Spam Email Classifier (Fixed PDF Reader + Confidence Graph)

import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber
from tkinter import Tk, Label, Button, Text, Frame, END
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_PATH = "lg9_pdf_model.pkl"

# -------------------------------
# Utility Functions
# -------------------------------
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"http\S+|\S+@\S+|[^a-z0-9 ]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def load_pdf_dataset(pdf_path):
    """Extract ham/spam messages from PDF dataset"""
    messages, labels = [], []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("ham"):
                        labels.append(0)
                        messages.append(clean_text(line[3:]))
                    elif line.lower().startswith("spam"):
                        labels.append(1)
                        messages.append(clean_text(line[4:]))
        print(f"✅ Loaded {len(messages)} messages from PDF")
    except Exception as e:
        print("Error reading PDF:", e)
    return messages, labels

# -------------------------------
# Classifier Wrapper
# -------------------------------
class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.model = MultinomialNB()
        self.trained = False

    def train(self, X_texts, y):
        X = self.vectorizer.fit_transform([clean_text(x) for x in X_texts])
        self.model.fit(X, y)
        self.trained = True
        return X, y

    def evaluate(self, X_texts, y):
        X = self.vectorizer.transform([clean_text(x) for x in X_texts])
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        return acc, p, r, f

    def predict_proba(self, txt):
        if not self.trained:
            raise Exception("Model not trained!")
        X = self.vectorizer.transform([clean_text(txt)])
        return self.model.predict_proba(X)[0]

    def save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"vec": self.vectorizer, "model": self.model}, f)

    def load(self):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vec"]
            self.model = data["model"]
            self.trained = True

# -------------------------------
# GUI Application
# -------------------------------
class SpamClassifierGUI:
    def __init__(self, root, dataset_path):
        self.root = root
        self.classifier = SpamClassifier()
        self.dataset_path = dataset_path
        self.messages, self.labels = [], []
        self._setup_gui()

    def _setup_gui(self):
        self.root.title("📧 LG-9 Spam Email Classifier")
        self.root.geometry("850x650")

        Label(self.root, text="LG-9 Spam Classifier", font=("Arial", 20, "bold")).pack(pady=10)

        Button(self.root, text="Load PDF Dataset", command=self.load_dataset, width=20).pack(pady=5)
        Button(self.root, text="Train Model", command=self.train_model, width=20).pack(pady=5)

        Label(self.root, text="Enter Email Text:", font=("Arial", 12)).pack(pady=5)
        self.entry = Text(self.root, width=90, height=6)
        self.entry.pack(pady=5)

        Button(self.root, text="Classify Email", command=self.classify, bg="lightblue").pack(pady=10)

        self.result_label = Label(self.root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=10)

        self.chart_frame = Frame(self.root)
        self.chart_frame.pack(pady=10)

    def load_dataset(self):
        self.messages, self.labels = load_pdf_dataset(self.dataset_path)
        if not self.messages:
            messagebox.showerror("Error", "No data loaded from PDF!")
        else:
            messagebox.showinfo("Info", f"Loaded {len(self.messages)} messages from PDF")

    def train_model(self):
        if not self.messages:
            messagebox.showwarning("Warning", "Please load dataset first!")
            return
        X, y = self.classifier.train(self.messages, self.labels)
        acc, p, r, f = self.classifier.evaluate(self.messages, self.labels)
        self.classifier.save()
        messagebox.showinfo("Training Results",
                            f"✅ Training Complete\n\nAccuracy: {acc:.2f}\nPrecision: {p:.2f}\nRecall: {r:.2f}\nF1: {f:.2f}")

    def classify(self):
        txt = self.entry.get("1.0", END).strip()
        if not txt:
            messagebox.showwarning("Warning", "Please enter text to classify!")
            return
        try:
            proba = self.classifier.predict_proba(txt)
            spam_prob, ham_prob = proba[1] * 100, proba[0] * 100
            if spam_prob > ham_prob:
                self.result_label.config(text=f"🚨 SPAM ({spam_prob:.1f}%)", fg="red")
            else:
                self.result_label.config(text=f"✅ HAM ({ham_prob:.1f}%)", fg="green")
            self.show_chart(spam_prob, ham_prob)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_chart(self, spam, ham):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Ham", "Spam"], [ham, spam], color=["green", "red"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability (%)")
        ax.set_title("Classification Confidence")
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    DATASET_PATH = "HAMSAPM.pdf"  # your dataset file
    root = Tk()
    app = SpamClassifierGUI(root, DATASET_PATH)
    root.mainloop()
