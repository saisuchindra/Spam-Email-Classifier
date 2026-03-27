# Spam Email Classifier

![Spam Classifier Banner](https://img.shields.io/badge/Spam%20Classifier-Python-blueviolet?style=for-the-badge&logo=python)

A modern, interactive, and accurate spam email classifier with PDF dataset support, confidence graph, and a beautiful Tkinter GUI.

---

## 🚀 Features

- **PDF Dataset Loader**: Extracts ham/spam messages directly from a PDF file.
- **Easy Training**: Train a Naive Bayes model with a single click.
- **Live Classification**: Classify any email text and get instant results.
- **Confidence Graph**: Visual bar chart shows classification confidence.
- **History & Feedback**: User-friendly error and info dialogs.
- **Modern GUI**: Clean, responsive Tkinter interface.

---

## 🖼️ App Preview

![App Screenshot](https://user-images.githubusercontent.com/your-username/spam-classifier-demo.png)

---

## 🛠️ Requirements

- Python 3.7+
- `numpy`
- `matplotlib`
- `pdfplumber`
- `scikit-learn`

Install dependencies:
```bash
pip install numpy matplotlib pdfplumber scikit-learn
```

---

## 📂 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saisuchindra/Spam-Email-Classifier.git
   cd Spam-Email-Classifier
   ```
2. **Add your PDF dataset:**
   - Place your dataset (e.g., `HAMSAPM.pdf`) in the project folder.
3. **Run the app:**
   ```bash
   python spam/.py
   ```

---

## 📝 PDF Dataset Format
- Each line in the PDF should start with `ham` or `spam` followed by the message text.
- Example:
  ```
  ham Hello, how are you?
  spam Congratulations! You won a prize.
  ```

---

## 📊 Model
- Uses **TF-IDF** vectorization and **Multinomial Naive Bayes** classifier.
- Model is saved as `lg9_pdf_model.pkl` after training.

---

## 🎨 GUI Highlights
- Colorful result labels (red for spam, green for ham)
- Interactive confidence bar chart
- Modern, clean layout

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
[MIT](LICENSE)

---

> Made with ❤️ by [saisuchindra](https://github.com/saisuchindra)
