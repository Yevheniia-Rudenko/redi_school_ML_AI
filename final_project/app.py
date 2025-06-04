from flask import Flask, render_template_string
import os

app = Flask(__name__)

# Относительный путь к index.html
file_path = os.path.join(os.path.dirname(__file__), "index.html")

with open(file_path, "r", encoding="utf-8") as f:
    html_content = f.read()


@app.route("/")
def home():
    return render_template_string(html_content)


if __name__ == "__main__":
    app.run(debug=True)
