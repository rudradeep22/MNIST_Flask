from flask import Flask, render_template, send_file
import numpy as np
from PIL import Image
from model import generate_image, generator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    generated_image = generate_image(generator)
    image = Image.fromarray((generated_image * 255).astype(np.uint8))
    image.save('static/generated_image.png')
    return send_file('static/generated_image.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
