import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from google.colab import files
from io import BytesIO
from PIL import Image

model = keras.applications.VGG16()

uploaded = files.upload()
img = Image.open(BytesIO(uploaded['ex224.jpg']))
plt.imshow( img )

# приводим к входному формату VGG-сети
img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

# прогоняем через сеть
res = model.predict( x )
print(np.argmax(res))
