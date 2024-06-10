import pickle
import matplotlib.pyplot as plt
import glob

for file in glob.glob("histories/*.pkl"):
    with open(file, 'rb') as inf:
        history = pickle.load(inf)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend() # ['train', 'test'], loc='upper left')
    plt.savefig(file.replace(".pkl", ".png"))
    plt.clf()

