import pickle
import matplotlib.pyplot as plt
import glob
import math

for file in glob.glob("histories/*.pkl"):
    with open(file, 'rb') as inf:
        history = pickle.load(inf)
    # print(history.keys())
    key_dict = {"lrs": "Learning-Rate-Scheduler", "bsize_64": "Batchsize 64", "dyn_aug": "Dynamic Augmentation", "es": "Early Stopping"}
    title = "Model loss" + "without" if "wo" in file else "with" + " " + key_dict[file.split("/")[-1].split("_")[1]] #+ " and " + key_dict[file.split("/")[-1].split("_")[2]]
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title("Model Loss ohne Optimierungen")
    plt.ylabel('Loss')
    plt.xlabel('Epoche')
    # plt.xscale('log', base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]) # , nonpositive='clip', linthresh=0.01, linscale=0.1)
    plt.yscale('log', base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]) # , nonpositive='clip', linthresh=0.01, linscale=0.1)
    # plt.xticks([1, 2, 5, 10, 20, 50], ['1', '2', '5', '10', '20', '50'])
    plt.yticks([0.1, 0.2, 0.5, 1, 2], ['0.1', '0.2', '0.5', '1', '2'])
    #add grid
    plt.grid(which='both')
    plt.legend() # ['train', 'test'], loc='upper left')
    plt.savefig(file.replace(".pkl", "log.png"))
    plt.clf()
    # plt.plot(history['learning_rate'], label='learning rate')
    # plt.title('Learning Rate')
    # plt.ylabel('learning rate')
    # plt.xlabel('epoch')
    # plt.yscale('log', base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]) # , nonpositive='clip', linthresh=0.01, linscale=0.1)
    # plt.yticks([0.001, 0.002, 0.005, 0.01, 0.02], ['0.001', '0.002', '0.005', '0.01', '0.02'])
    # plt.grid(which='both')
    # plt.legend()
    # plt.savefig(file.replace(".pkl", "lr.png"))
    # plt.clf()

