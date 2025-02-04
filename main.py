import download_images
import train
import create_s2
import test

import os


def create_file():
    if os.path.exists("./check_point") == False:
        os.mkdir("./check_point")
    if os.path.exists("./resources") == False:
        os.mkdir("./resources")
    if os.path.exists("./resources/cell") == False:
        os.mkdir("./resources/cell")
    if os.path.exists("./resources/cell/s2") == False:
        os.mkdir("./resources/cell/s2")
    if os.path.exists("./resources/cell/sheet") == False:
        os.mkdir("./resources/cell/sheet")
    if os.path.exists("./output") == False:
        os.mkdir("./output")


if __name__ == "__main__":
    create_s2.create_s2(50, 1000)
    create_s2.create_s2(50, 2000)
    create_s2.create_s2(50, 5000)
    train.read_data()
    train(20, 64, 0.001, 0.8, False)
    train.train(epochs=20, batch_size=64, lr=0.001, train_size=0.8, check_point=False)
