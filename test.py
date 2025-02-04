import os
from PIL import Image
import torch
import torchvision
import s2
import pandas as pd
from train import build_model, distance, read_data, preprocess
from multiprocessing import Pool
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

s2_path = "./resources/cell/s2"
img_size = 256
batch_size = 1


def read_img(filename):
    filepath = os.path.join("./resources/im2gps3ktest", filename)
    if not os.path.isdir(filepath):
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = torchvision.transforms.Resize([img_size, img_size])(img)
            img = torch.reshape(img, (1, 3, img_size, img_size))
            return img


def main():
    s2_list = os.listdir("./resources/cell/s2")
    with Pool(processes=8) as p:
        data_pool = p.map(read_img, os.listdir("./resources/im2gps3ktest"))

    data_pool = torch.concat(data_pool, dim=0)
    data_pool = preprocess(data_pool)

    img_id = os.listdir("./resources/im2gps3ktest")
    info_data = pd.read_csv("./resources/im2gps3k_places365.csv")
    data_details = info_data[info_data["IMG_ID"].isin(img_id)]
    data_details = data_details.set_index("IMG_ID")
    data_details = data_details.loc[img_id][["LAT", "LON"]]
    data_details.insert(0, "_", range(len(img_id)))

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.FiveCrop(224),
        ]
    )

    model = classifier = build_model()
    # check_point = np.load("./check_point/check_point.npy", allow_pickle="TRUE").item()
    check_point = np.load("./check_point/check_point.npy", allow_pickle="TRUE").item()
    # model = check_point["model"].to(device)
    # classifier_list = check_point["classifier"].to(device)
    model = torch.load("./output/best_model.pt").to(device)
    classifier_list = torch.load("./output/best_classifier.pt").to(device)
    for k in range(len(s2_list)):
        S2Manager = s2.S2CellManager()
        S2Manager.load_data(f"./resources/cell/s2/{s2_list[k]}")
        target_pool = data_details.values

        read_data()

        predict_list = []
        classifier = classifier_list[k]
        model.eval()
        classifier.eval()
        distance_list = []
        data = data_pool
        with torch.no_grad():
            while data.shape[0] > 0:
                X_test = data[:batch_size].to(torch.float32).to(device)
                data = data[batch_size:]
                X_test = transform(X_test)
                X_test = torch.cat(X_test, dim=0)

                output = model(X_test)
                output = classifier(output)
                output = torch.nn.functional.softmax(output, dim=1)
                output = torch.mean(output, dim=0)
                predict = torch.argmax(output).to("cpu")

                predict_list.append(predict)

            for i in range(len(predict_list)):
                distance_list.append(S2Manager.test_score(target_pool[i][1], target_pool[i][2], int(predict_list[i]), mode="math"))

        distance_list = np.array(distance_list)
        result = [
            sum(distance_list <= 2500) / len(distance_list),
            sum(distance_list <= 750) / len(distance_list),
            sum(distance_list <= 200) / len(distance_list),
            sum(distance_list <= 25) / len(distance_list),
            sum(distance_list <= 1) / len(distance_list),
        ]
        print(result)


if __name__ == "__main__":
    main()
