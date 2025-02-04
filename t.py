import s2
import torch
import torchvision
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import msgpack
from io import BytesIO
from multiprocessing import Pool
from PIL import Image
from tqdm import tqdm
import random
import time
import os
import warnings

# Define the type of training (A, B, or C)
TYPE = "C"
"""
TYPE A: Original paper
TYPE B: Multi-Task Learning (MTL) with 3 types of classification (S3, S16, S365)
TYPE C: MTL with one type of classification (S3)
"""

# Suppress warnings
warnings.filterwarnings("ignore")
sns.set(style="darkgrid")

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize lists and paths
info_data = []  # Stores information data
S2Manager_list = []  # Stores instances of S2CellManager
train_msg_path = "resources\images\mp16"  # Path to training images
val_msg_path = "resources\images\yfcc25600"  # Path to validation images
s2_path = "./resources/cell/s2"  # Path to S2 data
info_sheet_path = "./resources/cell/sheet"  # Path to information sheets
train_msg_path_list = os.listdir(train_msg_path)  # List of training image filenames
val_msg_path_list = os.listdir(val_msg_path)  # List of validation image filenames
img_size = 256  # Size of input images
partitionings = []  # Stores the number of classes for each task


def read_data():
    global info_data, S2Manager_list, partitionings

    # Read S2 data and populate S2Manager_list and partitionings
    s2_file_list = os.listdir(s2_path)
    for i in range(len(s2_file_list)):
        S2Manager = s2.S2CellManager()
        S2Manager.load_data(f"{s2_path}/{s2_file_list[i]}")
        S2Manager_list.append(S2Manager)
        partitionings.append(len(S2Manager.class_list))

    # Add additional classification based on the type
    if TYPE == "B":
        partitionings.extend([3, 16, 365])
    elif TYPE == "C":
        partitionings.append(3)

    # Read information sheets and store them in info_data
    info_file_list = os.listdir(info_sheet_path)
    for i in range(len(info_file_list)):
        info_sheet = pd.read_csv(f"{info_sheet_path}/{info_file_list[i]}")
        info_data.append(info_sheet)


def build_model():
    # Build ResNet50 model and modify the output layer for multiple tasks
    model = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.flatten = torch.nn.Flatten(start_dim=1)

    # Build multiple linear classifiers based on partitionings
    classifier = torch.nn.ModuleList([torch.nn.Linear(2048, partitionings[i]) for i in range(len(partitionings))])

    # Move model and classifiers to device (GPU if available)
    model = model.to(device)
    classifier = classifier.to(device)

    return model, classifier


# Function to resize image
def resize_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.convert("RGB")
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = torchvision.transforms.Resize([320, 320])(img)
    img = torch.reshape(img, (1, 3, 320, 320))
    return img


# Function to get message data (images and their corresponding labels)
def get_msg(msg_path, path):
    new_data = None
    img_id = []
    img_data = []
    with open(os.path.join(f"{msg_path}\{path}"), "rb") as f:
        unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024)
        for x in unpacker:
            if x is None:
                continue
            img_id.append(x["id"])
            img_data.append(x["image"])

        with Pool(processes=8) as pool:
            new_data = pool.map(resize_image, img_data)
        new_data = torch.cat(new_data, dim=0)

    new_data = preprocess(new_data)
    new_target = torch.Tensor()
    for i in range(len(info_data)):
        data_details = info_data[i][info_data[i]["id"].isin(img_id)]
        data_details = data_details.set_index("id")
        data_details = data_details.loc[img_id]["class"]
        data_details = torch.tensor(data_details.values)
        data_details = torch.unsqueeze(data_details, dim=0)
        new_target = torch.concat((new_target, data_details), dim=0)

    if TYPE == "B" or TYPE == "C":
        if TYPE == "B":
            name_list = ["S3_Label", "S16_Label", "S365_Label"]
        elif TYPE == "C":
            name_list = ["S3_Label"]
        for i in range(len(name_list)):
            data_details = info_data[0][info_data[0]["id"].isin(img_id)]
            data_details = data_details.set_index("id")
            data_details = data_details.loc[img_id][name_list[i]]
            data_details = torch.tensor(data_details.values)
            data_details = torch.unsqueeze(data_details, dim=0)
            new_target = torch.concat((new_target, data_details), dim=0)

    return new_data, new_target


# Function for analysis
def analysis(predict, predict_proba, target):
    acc = metrics.accuracy_score(target, predict)
    auc = metrics.roc_auc_score(target, predict_proba)
    recall = metrics.recall_score(target, predict)
    precision = metrics.precision_score(target, predict)
    f1 = metrics.f1_score(target, predict)
    cm = metrics.confusion_matrix(target, predict)

    return [acc, auc, recall, precision, f1, cm]


# Function to calculate distance
def distance(predict, target):
    distance_list = []
    for i in range(len(S2Manager_list)):
        distance = []
        for j in range(len(predict)):
            distance.append(S2Manager_list[i].get_distance(int(predict[i][j]), int(target[i][j])))
        distance_list.append(sum(distance) / len(distance))

    return distance_list


# Function to read image
def read_img(filename):
    filepath = os.path.join("./resources/im2gps3ktest", filename)
    if not os.path.isdir(filepath):
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = torchvision.transforms.Resize([img_size, img_size])(img)
            img = torch.reshape(img, (1, 3, img_size, img_size))
            return img


# Validation function
def val(model, classifier_list):
    # Batch size for validation
    batch_size = 128
    # Read images from validation directory
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

    distance_value = []
    for k in range(len(S2Manager_list)):
        target_pool = data_details.values

        predict_list = torch.Tensor()
        classifier = classifier_list[k]
        model.eval()
        classifier.eval()
        distance_list = []
        data = data_pool
        with torch.no_grad():
            while data.shape[0] > 0:
                X_test = data[:batch_size].to(torch.float32).to(device)
                data = data[batch_size:]

                output = model(X_test)
                output = classifier(output)
                predict = torch.argmax(output, dim=1).to("cpu")

                predict_list = torch.concat((predict_list, predict))

            for i in range(len(predict_list)):
                distance_list.append(S2Manager_list[k].test_score(target_pool[i][1], target_pool[i][2], int(predict_list[i]), mode="math"))

        distance_list = np.array(distance_list)
        result = [
            sum(distance_list <= 2500) / len(distance_list),
            sum(distance_list <= 750) / len(distance_list),
            sum(distance_list <= 200) / len(distance_list),
            sum(distance_list <= 25) / len(distance_list),
            sum(distance_list <= 1) / len(distance_list),
        ]
        print(result)
        distance_value.append((sum(distance_list) / len(distance_list)))

    return distance_value


# Training function
def train(epochs, batch_size, lr, check_point=False):
    def check_point_function():
        if check_point:
            check_point1 = {
                "loss_history": loss_history,
                "score_history": score_history,
                "epoch": epoch,
                "model": model,
                "classifier": classifier,
                "opt": opt,
                "data_pool": data_pool,
                "target_pool": target_pool,
                "msg_index": msg_index,
                "data_length": data_length,
                "loss_total": loss_total,
            }
            np.save("./check_point/check_point", check_point1)

    # Build model and optimizer
    model, classifier = build_model()
    opt = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
    torch.optim.lr_scheduler.StepLR(opt, 5, 0.5)
    # opt = torch.optim.Adam(model.parameters(), lr)
    opt.zero_grad()
    loss_function = torch.nn.CrossEntropyLoss()
    epoch = 0
    msg_index = 0
    data_length = 0
    loss_total = [0 for i in range(len(partitionings))]
    data_pool = torch.Tensor()
    target_pool = torch.Tensor()

    msg_path_list = train_msg_path_list

    loss_history = []
    score_history = []

    # Load checkpoint if available
    if check_point and len(os.listdir("./check_point")) > 0:
        print("Load CheckPoint")
        check_point1 = np.load("./check_point/check_point.npy", allow_pickle="TRUE").item()

        loss_history = check_point1["loss_history"]
        score_history = check_point1["score_history"]
        epoch = check_point1["epoch"]
        model = check_point1["model"]
        classifier = check_point1["classifier"]
        opt = check_point1["opt"]
        data_pool = check_point1["data_pool"]
        target_pool = check_point1["target_pool"]
        msg_index = check_point1["msg_index"]
        data_length = check_point1["data_length"]
        loss_total = check_point1["loss_total"]
        print(f"Epoch:{epoch}")

    # Perform validation
    val(model, classifier)

    # Training loop
    while epoch < epochs:
        start_time = time.time()
        tqdm_bar = tqdm(total=len(msg_path_list), initial=msg_index)
        random.shuffle(msg_path_list)
        while msg_index < len(msg_path_list):
            for _ in range(3):
                if msg_index >= len(msg_path_list):
                    break
                new_data = get_msg(train_msg_path, msg_path_list[msg_index])
                data_pool = torch.concat((data_pool, new_data[0]), axis=0)
                target_pool = torch.concat((target_pool, new_data[1]), axis=1)
                tqdm_bar.update()
                data_length += new_data[0].shape[0]
                msg_index = msg_index + 1

            random_index = torch.randperm(data_pool.shape[0])
            data_pool = data_pool[random_index].view(data_pool.size())
            target_pool = target_pool[:, random_index].view(target_pool.size())

            while data_pool.shape[0] >= batch_size:
                X_train = data_pool[:batch_size].to(device).to(torch.float32)
                Y_train = target_pool[:, :batch_size].to(device).to(torch.long)
                data_pool = data_pool[batch_size:]
                target_pool = target_pool[:, batch_size:]
                with torch.cuda.amp.autocast():
                    output_ = model(X_train)
                    output = [classifier[i](output_) for i in range(len(partitionings))]
                    losses = [loss_function(output[i], Y_train[i]) for i in range(len(partitionings))]
                    loss = sum(losses)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                loss_total = [loss_total[i] + losses[i].item() for i in range(len(partitionings))]

            if msg_index % 100 == 0:
                check_point_function()

        tqdm_bar.clear()
        end_time = time.time()
        loss_total = [loss_total[i] / len(partitionings) / data_length for i in range(len(partitionings))]
        loss_history.append(loss_total)
        print(f"Epoch {epoch}, loss {loss_total}, time {end_time-start_time}s")

        if epoch % 1 == 0:
            model.eval()
            classifier.eval()
            score = val(model, classifier)
            score_history.append(score)
            model.train()
            classifier.train()

            if save_model(score_history):
                torch.save(model, "./output/best_model.pt")
                torch.save(classifier, "./output/best_classifier.pt")

            score_history_df = pd.DataFrame(score_history)
            loss_history_df = pd.DataFrame(loss_history)

            sns.lineplot(data=score_history_df[range(score_history_df.shape[1])])
            plt.savefig("./output/score.png")
            plt.clf()

            sns.lineplot(data=loss_history_df[range(loss_history_df.shape[1])])
            plt.savefig("./output/loss.png")
            plt.clf()

        epoch += 1
        msg_index = 0
        data_length = 0

        loss_total = [0 for i in range(len(partitionings))]

        check_point_function()


# Function to determine if the model should be saved
def save_model(score):
    if len(score) == 1:
        return True
    score_avg = None
    for i in range(len(score)):
        if score_avg is None:
            score_avg = sum(score[i]) / len(score[i])
        else:
            score_avg = min(score_avg, sum(score[i]) / len(score[i]))
            if i == len(score) - 1 and score_avg == sum(score[i]) / len(score[i]):
                return True
    return False


# Entry point of the script
if __name__ == "__main__":
    # Read data and start training
    read_data()
    train(40, 64, 0.01, True)  # Epochs, batch size, learning rate, checkpoint
