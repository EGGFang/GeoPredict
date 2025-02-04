import s2
import pandas as pd
import os
import msgpack
import time
import json
import torch
import torchvision
from PIL import Image
from io import BytesIO
from multiprocessing import Pool

sheet_path = "resources\mp16_places365.csv"
msg_path = "resources\images\mp16"
img_size = 320


def resize_image(image_data):
    img = Image.open(BytesIO(image_data))
    img = img.convert("RGB")
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = torchvision.transforms.Resize([img_size, img_size])(img)
    img = torch.reshape(img, (3, img_size, img_size))
    return img


def get_msg(path):
    new_data = []
    img_id = []
    img_data = []
    with open(os.path.join(f"{msg_path}\{path}"), "rb") as f:
        unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024)
        for x in unpacker:
            if x is None:
                continue

            img_id.append(x["id"])
            img_data.append(x["image"])

        pool = Pool(processes=8)
        for x in pool.imap(resize_image, img_data):
            new_data.append(x[0])

        new_data = [x.tolist() for x in new_data]
        new_data = torch.tensor(new_data)

    torch.save(new_data, f"./resources/data/{path[:-4]}_img.pt")
    torch.save(img_id, f"./resources/data/{path[:-4]}_id.pt")

    print(f"{path} done")

    return img_id


def create_s2(min_tau=50, max_tau=2000):
    # Initialize S2 Cell Manager with specified thresholds
    S2Manager = s2.S2CellManager(min_tau, max_tau)

    data_dict = pd.DataFrame()
    data_sheet = pd.read_csv(sheet_path)

    msg_list = os.listdir(msg_path)
    # Iterate through the messages in the path
    img_id = []
    for i in range(len(msg_list)):
        st = time.time()
        img_id.append(get_msg(msg_list[i]))
        print(time.time() - st)

    exit()
    img_details = data_sheet[data_sheet["IMG_ID"].isin(img_id)]
    img_details = img_details[["IMG_ID", "LAT", "LON"]]
    img_details = img_details.rename(columns={"IMG_ID": "id", "LAT": "lat", "LON": "lng"})
    data_dict = pd.concat([data_dict, img_details])
    # Create S2 cells from the gathered data
    S2Manager.create_cell(img_details.values.tolist())
    # Save filtered data to a CSV file
    data_dict.to_csv("./resources/filited_data.csv")
    # Save S2 cell data to a CSV file
    cell_dict = S2Manager.cell_dict
    with open("./resources/s2_cell.csv", "w") as f:
        json.dump(cell_dict, f)

    target = S2Manager.get_class(data_dict.values)
    data_dict = data_dict.insert(3, "class", target)

    # Save filtered data to a CSV file
    data_dict.to_csv("./resources/filited_data.csv", index=False)
    # Save S2 cell data to a CSV file
    cell_dict = S2Manager.cell_dict
    with open("./resources/s2_cell.csv", "w") as f:
        json.dump(cell_dict, f)


if __name__ == "__main__":
    create_s2(50, 5000)
