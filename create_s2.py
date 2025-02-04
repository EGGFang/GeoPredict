import s2
import pandas as pd
import os
import msgpack
import time
import json
from multiprocessing import Pool

sheet_path = "resources\mp16_places365.csv"
msg_path = "resources\images\mp16"


def create_s2(min_tau=50, max_tau=2000):
    # Initialize S2 Cell Manager with specified thresholds
    S2Manager = s2.S2CellManager(min_tau, max_tau)

    data_dict = pd.DataFrame()
    data_sheet = pd.read_csv(sheet_path)

    msg_list = os.listdir(msg_path)
    counter = 1
    # Iterate through the messages in the path
    for msg_file in msg_list:
        s_t = time.time()
        print(f"Running {msg_file}, ({counter}/{len(msg_list)})")
        with open(os.path.join(f"{msg_path}\{msg_file}"), "rb") as f:
            # Unpack the messages
            unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024)

            img_id = []
            # Iterate through unpacked data
            for x in unpacker:
                if x is None:
                    continue
                img_id.append(x["id"])

            img_details = data_sheet[data_sheet["IMG_ID"].isin(img_id)]
            img_details = img_details[["IMG_ID", "LAT", "LON","S3_Label","S16_Label","S365_Label"]]
            img_details = img_details.rename(columns={"IMG_ID": "id", "LAT": "lat", "LON": "lng"})
            data_dict = pd.concat([data_dict, img_details])
            # Create S2 cells from the gathered data
            S2Manager.create_cell(img_details.values.tolist())
        e_t = time.time()
        print(f"Spent {round(e_t-s_t)}s")
        if counter % 10 == 0:
            # Save filtered data to a CSV file
            data_dict.to_csv("./resources/filited_data.csv")
            # Save S2 cell data to a CSV file
            cell_dict = S2Manager.cell_dict
            with open("./resources/s2_cell.csv", "w") as f:
                json.dump(cell_dict, f)

        counter += 1
    S2Manager.delete_cell()
    with Pool(processes=8) as pool:
        target = pool.starmap(S2Manager.get_class, data_dict.values)
    data_dict.insert(3, "class", target)

    # Save filtered data to a CSV file
    data_dict.to_csv(f"./resources/cell/sheet/filited_data_{min_tau}_{max_tau}.csv", index=False)
    # Save S2 cell data to a CSV file
    cell_dict = S2Manager.cell_dict
    with open(f"./resources/cell/s2/s2_cell_{min_tau}_{max_tau}.csv", "w") as f:
        json.dump(cell_dict, f)


def get_val_data():
    s2_path_list = os.listdir("./resources/cell/s2")
    sheet_path_list = os.listdir("./resources/cell/sheet")
    msg_path_list = os.listdir("./resources//images/yfcc25600")
    data_sheet = pd.read_csv("./resources/yfcc25600_places365.csv")
    for i in range(len(s2_path_list)):
        S2Manager = s2.S2CellManager()
        S2Manager.load_data(f"./resources/cell/s2/{s2_path_list[i]}")
        data_dict = pd.read_csv(f"./resources/cell/sheet/{sheet_path_list[i]}")
        new_data_dict = pd.DataFrame()
        for msg_file in msg_path_list:
            with open(os.path.join(f"./resources/images/yfcc25600/{msg_file}"), "rb") as f:
                # Unpack the messages
                unpacker = msgpack.Unpacker(f, max_buffer_size=1024 * 1024 * 1024)
                img_id = []
                # Iterate through unpacked data
                for x in unpacker:
                    if x is None:
                        continue
                    img_id.append(x["id"])

                img_details = data_sheet[data_sheet["IMG_ID"].isin(img_id)]
                img_details = img_details[["IMG_ID", "LAT", "LON","S3_Label","S16_Label","S365_Label"]]
                img_details = img_details.rename(columns={"IMG_ID": "id", "LAT": "lat", "LON": "lng"})
                new_data_dict = pd.concat([new_data_dict, img_details])

        with Pool(processes=8) as pool:
            target = pool.starmap(S2Manager.get_class, new_data_dict.values)
        new_data_dict.insert(3, "class", target)
        data_dict = pd.concat([data_dict, new_data_dict])
        # Save filtered data to a CSV file
        data_dict.to_csv(f"./resources/cell/sheet/{sheet_path_list[i]}", index=False)


if __name__ == "__main__":
    # create_s2(50, 1000)
    create_s2(50, 2000)
    create_s2(50, 5000)
    get_val_data()
