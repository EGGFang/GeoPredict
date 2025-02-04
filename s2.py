import s2sphere
import json
import math


class S2CellManager:
    def __init__(self, min_tau=50, max_tau=2000):
        # Initialize S2CellManager with minimum and maximum image counts
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.cell_dict = {}  # Dictionary to store cells and associated data
        self.class_list = []  # List to store cell IDs
        self.center_lat_lng = None  # Center lat and lng in cells

        # Initialize the base cell
        self.init_cell()

    def init_cell(self):
        # Initialize the base cell with face at 0 to 5
        for i in range(6):
            cell_id = s2sphere.CellId.from_face_pos_level(i, 0, 0)
            self.cell_dict[str(cell_id.to_token())] = []  # Add an empty list for base cell

    def create_cell(self, data):
        # Create cells and insert data based on input dataset
        for item in data:
            data_id, lat, lng = item[0], item[1], item[2]  # Extract data [id, lat, lng]

            self.insert_cell(data_id, lat, lng)

        self.class_list = list(self.cell_dict.keys())  # Update the class_listS

    def insert_cell(self, data_id, lat, lng):
        data_lat_lng = s2sphere.LatLng.from_degrees(lat, lng)
        # Insert data into appropriate cell
        cell_dict_key = self.cell_dict.keys()
        data_cell_id = s2sphere.CellId.from_lat_lng(data_lat_lng).parent(30)
        for cell_key in cell_dict_key:
            cell_id = s2sphere.CellId.from_token(cell_key)
            if cell_id.contains(data_cell_id):
                # Insert data into the cell
                self.cell_dict[cell_key].append([str(data_id), lat, lng])

                # Check if the cell exceeds the maximum limit and needs to be split
                if len(self.cell_dict[cell_key]) > self.max_tau and cell_id.level() < 30:
                    # Split the cell into child cells
                    child_cells = cell_id.children(cell_id.level() + 1)
                    for child_cell_id in child_cells:
                        self.cell_dict[child_cell_id.to_token()] = []

                    cell_value = self.cell_dict[cell_key]
                    del self.cell_dict[cell_key]
                    for d in cell_value:
                        # Recursively insert data into child cells
                        self.insert_cell(d[0], d[1], d[2])
                return

    def get_lat_lng(self, class_id):
        # Get the center and vertices of a cell based on class ID
        cell_id = self.class_list[class_id]
        cell_id = s2sphere.CellId.from_token(cell_id)
        cell = s2sphere.Cell(cell_id)
        cell_center_point = cell.get_center()
        cell_center = s2sphere.LatLng.from_point(cell_center_point)
        vertices = [cell.get_vertex(i) for i in range(4)]
        return [cell_center.lat().degrees, cell_center.lng().degrees], vertices

    def get_distance(self, class1, class2, mode="image"):
        if mode == "cell":
            class1_point = s2sphere.Cell(s2sphere.CellId.from_token(self.class_list[class1])).get_center()
            class2_point = s2sphere.Cell(s2sphere.CellId.from_token(self.class_list[class2])).get_center()
        elif mode == "image":
            if self.center_lat_lng is None:
                self.__calc_cell_center_latlng()
            lat1, lng1 = self.center_lat_lng[class1]
            lat2, lng2 = self.center_lat_lng[class2]
            class1_point = s2sphere.LatLng.from_degrees(lat1, lng1)
            class2_point = s2sphere.LatLng.from_degrees(lat2, lng2)
        elif mode == "math":
            if self.center_lat_lng is None:
                self.__calc_cell_center_latlng()
            lat1, lng1 = self.center_lat_lng[class1]
            lat2, lng2 = self.center_lat_lng[class2]

            R = 6371
            factor_rad = 0.01745329252
            longitudes = factor_rad * lng1
            longitudes_gt = factor_rad * lng2
            latitudes = factor_rad * lat1
            latitudes_gt = factor_rad * lat2
            delta_long = longitudes_gt - longitudes
            delta_lat = latitudes_gt - latitudes
            subterm0 = math.sin(delta_lat / 2) ** 2
            subterm1 = math.cos(latitudes) * math.cos(latitudes_gt)
            subterm2 = math.sin(delta_long / 2) ** 2
            subterm1 = subterm1 * subterm2
            a = subterm0 + subterm1
            c = 2 * math.asin(math.sqrt(a))
            gcd = R * c

            return gcd
        else:
            return None
        distance_in_km = class1_point.get_distance(class2_point).degrees * 111.32

        return distance_in_km
    
    def test_score(self,lat,lng,class_id, mode="image"):
        if self.center_lat_lng is None:
                self.__calc_cell_center_latlng()
        lat1, lng1 = lat,lng
        lat2, lng2 = self.center_lat_lng[class_id]

        if mode=="image":
            class1_point = s2sphere.LatLng.from_degrees(lat1, lng1)
            class2_point = s2sphere.LatLng.from_degrees(lat2, lng2)
            distance_in_km = class1_point.get_distance(class2_point).degrees * 111.32
            return distance_in_km
        elif mode=="math":
            R = 6371
            factor_rad = 0.01745329252
            longitudes = factor_rad * lng1
            longitudes_gt = factor_rad * lng2
            latitudes = factor_rad * lat1
            latitudes_gt = factor_rad * lat2
            delta_long = longitudes_gt - longitudes
            delta_lat = latitudes_gt - latitudes
            subterm0 = math.sin(delta_lat / 2) ** 2
            subterm1 = math.cos(latitudes) * math.cos(latitudes_gt)
            subterm2 = math.sin(delta_long / 2) ** 2
            subterm1 = subterm1 * subterm2
            a = subterm0 + subterm1
            c = 2 * math.asin(math.sqrt(a))
            gcd = R * c
            return gcd

        

    def get_class(self, img_id, lat, lng,S3_Label,S16_Label,S365_Label):
        # Get the class ID based on latitude and longitude
        data_lat_lng = s2sphere.LatLng.from_degrees(lat, lng)
        data_cell_id = s2sphere.CellId.from_lat_lng(data_lat_lng)
        for j in range(len(self.class_list)):
            cell_id = s2sphere.CellId.from_token(self.class_list[j])
            if cell_id.contains(data_cell_id):
                return j

    def __calc_cell_center_latlng(self):
        self.center_lat_lng = []
        for cellid in self.class_list:
            cell_value = self.cell_dict[cellid]
            lat_total = 0.0
            lng_total = 0.0
            for item in cell_value:
                lat_total += item[1]
                lng_total += item[2]

            self.center_lat_lng.append([lat_total / len(cell_value), lng_total / len(cell_value)])

    def delete_cell(self):
        cell_dict_key = list(self.cell_dict.keys())
        for cell_key in cell_dict_key:
            if len(self.cell_dict[cell_key]) < self.min_tau:
                del self.cell_dict[cell_key]
        self.class_list = list(self.cell_dict.keys())

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            self.cell_dict = json.load(f)
            self.class_list = list(self.cell_dict.keys())
            # self.__calc_cell_center_latlng()  # Update the center_lat_lng


# Example usage:
if __name__ == "__main__":
    dataset = [
        [1, 40.7128, -74.0060],
        [2, 34.0522, -118.2437],
        [3, 51.5074, -0.1278],
        [4, 41.8781, -87.6298],
        [5, 37.7749, -122.4194],
        [6, 48.8566, 2.3522],
        [7, 33.6844, -117.8265],
        [8, 34.0522, -118.2437],
        [9, 40.7128, -74.0060],
        [10, 34.0522, -118.2437],
        [11, 51.5074, -0.1278],
        [12, 41.8781, -87.6298],
        [13, 37.7749, -122.4194],
        [14, 48.8566, 2.3522],
        [15, 33.6844, -117.8265],
        [16, 40.7128, -74.0060],
        [17, 34.0522, -118.2437],
        [18, 51.5074, -0.1278],
        [19, 41.8781, -87.6298],
        [20, 37.7749, -122.4194],
    ]

    min_tau = 5  # Minimum number of images per cell
    max_tau = 10  # Maximum number of images per cell

    manager = S2CellManager(min_tau, max_tau)
    classes = manager.create_cell(dataset)

    for i, cell_id in enumerate(classes):
        center, vertices = manager.get_lat_lng(i)
        print(f"Class {i}: Center Coordinates - Latitude: {center[0]}, Longitude: {center[1]}")

    test_lat, test_lng = 40.7128, -74.0060  # Test coordinates
    class_id = manager.get_class(test_lat, test_lng)
    print(f"Class ID for ({test_lat}, {test_lng}): {class_id}")
