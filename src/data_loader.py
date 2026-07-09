import os
import pickle

class DEAPDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_subject_files(self):
        files = sorted(
            [f for f in os.listdir(self.dataset_path) if f.endswith(".dat")]
        )
        return files

    def load_subject(self, filename):
        file_path = os.path.join(self.dataset_path, filename)

        with open(file_path, "rb") as file:
            subject = pickle.load(file, encoding="latin1")

        return subject


if __name__ == "__main__":

    dataset_path = "dataset/data_preprocessed_python"

    loader = DEAPDataLoader(dataset_path)

    files = loader.get_subject_files()

    print("=" * 50)
    print("DEAP DATASET LOADED")
    print("=" * 50)

    print(f"Total Subjects : {len(files)}")

    subject = loader.load_subject(files[0])

    print("\nKeys : ", subject.keys())

    print("\nLabels Shape :", subject["labels"].shape)

    print("Data Shape :", subject["data"].shape)