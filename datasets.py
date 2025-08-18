#classe FruitDataset

import os, json, cv2, torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

CLASS_MAP = {
    "apple": 1,
    "blueberry": 2,
    "cherry": 3,
    "kiwi": 4,
    "orange": 5,
    "strawberry": 6,
}

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform if transform else ToTensor()

        # scansiona classi
        for cls in CLASS_MAP.keys():
            img_dir = os.path.join(root_dir, cls, "img")
            ann_dir = os.path.join(root_dir, cls, "label")

            for fname in os.listdir(img_dir):
                if fname.endswith(".jpg"):
                    stem = fname[:-4]
                    img_path = os.path.join(img_dir, fname)
                    ann_path = os.path.join(ann_dir, stem + ".json")
                    if os.path.exists(ann_path):
                        self.samples.append((img_path, ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # carica immagine
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        with open(ann_path, "r") as f:
            data = json.load(f)

        boxes, labels = [], []
        for obj in data["objects"]:
            cls = CLASS_MAP[obj["classTitle"]]
            (xmin, ymin), (xmax, ymax) = obj["points"]["exterior"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        image = self.transform(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))
