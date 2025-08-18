# train_frcnn.py
import os, argparse, torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from datasets import FruitDataset, collate_fn, CLASS_MAP

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="COCO_V1"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # dataset di training
    train_ds = FruitDataset(root_dir=args.train_dir, transform=None)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn
    )

    # modello
    num_classes = 1 + len(CLASS_MAP)  # background + 6 frutti
    model = get_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, loss={running/len(train_loader):.4f}")

    # salva modello
    torch.save(model.state_dict(), args.out)
    print(f"✅ Training finito, modello salvato in {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", type=str, default="./train")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=str, default="model_frcnn.pth")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    train(args)
