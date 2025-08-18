import os, cv2, torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets import CLASS_MAP

def get_model(num_classes, weights_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_prediction(img_path, outputs, out_dir="predictions", score_thr=0.5):
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    boxes = outputs[0]["boxes"].detach().cpu().numpy()
    scores = outputs[0]["scores"].detach().cpu().numpy()
    labels = outputs[0]["labels"].detach().cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cls_name = [k for k, v in CLASS_MAP.items() if v == label][0]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{cls_name}:{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"✅ Salvata immagine con predizioni in {out_path}")

if __name__ == "__main__":
    weights_path = "model_frcnn.pth"
    images_dir = "./images"
    out_dir = "predictions"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1 + len(CLASS_MAP)

    model = get_model(num_classes, weights_path, device)

    # scorri tutte le immagini in images/
    for fname in os.listdir(images_dir):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(images_dir, fname)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_tensor = torchvision.transforms.ToTensor()(image).to(device)

        outputs = model([image_tensor])
        save_prediction(img_path, outputs, out_dir=out_dir, score_thr=0.5)

    print("🎉 Tutte le immagini di test sono state processate!")
