import torch
from torchvision import transforms
from PIL import Image
from models.MyCNN import ResNet50
from utils.utils import load_checkpoint
import torch.nn.functional as F 
import cv2
from detect_face import infer_image
import numpy as np
import os

def align_face(img, landmark, output_size=(160,160)):
    # 標準人臉五點模板 (對應 112x112 或 160x160 尺寸)
    # 這裡用 ArcFace 的 5點模板
    ref_landmarks = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # 根據輸出大小縮放
    ref_landmarks = ref_landmarks * (output_size[0] / 112.0)

    # landmark 轉 numpy
    src = np.array(landmark, dtype=np.float32).reshape(5, 2)

    # 計算仿射矩陣
    M, _ = cv2.estimateAffinePartial2D(src, ref_landmarks, method=cv2.LMEDS)

    # 對齊臉
    aligned_face = cv2.warpAffine(img, M, output_size, borderValue=0.0)

    return aligned_face

def crop_face(img, box, size=112, margin=0):
    h, w, _ = img.shape
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    x1 = int(max(0, x1 - margin))
    y1 = int(max(0, y1 - margin))
    x2 = int(min(w, x2 + margin))
    y2 = int(min(h, y2 + margin))

    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (size, size))
    return face

def get_embedding(face_img, transform, device, model):
    
    image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x)

    return F.normalize(emb.squeeze(), p=2, dim=0).cpu()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  #隨機切片
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), #模糊
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  #調色溫 亮度 對比
        transforms.RandomGrayscale(p=0.1),      #灰階
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)), #隨機抹去
    ])

    path = 'models/MyCnn.pth'
    out_path = 'infer_face_data'
    model = ResNet50()

    load_checkpoint(device=device,optimizer=None,model=model,path=path)


    model.to(device)
    model.eval()

    face_db = {}
    people = {
        'Ken': [
            'FaceDataBase/face_data/Ken01.jpg','FaceDataBase/face_data/Ken06.jpg','FaceDataBase/face_data/Ken02.jpg','FaceDataBase/face_data/Ken07.jpg',
            'FaceDataBase/face_data/Ken03.jpg','FaceDataBase/face_data/Ken08.jpg','FaceDataBase/face_data/Ken19.jpg','FaceDataBase/face_data/Ken20.jpg',
            'FaceDataBase/face_data/Ken04.jpg','FaceDataBase/face_data/Ken09.jpg','FaceDataBase/face_data/Ken05.jpg','FaceDataBase/face_data/Ken10.jpg',
            'FaceDataBase/face_data/Ken11.jpg','FaceDataBase/face_data/Ken12.jpg','FaceDataBase/face_data/Ken13.jpg','FaceDataBase/face_data/Ken14.jpg',
            'FaceDataBase/face_data/Ken15.jpg','FaceDataBase/face_data/Ken16.jpg','FaceDataBase/face_data/Ken17.jpg','FaceDataBase/face_data/Ken18.jpg',
            'FaceDataBase/face_data/Ken21.jpg','FaceDataBase/face_data/Ken22.jpg','FaceDataBase/face_data/Ken23.jpg','FaceDataBase/face_data/Ken24.jpg',
            'FaceDataBase/face_data/Ken25.jpg','FaceDataBase/face_data/Ken26.jpg','FaceDataBase/face_data/Ken27.jpg','FaceDataBase/face_data/Ken28.jpg',
            'FaceDataBase/face_data/Ken29.jpg','FaceDataBase/face_data/Ken30.jpg','FaceDataBase/face_data/Ken31.jpg',
        ],
        #'Tseng': ['FaceDataBase/face_data/Tseng01.jpg']
    }
    id = 1
    for name, imgs in people.items():
        embeddings = []
        for p in imgs:
            img = cv2.imread(p)
            boxes,landmark = infer_image(img)
            if boxes is None:
                print(f"[WARN] {p} 沒有偵測到臉")
                continue
            
            face = align_face(img, landmark[0], output_size=(112,112))
            #face = crop_face(img, boxes[0]) 
            cv2.imwrite(os.path.join(out_path,f"{id}.jpg"),face)
            id += 1
            e = get_embedding(face, transform, device, model)
            embeddings.append(e)

        if embeddings is not None:
            avg_emb = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=0)
            face_db[name] = avg_emb

    torch.save(face_db, 'FaceDataBase/face_db.pt')
    print("[INFO] face_db.pt 已建立完成 ✅")

if __name__ == '__main__':
    main()
