import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from final.detect_face import infer_image
from final.utils.utils import load_checkpoint
from final.models.MyCNN import ResNet50
from torchvision import transforms
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- 模型載入 -----------------
model_path = os.path.join(BASE_DIR, "models")
pnet = torch.jit.load(os.path.join(model_path, 'PNet.pth')).to(device).eval()
rnet = torch.jit.load(os.path.join(model_path, 'RNet.pth')).to(device).eval()
onet = torch.jit.load(os.path.join(model_path, 'ONet.pth')).to(device).eval()

# MyCNN
mycnn_path = os.path.join(model_path, "MyCnn.pth")
model = ResNet50().to(device)
load_checkpoint(device=device, optimizer=None, model=model, path=mycnn_path)
model.eval()

# FSRCNN
srx4_path = os.path.join(model_path,"FSRCNN_x4.pb")
srx4 = cv2.dnn_superres.DnnSuperResImpl_create()
srx4.readModel(srx4_path)
srx4.setModel("fsrcnn", 4)

transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
face_db = torch.load(os.path.join(BASE_DIR, 'FaceDataBase/face_db.pt'), map_location='cpu')

# ----------------- 人臉比對 -----------------
def match_face(embedding, face_db, threshold=0.6):
    embedding = F.normalize(embedding, p=2, dim=1)
    best_match, best_score = "Unknown", threshold
    for name, ref in face_db.items():
        ref = F.normalize(ref.unsqueeze(0), p=2, dim=1).to(embedding.device)
        score = F.cosine_similarity(embedding, ref).item()
        if score > best_score:
            best_match, best_score = name, score
    return best_match, best_score

def align_face(img, landmark, output_size=(112,112)):
    ref_landmarks = np.array([
        [38.2946,51.6963],
        [73.5318,51.5014],
        [56.0252,71.7366],
        [41.5493,92.3655],
        [70.7299,92.2041]
    ], dtype=np.float32)
    ref_landmarks = ref_landmarks * (output_size[0]/112.0)
    src = np.array(landmark, dtype=np.float32).reshape(5,2)
    M, _ = cv2.estimateAffinePartial2D(src, ref_landmarks, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, output_size, borderValue=0.0)

def draw_face(img, boxes_c, landmarks):
    h, w, _ = img.shape
    unknown_detected = False
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i,:4]
        score = boxes_c[i,4]
        corpbbox = [max(0,int(bbox[0])), max(0,int(bbox[1])),
                    min(w,int(bbox[2])), min(h,int(bbox[3]))]

        if corpbbox[2]<=corpbbox[0] or corpbbox[3]<=corpbbox[1]:
            continue

        crop_w = corpbbox[2]-corpbbox[0]
        crop_h = corpbbox[3]-corpbbox[1]

        # 畫框
        cv2.rectangle(img,(corpbbox[0],corpbbox[1]),
                      (corpbbox[2],corpbbox[3]),(0,255,0),2)

        crop_img = align_face(img, landmarks[i], output_size=(112,112))

        if crop_h <= 50 or crop_w <=40:
            continue
        elif crop_h<80 or crop_w<60:
            threshold=0.18
            crop_img = np.clip(crop_img,0,255).astype("uint8")
            upscaled = srx4.upsample(crop_img)
            rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        else:
            threshold=0.3
            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        # 使用 PILImage.fromarray
        reimg = PILImage.fromarray(rgb)
        x = transform(reimg).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x)
        name, score = match_face(emb, face_db, threshold=threshold)

        cv2.putText(img,f"{name},{score:.2f}",
                    (corpbbox[0],corpbbox[1]-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.putText(img,f"w={crop_w:.2f},h={crop_h:.2f}",
                    (corpbbox[0],corpbbox[3]+15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        if name=="Unknown":
            unknown_detected=True
    return unknown_detected

# ----------------- ROS2 Node -----------------
class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector_node')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/oak/rgb/image_raw/compressed',
            self.listener_callback,
            10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/tb4/amcl_pose',
            self.pose_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("✅ FaceDetectorNode 已啟動")

        # 錄影變數
        self.out = None
        self.recording = False
        self.no_face_counter = 0
        self.max_no_face_frames = 15
        self.current_pose = (0.0,0.0)
        self.unknown_counter = 0
        self.min_unknown_frames = 20
        self.video_dir = os.path.join(BASE_DIR,"video")
        os.makedirs(self.video_dir,exist_ok=True)

    def pose_callback(self,msg):
        self.current_pose = (msg.pose.pose.position.x,msg.pose.pose.position.y)

    def listener_callback(self,msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        boxes_c, landmarks = infer_image(frame)
        unknown_detected = False
        if boxes_c is not None:
            unknown_detected = draw_face(frame, boxes_c, landmarks)

        # Unknown 計數
        if unknown_detected:
            self.unknown_counter+=1
        else:
            self.unknown_counter=0

        # 開始錄影
        if self.unknown_counter>=self.min_unknown_frames and not self.recording:
            self.start_recording(frame)

        if self.recording:
            self.out.write(frame)
            if boxes_c is None:
                self.no_face_counter+=1
                if self.no_face_counter>self.max_no_face_frames:
                    self.stop_recording()
            else:
                self.no_face_counter=0

        cv2.imshow("Face Detection",frame)
        cv2.waitKey(1)

    def start_recording(self,frame):
        x,y = self.current_pose
        filename = os.path.join(self.video_dir,f"face_{x:.2f}_{y:.2f}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(filename,fourcc,20.0,(frame.shape[1],frame.shape[0]))
        self.recording=True
        self.get_logger().info(f"🎥 開始錄影 {filename}")
        cv2.imwrite(filename.replace(".avi",".jpg"),frame)
        self.unknown_counter=0

    def stop_recording(self):
        self.get_logger().info("⏹ 停止錄影")
        if self.out:
            time.sleep(0.5)
            self.out.release()
            self.out=None
        self.recording=False
        self.no_face_counter=0

# ----------------- main -----------------
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

