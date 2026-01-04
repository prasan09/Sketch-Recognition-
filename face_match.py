import os
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import cosine_similarity
import cv2
import mediapipe as mp

# ---------------- CONFIG ----------------
DATASET_PATH = "C:/Users/Prasan Baligar/OneDrive/Desktop/sketch detection/Bollywood_celeb_face_localized"
PKL_PATH = "db.pkl"

# ---------------- MODELS ----------------
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained="vggface2").eval()

# ---------------- HELPERS ----------------
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None
    with torch.no_grad():
        return model(face.unsqueeze(0))

def build_db():
    db = {}
    for actor in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor)
        if not os.path.isdir(actor_path):
            continue

        embeddings = []
        for img_name in os.listdir(actor_path):
            img_path = os.path.join(actor_path, img_name)
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            db[actor] = torch.mean(torch.stack(embeddings), dim=0)

    return db

def load_db():
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        db = build_db()
        with open(PKL_PATH, "wb") as f:
            pickle.dump(db, f)
        return db

# ---------------- GLOBAL DB (LAZY) ----------------
_db = None

def recognize_face(test_image_path):
    global _db
    if _db is None:
        _db = load_db()

    test_emb = get_embedding(test_image_path)
    if test_emb is None:
        return None, None

    best_match = None
    best_score = -1

    for actor, emb in _db.items():
        score = cosine_similarity(test_emb, emb).item()
        if score > best_score:
            best_score = score
            best_match = actor

    return best_match, round(best_score, 2)

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def landmark_score(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark

    eye = abs(lm[33].x - lm[263].x)
    nose = abs(lm[1].y - lm[2].y)
    jaw = abs(lm[234].x - lm[454].x)

    return eye, nose, jaw


def artist_feedback(sketch_path, actor_name):
    actor_folder = os.path.join(DATASET_PATH, actor_name)
    ref_img = os.path.join(actor_folder, os.listdir(actor_folder)[0])

    ref = landmark_score(ref_img)
    sketch = landmark_score(sketch_path)

    if ref is None or sketch is None:
        return None

    feedback = {
        "Eyes": round(sketch[0] / ref[0] * 100, 1),
        "Nose": round(sketch[1] / ref[1] * 100, 1),
        "Jaw": round(sketch[2] / ref[2] * 100, 1)
    }

    return feedback