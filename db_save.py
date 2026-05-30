from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import gspread
import os 
import json
from google.oauth2.service_account import Credentials

app = Flask(__name__)

# =========================
# SIFT
# =========================
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# =========================
# URL → 画像
# =========================
def url_to_image(url):

    try:
        r = requests.get(url, timeout=10)

        img_array = np.frombuffer(
            r.content,
            np.uint8
        )

        return cv2.imdecode(
            img_array,
            cv2.IMREAD_COLOR
        )

    except:
        return None

# =========================
# 特徴量
# =========================
def get_des(img):

    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    kp, des = sift.detectAndCompute(
        gray,
        None
    )

    return des

# =========================
# 比較
# =========================
def compare(des1, des2):

    if des1 is None or des2 is None:
        return 0

    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)

# =========================
# スプレッドシート
# =========================
scope = [
    "https://www.googleapis.com/auth/spreadsheets"
]

service_account_json = os.environ.get( 
    "SERVICE_ACCOUNT_JSON" 
) 

service_account_info = json.loads( 
    service_account_json 
) 

creds = Credentials.from_service_account_info( 
    service_account_info, 
    scopes=scope 
)

client = gspread.authorize(creds)

sheet = client.open_by_key(
    "10IxAgfcm8FrrUpAJH3S1rRQxrluQk58Egm86wM-j6D4"
).sheet1

data = sheet.get_all_values()

# =========================
# DB
# =========================
db_des = {}

for row in data[1:]:

    if len(row) < 2:
        continue

    url = row[0]
    name = row[1]

    img = url_to_image(url)

    db_des[name] = get_des(img)

    print("登録:", name)

# =========================
# API
# =========================
@app.route("/check", methods=["POST"])
def check():

    image_url = request.json["image_url"]

    input_img = url_to_image(image_url)

    input_des = get_des(input_img)

    best_name = None
    best_score = 0

    for name, des in db_des.items():

        score = compare(input_des, des)

        if score > best_score:
            best_score = score
            best_name = name

    if input_des is not None:
        input_count = len(input_des)
    else:
        input_count = 1

    confidence = (
        best_score / input_count
    ) * 100

    if confidence < 80:

        return jsonify({
            "result": "該当なし",
            "confidence": round(confidence, 2)
        })

    return jsonify({
        "result": best_name,
        "confidence": round(confidence, 2)
    })

# =========================
# 起動
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
