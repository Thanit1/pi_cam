#import libraryที่จำเป็น
import numpy as np 
import cv2
import flask
from flask import Flask, render_template, Response

#รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
#สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")
#เลือกวิดีโอ/เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตรวจสอบว่าสามารถเปิดกล้องได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการเชื่อมต่อกล้องหรือลองใช้ index อื่น")
    exit()

# ตัวแปรสำหรับเก็บตำแหน่งของคนที่ตรวจจับได้
detected_persons = []

app = Flask(__name__)

def generate_frames():
    while True:
        #เริ่มอ่านในแต่ละเฟรม
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านเฟรมได้ กำลังออกจากโปรแกรม...")
            break

        (h,w) = frame.shape[:2]
        #ทำpreprocessing
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
        try:
            detections = net.forward()
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
            continue

        # เคลียร์ตำแหน่งคนที่ตรวจจับได้ในเฟรมก่อนหน้า
        detected_persons.clear()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]
            #กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 เพิ่มลดได้ตามต้องการ
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                # ตรวจสอบเฉพาะคลาส "PERSON"
                if CLASSES[class_index] == "PERSON":
                    box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # เพิ่มตำแหน่งของคนที่ตรวจจับได้ลงในลิสต์
                    detected_persons.append((startX, startY, endX, endY))

                    #ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
                    label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                    cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                    y = startY - 15 if startY-15>15 else startY+15
                    cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

        # แสดงจำนวนคนที่ตรวจจับได้บนเฟรม
        cv2.putText(frame, f"Count: {len(detected_persons)}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)  # เปลี่ยนสีตัวหนังสือเป็นสีแดง

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

#หลังเลิกใช้แล้วเคลียร์memoryและปิดกล้อง
cap.release()
cv2.destroyAllWindows()