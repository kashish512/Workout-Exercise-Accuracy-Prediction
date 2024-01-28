from fastapi import FastAPI, File, UploadFile
import cv2, os, io, math
from google.cloud import storage
import mediapipe as mp
from google.cloud import storage
import numpy as np
from ultralytics import YOLO

storage_bucket_name = 'exercise-api-storage'
input_bucket_name = 'exercise-api-input-video'
output_bucket_name = 'exercise-api-output-video'

path=r"first-parser-410916-64da59f4283f.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path

client = storage.Client()

app = FastAPI()

model = YOLO('last.pt')

@app.get("/")
async def get_uploadfile():
    return {"message": "Upload your exrcise on predict"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

class poseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.model_complexity=model_complexity #Added
        self.smooth_landmarks=smooth_landmarks #Added
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, 
                                     self.model_complexity, self.smooth_landmarks, 
                                     self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        img=np.array(img)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        img=np.array(img)
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, opimg, p1, p2, p3, draw=True, angledraw=False):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(opimg, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(opimg, (x3, y3), (x2, y2), (255, 0, 0), 3)
            cv2.circle(opimg, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(opimg, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(opimg, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(opimg, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(opimg, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(opimg, (x3, y3), 15, (0, 0, 255), 2)
            if angledraw:
                cv2.putText(opimg, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        
        return angle
    
    def findPoseonly(self, img, opimg, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(opimg, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return opimg
    
    def findPositiononly(self, img, opimg, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(opimg, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

def videooutput(answer, name):
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    video_output = cv2.VideoWriter(f"output_video_{name}", fourcc, 10.0, (1200, 720))

    from google.cloud import storage

    
    cap = cv2.VideoCapture(f"downloaded_{name}")

    detector = poseDetector()
    direc = 0
    while True:
        success, img = cap.read()
        if(success==False):
            break
        img = cv2.resize(img, (1200, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        h, w, c = img.shape
        opimg=np.zeros([h, w, c], dtype=np.uint8)
        opimg.fill(255)

        res = detector.pose.process(img)
        detector.mpDraw.draw_landmarks(opimg, res.pose_landmarks, detector.mpPose.POSE_CONNECTIONS, detector.mpDraw.DrawingSpec((0, 0, 255), 2, 2),
                               detector.mpDraw.DrawingSpec((255, 0, 0), 2, 2))
        
        
        if answer=='overhead squats':
            
            if len(lmList) != 0:
                # Right knee
                angle1 = detector.findAngle(img, opimg, 24, 26, 28)
                # Left knee
                angle2 = detector.findAngle(img, opimg, 27, 25, 23)

                #left hip
                angle3 = detector.findAngle(img, opimg, 23, 24, 26)
                #right hip
                angle4 = detector.findAngle(img, opimg, 25, 23, 24)

                #left shoulder
                angle5 = detector.findAngle(img, opimg, 14, 12, 24)
                #right shoulder
                angle6 = detector.findAngle(img, opimg, 13, 11, 23)

                ans=0.1*(angle3+angle4)/2+0.1*(angle5+angle6)/2-0.8*(angle1+angle2)/2
        #         print(ans)

                per = np.interp(ans, (-117, -23), (0, 100))
                bar = np.interp(ans, (-117, -23), (650, 100))
                # print(angle, per)

                # Check for the depth
                color = (0, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if direc == 0:
                        direc = 1
                if per == 0:
                    color = (0, 255, 0)
                    if direc == 1:
                        direc = 0


                cv2.rectangle(opimg, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(opimg, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(opimg, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2.5,color, 2)
                cv2.putText(opimg, f'{answer}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5,color, 2)
                
                
              # cv2.imshow("Image", opimg)
            frame = cv2.convertScaleAbs(opimg)
            video_output.write(frame)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                  
        elif answer=='glute bridges':
            
            
            if len(lmList) != 0:
                    
                angle1 = detector.findAngle(img,opimg, 11, 23, 25)
        # Left knee
                angle2 = detector.findAngle(img,opimg, 12, 24, 26)

                ans=(angle1+angle2)/2
        #         print(ans)

                per = np.interp(ans, (135, 180), (0, 100))
                bar = np.interp(ans, (135, 180), (650, 100))
                # print(angle, per)

                # Check for the depth
                color = (0, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                if per == 0:
                    color = (0, 255, 0)

                # Draw Bar
                cv2.rectangle(opimg, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(opimg, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(opimg, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
                cv2.putText(opimg, f'{answer}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
            
            frame = cv2.convertScaleAbs(opimg)
            video_output.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  

                        
        elif answer=='backlunges':
            
            if len(lmList) != 0:
                # Right knee
                angle1 = detector.findAngle(img, opimg, 28, 26, 24)
                # Left knee
                angle2 = detector.findAngle(img, opimg, 27, 25, 23)

                ans=-0.8*(angle1+angle2)/2
    #         print(ans)

                per = np.interp(ans, (-133, -72), (0, 100))
                bar = np.interp(ans, (-133, -72), (650, 100))
                # print(angle, per)

                # Check for the depth
                color = (0, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                if per == 0:
                    color = (0, 255, 0)
                # Draw Bar
                cv2.rectangle(opimg, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(opimg, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(opimg, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
                cv2.putText(opimg, f'{answer}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
                            
            frame = cv2.convertScaleAbs(opimg)
            video_output.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elif answer=='box-jumps':
            
            if len(lmList) != 0:
                # Right knee
                angle1 = detector.findAngle(img, opimg, 28, 26, 24)
                # Left knee
                angle2 = detector.findAngle(img, opimg, 27, 25, 23)

                ans=-0.8*(angle1+angle2)/2
    #         print(ans)

                per = np.interp(ans, (200, 264), (0, 100))
                bar = np.interp(ans, (200, 264), (650, 100))
                # print(angle, per)

                # Check for the depth
                color = (0, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                if per == 0:
                    color = (0, 255, 0)
                # Draw Bar
                cv2.rectangle(opimg, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(opimg, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(opimg, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
                cv2.putText(opimg, f'{answer}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5,
                            color, 2)
                            
            frame = cv2.convertScaleAbs(opimg)
            video_output.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
    cap.release()
    video_output.release()
            # Destroy all the windows
    cv2.destroyAllWindows()

@app.post("/output_video")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        file.file.seek(0)
        input_blob_name = f"uploaded_{file.filename}"
        output_blob_name = f"output_video_{file.filename}"

        input_bucket = client.bucket(input_bucket_name)

        input_blob = input_bucket.blob(input_blob_name)
        input_blob.upload_from_file(file.file)

        bucket = client.bucket(input_bucket_name)
        blob = bucket.blob(input_blob_name)
        blob.download_to_filename(f"downloaded_{file.filename}")
        results = model(f"downloaded_{file.filename}")

        # rseults=model(blob.public_url)
        # Upload the input video to the input bucket
        # Process the results

        names_dict = results[0].names
        probs = []
        y = np.array([0, 0, 0, 0])
        
        for i in range(0, len(results)):
            x = results[i].probs.data.tolist()
            probs.append(x)
            y = y + x
        # ...
        predicted_class = names_dict[np.argmax(y)]
        videooutput(predicted_class,file.filename)

        # Upload the output video to the output bucket
        output_bucket = client.bucket(output_bucket_name)
        output_blob = output_bucket.blob(output_blob_name)
        with open(f"output_video_{file.filename}", 'rb') as video_file:
            output_blob.upload_from_file(video_file)
        muscles=""
        if(predicted_class=='overhead squats'):
            muscles="Quadriceps, Hamstrings, Deltoids, Latissimus dorsi"
        if(predicted_class=='glute bridges'):
            muscles="Gluteus Maximus, Hamstrings, Quadriceps, Erector Spinae" 
        if(predicted_class=='backlunges'):
            muscles="Quadriceps, Hamstrings, Gluteus Maximus, Erector Spinae"
        if(predicted_class=='box-jumps'):
            muscles="Quadriceps, Gastrocnemius and Soleus, Hamstrings, Gluteus maximum, Abductors, Adductors" 
        return {"exercise": predicted_class, "URL": output_blob.public_url, "Muscles Involved": muscles}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug= True)

# get_ipython().system('jupyter nbconvert --to script classify.ipynb')