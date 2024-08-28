import cv2
import mediapipe as mp

class pose:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    after_img = ""
    StartFlg = False
    rsp = ""
    FinishFlg = False

    def PoseLoop(self):
        cap = cv2.VideoCapture(4)
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #before_image = image.copy()

                results = pose.process(image)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    keypoint_pos = []
                    for i in landmarks:
                        # Acquire x, y but don't forget to convert to integer.
                        x = int(i.x * image.shape[1])
                        y = int(i.y * image.shape[0])
                        # Annotate landmarks or do whatever you want.
                        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                        keypoint_pos.append((x, y))
                    #print(keypoint_pos)
                    print(landmarks)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if __name__ == '__main__':
                    cv2.imshow('MediaPipe Hands', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

        cap.release()

if __name__ == '__main__':
    m_pose = pose()
    m_pose.PoseLoop()
