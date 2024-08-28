import cv2
import mediapipe as mp

class Pose:
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
                before_image = image.copy()

                results = pose.process(image)
                
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS)
                    print(results.pose_landmarks.landmark[0].x)

                self.after_img = cv2.flip(image - before_image, 1)

                if __name__ == '__main__':
                    cv2.imshow('MediaPipe Hands', self.after_img)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

        cap.release()

if __name__ == '__main__':
    m_pose = Pose()
    m_pose.PoseLoop()