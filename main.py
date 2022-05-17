import cv2
from gui_buttons import Buttons


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

class_names = []
with open("dnn_model/classes.txt" , "r") as f:
    for cla in f:
        cla = cla.strip()
        class_names.append(cla)
print(class_names)
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


buttons = Buttons()
buttons.add_button("person", 20, 20)
buttons.add_button("cell phone", 20, 100)
buttons.add_button("keyboard", 20, 180)
buttons.add_button("book", 20, 260)
buttons.add_button("scissors", 20, 340)

def click_button(event,x,y,flag,params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        buttons.button_click(x,y)


cv2.namedWindow("frame")
cv2.setMouseCallback("frame",click_button)

while True:
    ret,frame = cap.read()
    active_buttons = buttons.active_buttons_list()

    (class_ids,scores,bboxes)= model.detect(frame,confThreshold=0.3,nmsThreshold=.4)
    for class_id,score,bboxe in zip(class_ids,scores,bboxes):
        x,y,w,h = bboxe
        class_name = class_names[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y -10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    buttons.display_buttons(frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()