import argparse
import cv2
import YOLO


def getColours(cls_num):

    # get colour for classes
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num) // len(base_colors) % 256
        for i in range(3)
    ]

    return tuple(color)


def drawBoundingBox(results, frame):

    # get the class names
    class_names = results.names

    for box in results.box:
        # only consider detections with conf > 40%
        if box.conf > 0.4:
            # get coordinates
            [x1, x2, y1, y2] = box.xyxy[0]
            x1, x2, y1, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class index
            cls = int(box.cls[0])

            class_name = class_names[cls]

            # get color for class
            color = getColours(cls)

            # draw rectanlge around detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # display class name and conf
            cv2.putText(
                frame,
                f"{class_names[int(box.cls[0])]} {box.conf[0]:.2f}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return frame


def main():

    # add cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to yolo model")
    parser.add_argument("--cam", help="camera index", type=int)
    args = parser.parse_args()

    # create YOLO model and export engine
    model = YOLO(args.model)

    # export inference engine
    model.export(format="engine")

    trt_model = YOLO("yolo11n.engine")

    # init cam object
    cap = cv2.VideoCapture(args.cam)

    while True:

        # fetch frame from cam
        ret, frame = cap.read()

        if not ret:
            continue

        # run inference
        results = trt_model(frame)

        # draw bounding box
        drawBoundingBox(results, frame)

        # show open cv frame
        cv2.imshow(frame, "frame")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
