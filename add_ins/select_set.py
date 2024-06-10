import json
import cv2


def select_set():
    with open("../../source/IMG_ROOTS/1280x960_CVATROOT/train_set/train_set.json", "r") as inf:
        files = [json.loads(i) for i in inf]
    with open("new_train_file1.json", "a") as tf:
        with open("new_val_set1.json", "a") as ouf:
            count = 0
            for i, elem in enumerate(files[::-1]):
                tmp_img = cv2.imread("../../source/IMG_ROOTS/1280x960_CVATROOT/train_set/" + elem["raw_file"])
                cv2.imshow(f"{i:04d}", tmp_img)
                k = cv2.waitKey(0)
                if k == ord('j') and count < 200:
                    count += 1
                    print(f"Selected {len(files) + 1 - i:04d}; {count=}")
                    ouf.write(json.dumps(elem) + '\n')
                    ouf.flush()
                else:
                    tf.write(json.dumps(elem) + '\n')
                    tf.flush()
                cv2.destroyAllWindows()
                if k == ord('q'):
                    count = 200
                    tf.write(json.dumps(elem) + '\n')
                    tf.flush()


if __name__ == '__main__':
    select_set()
