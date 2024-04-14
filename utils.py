import imutils
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


def videos_to_frames(root_path):
    """
    This function decomposes videos into frames
    Frames are recorded into directory which has same name as video (videos extentions)
    :param root_path: Path which encloses the video files
    :return: None
    """
    dir_list = sorted([i for i in os.listdir(root_path) if i.endswith(".mp4")])
    for video_path in dir_list:
        video = cv2.VideoCapture(os.path.join(root_path, video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        count = 0
        # Start converting the video
        try:
            os.mkdir(os.path.join(root_path, video_path.split('.')[0]))
        except FileExistsError:
            pass
        while video.isOpened():
            # Extract the frame
            ret, frame = video.read()
            if not ret:
                continue
            img_file_path = os.path.join(root_path, video_path.split('.')[0], "%#05d.jpg" % (count + 1))
            cv2.imwrite(img_file_path, frame)
            count = count + 1
            # If there are no more frames left
            if count > (video_length - 1):
                video.release()
                break


def extract_faces(root_path):
    """
    This function takes decomposed frames and extracts the faces out of them
    :param root_path: Path which encloses the video files
    :return: None
    """
    video_names = sorted([name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name)) and not name.startswith(".")])
    mtcnn = MTCNN(keep_all=True)
    for video_name in video_names:
        image_names = os.listdir(os.path.join(root_path, video_name))
        print(video_name, " --> ", len(image_names))
        image_names.sort()
        image_full_paths = sorted([os.path.join(root_path, video_name, x) for x in image_names if x.endswith(".jpg")])
        try:
            os.mkdir(os.path.join(root_path, video_name, "cropped"))
        except FileExistsError:
            continue
        for img_path in image_full_paths:
            frame = cv2.imread(img_path)
            img = Image.open(img_path)
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                boxes = boxes.astype(int)
                x, y, w, h = boxes[0]
                frame = frame[y:h, x:w]

                path = os.path.split(img_path)
                path = os.path.join(*path[:-1], "cropped", path[-1])
                try:
                    cv2.imwrite(path, frame)
                except:
                    print("problem saving-> ", img_path)
            else:
                print("none box--> ", img_path)


def interpolate_missing_frames(root_path):
    """
    This function finds missing face images by looking at names and interpolates with the closest existing face image
    :param root_path: Path which encloses the video files
    :return: None
    """
    video_names = sorted([name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name)) and not name.startswith(".")])
    cropped_image_names = ["%#05d.jpg" % (count + 1) for count in range(0, 299)]
    for video_name in video_names:
        image_names = sorted(e for e in os.listdir(os.path.join(root_path, video_name, "cropped")) if e.endswith(".jpg"))
        image_full_paths = sorted([os.path.join(root_path, video_name, x) for x in image_names if x.endswith(".jpg")])
        count = 0
        for expected_image in cropped_image_names:
            # if there is missing cropped image, we find the closest one and interpolate with it
            if not expected_image in image_names:
                expected_image_int = int(expected_image.split(".")[0])
                image_name_int = int(min(image_names, key=lambda x:abs(int(x.split(".")[0])-expected_image_int)).split(".")[0])
                print(image_name_int)
                path_to_open = os.path.join(root_path, video_name, "cropped", "%#05d.jpg" % image_name_int)
                path_to_save = os.path.join(root_path, video_name, "cropped", expected_image)
                frame = cv2.imread(path_to_open)
                cv2.imwrite(path_to_save, frame)
                count += 1
        print("interpolation count: ", video_name, " --> ", count)


def resize_excess(root_path, expected_height=350, expected_width=250):
    """
    This function downsizes the face images without distorting aspect ratio
    :param root_path: Path which encloses the video files
    :param expected_height: max height of image
    :param expected_width: max width of image
    :return: None
    """
    print("resize excess")
    expected_aspect_ratio = expected_height / expected_width
    video_names = sorted([name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name)) and not name.startswith(".")])
    for video_name in video_names:
        image_names = [e for e in os.listdir(os.path.join(root_path, video_name, "cropped")) if e.endswith(".jpg")]
        print("crop correction: ", video_name, " --> ", len(image_names))
        image_names.sort()
        for image_name in image_names:
            with Image.open(os.path.join(root_path, video_name, "cropped", image_name)) as image:
                width, height = image.size
                # if one dimension is larger than usual
                if height > expected_height or width > 250:
                    img = cv2.imread(os.path.join(root_path, video_name, "cropped", image_name))
                    print(img.shape)
                    if height > expected_height and width <= expected_width:
                        img = imutils.resize(img, height=expected_height)
                    elif height <= expected_height and width > expected_width:
                        img = imutils.resize(img, width=expected_width)
                    else:
                        aspect_ratio = height / width
                        if aspect_ratio > expected_aspect_ratio:
                            img = imutils.resize(img, height=expected_height)
                        else:
                            img = imutils.resize(img, width=expected_width)
                    print(img.shape)
                    print(os.path.join(root_path, video_name, "cropped", image_name))
                    cv2.imwrite(os.path.join(root_path, video_name, "cropped", image_name), img)


def pad_resize(root_path, pad_height=350, pad_width=250, resize_to=(70,50)):
    """
    This function upsize the image as fits into (pad_height x pad_width) window without distorting the aspect ratio
    Then it pads the image and makes the dimensions (pad_height x pad_width)
    Then it resizes to 'resize_to' and overwrites the existing image
    :param root_path: Path which encloses the video files
    :param pad_height: Expected height of intermediate image
    :param pad_width: Expected width of intermediate image
    :param resize_to: Expected final image dims
    :return: None
    """
    video_paths = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and not f.startswith(".")])
    trans = torchvision.transforms.Resize(resize_to)
    t2 = torchvision.transforms.ToPILImage()
    for i, video_name in enumerate(video_paths):
        print(video_name, end="->")
        image_names = sorted(os.listdir(os.path.join(root_path, video_name, "cropped")))
        for j, image_name in enumerate(image_names):
            img_path = os.path.join(root_path, video_name, "cropped", image_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[0], image.shape[1]

            if image.shape[1] < pad_height and image.shape[2] < pad_width:
                if h / w > (pad_height/pad_width):
                    image = imutils.resize(image, height=pad_height)
                else:
                    image = imutils.resize(image, width=pad_width)
                h = int(image.shape[0])
                w = int(image.shape[1])

            extra_top = (pad_height - h) // 2
            extra_bottom = (pad_height - h) - extra_top
            extra_left = (pad_width - w) // 2
            extra_right = (pad_width - w) - extra_left
            padding = (extra_left, extra_top, extra_right, extra_bottom)

            img_to_save = trans(F.pad(Image.fromarray(image), padding, 255, 'constant'))
            img_to_save = cv2.cvtColor(np.array(img_to_save), cv2.COLOR_RGB2BGR)
            os.remove(img_path)
            cv2.imwrite(img_path, img_to_save)
        print(j)


def save_as_tensor(root_path, height=70, width=50):
    """
    This function traverses through the root_path forms a tensor by (dataset len, sequence len, RGB channel, height, width)
    :param root_path:
    :param height:
    :param width:
    :return:
    """
    root_path = root_path
    seq_len = 299
    video_paths = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and not f.startswith(".")])

    cropped_image_names = ["%#05d.jpg" % (count + 1) for count in range(0, 299)]
    for i, video_name in enumerate(video_paths):
        print(video_name)
        tensor = torch.empty((seq_len, 3, height, width), dtype=torch.uint8)
        for j, img in enumerate(cropped_image_names):
            tensor[j, :] = torchvision.io.read_image(os.path.join(root_path, video_name, "cropped", img))
        torch.save(tensor, os.path.join(root_path, f"{video_name}.pt"))