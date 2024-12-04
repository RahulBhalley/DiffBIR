import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from facexlib.detection import init_detection_model
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor, imwrite

from .common import load_file_from_url

def get_largest_face(det_faces, h, w):
    """Find the largest face bounding box from detected faces.

    Args:
        det_faces (list): List of detected face bounding boxes, each containing
            [left, top, right, bottom] coordinates.
        h (int): Height of the original image.
        w (int): Width of the original image.

    Returns:
        tuple: A tuple containing:
            - list: The bounding box coordinates of the largest face [left, top, right, bottom]
            - int: Index of the largest face in the input list

    Note:
        Coordinates are clamped to image boundaries (0 to width/height).
    """
    def get_location(val, length):
        """Clamp coordinate value within image boundaries.

        Args:
            val (float): Coordinate value to clamp
            length (int): Maximum boundary (width or height)

        Returns:
            int: Clamped coordinate value between 0 and length
        """
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    """Find the face closest to the center of the image.

    Args:
        det_faces (list): List of detected face bounding boxes, each containing
            [left, top, right, bottom] coordinates.
        h (int, optional): Height of the image. Defaults to 0.
        w (int, optional): Width of the image. Defaults to 0.
        center (tuple, optional): Custom center point (x,y). If None, uses image center.
            Defaults to None.

    Returns:
        tuple: A tuple containing:
            - list: The bounding box coordinates of the most central face
            - int: Index of the most central face in the input list
    """
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper class for face detection, alignment and restoration.

    This class provides functionality for:
    1. Face detection using different models (RetinaFace, dlib)
    2. Face alignment using 5-point or 3-point landmarks
    3. Face cropping and restoration
    4. Optional face parsing

    Args:
        upscale_factor (int): Factor to upscale the output image
        face_size (int, optional): Size of the cropped face. Defaults to 512.
        crop_ratio (tuple, optional): Crop ratio (height, width). Defaults to (1,1).
        det_model (str, optional): Detection model name. Defaults to 'retinaface_resnet50'.
        save_ext (str, optional): Save file extension. Defaults to 'png'.
        template_3points (bool, optional): Use 3-point instead of 5-point template.
            Defaults to False.
        pad_blur (bool, optional): Apply padding blur. Defaults to False.
        use_parse (bool, optional): Use face parsing. Defaults to False.
        device (torch.device, optional): Device to run model on. Defaults to None.

    Attributes:
        face_template (ndarray): Landmark template for face alignment
        all_landmarks_5 (list): Detected 5-point landmarks for all faces
        det_faces (list): Detected face bounding boxes
        affine_matrices (list): Affine transformation matrices
        inverse_affine_matrices (list): Inverse affine transformation matrices
        cropped_faces (list): Cropped face images
        restored_faces (list): Restored face images
        pad_input_imgs (list): Padded input images
    """

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = int(upscale_factor)
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model

        if self.det_model == 'dlib':
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941],
                                        [337.91089109, 488.38613861], [437.95049505, 493.51485149],
                                        [513.58415842, 678.5049505]])
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512 
            # facexlib
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # init face detection model
        self.face_detector = init_detection_model(det_model, half=False, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(model_name='parsenet', device=self.device)

    def set_upscale_factor(self, upscale_factor):
        """Set the upscale factor for restoration.

        Args:
            upscale_factor (int): New upscale factor
        """
        self.upscale_factor = upscale_factor

    def read_image(self, img):
        """Read and preprocess input image.

        Args:
            img (str or ndarray): Image path or cv2 loaded image array

        Note:
            - Converts 16-bit images to 8-bit
            - Converts grayscale to BGR
            - Removes alpha channel if present
            - Resizes image if smaller than 512px
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # BGRA image with alpha channel
            img = img[:, :, 0:3]

        self.input_img = img

        if min(self.input_img.shape[:2])<512:
            f = 512.0/min(self.input_img.shape[:2])
            self.input_img = cv2.resize(self.input_img, (0,0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

    def init_dlib(self, detection_path, landmark5_path):
        """Initialize dlib face detector and landmark predictor.

        Args:
            detection_path (str): Path to dlib face detection model
            landmark5_path (str): Path to dlib 5-point landmark model

        Returns:
            tuple: Containing:
                - dlib.cnn_face_detection_model_v1: Face detector
                - dlib.shape_predictor: Landmark predictor

        Raises:
            ImportError: If dlib is not installed
        """
        try:
            import dlib
        except ImportError:
            print('Please install dlib by running:' 'conda install -c conda-forge dlib')
        detection_path = load_file_from_url(url=detection_path, model_dir='weights/dlib', progress=True, file_name=None)
        landmark5_path = load_file_from_url(url=landmark5_path, model_dir='weights/dlib', progress=True, file_name=None)
        face_detector = dlib.cnn_face_detection_model_v1(detection_path)
        shape_predictor_5 = dlib.shape_predictor(landmark5_path)
        return face_detector, shape_predictor_5

    def get_face_landmarks_5_dlib(self,
                                only_keep_largest=False,
                                scale=1):
        """Detect faces and 5-point landmarks using dlib.

        Args:
            only_keep_largest (bool, optional): Only keep largest detected face.
                Defaults to False.
            scale (int, optional): Image scale factor. Defaults to 1.

        Returns:
            int: Number of faces detected with valid landmarks
        """
        det_faces = self.face_detector(self.input_img, scale)

        if len(det_faces) == 0:
            print('No face detected. Try to increase upsample_num_times.')
            return 0
        else:
            if only_keep_largest:
                print('Detect several faces and only keep the largest.')
                face_areas = []
                for i in range(len(det_faces)):
                    face_area = (det_faces[i].rect.right() - det_faces[i].rect.left()) * (
                        det_faces[i].rect.bottom() - det_faces[i].rect.top())
                    face_areas.append(face_area)
                largest_idx = face_areas.index(max(face_areas))
                self.det_faces = [det_faces[largest_idx]]
            else:
                self.det_faces = det_faces

        if len(self.det_faces) == 0:
            return 0

        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)

        return len(self.all_landmarks_5)


    def get_face_landmarks_5(self,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        """Detect faces and extract 5-point landmarks.

        This method detects faces in the input image and extracts 5 facial landmarks 
        (eyes, nose, mouth corners) for each detected face. It can optionally keep only
        the largest face or most central face, and handle image resizing and blurring.

        Args:
            only_keep_largest (bool, optional): If True, only keep the face with largest
                bounding box. Defaults to False.
            only_center_face (bool, optional): If True, only keep the face closest to 
                image center. Defaults to False.
            resize (int, optional): Target size for resizing image before detection. 
                Image will be scaled so smallest dimension equals this value.
                If None, no resizing is performed. Defaults to None.
            blur_ratio (float, optional): Ratio for calculating blur kernel size when
                padding images. Kernel size = face size * blur_ratio. 
                Defaults to 0.01.
            eye_dist_threshold (float, optional): Minimum required distance between eyes.
                Faces with eye distance below this are filtered out.
                If None, no filtering is done. Defaults to None.

        Returns:
            int: Number of faces detected with valid landmarks

        Note:
            - If self.det_model is 'dlib', delegates to get_face_landmarks_5_dlib()
            - Detected faces and landmarks are stored in self.det_faces and 
              self.all_landmarks_5
            - If self.pad_blur is True, padded and blurred versions of face images
              are stored in self.pad_input_imgs
            - Landmarks are scaled back to original image coordinates if resizing
              was applied
        """
        if self.det_model == 'dlib':
            return self.get_face_landmarks_5_dlib(only_keep_largest)

        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = resize / min(h, w)
            scale = max(1, scale) # always scale up
            h, w = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            input_img = cv2.resize(self.input_img, (w, h), interpolation=interp)

        with torch.no_grad():
            bboxes = self.face_detector.detect_faces(input_img)

        if bboxes is None or bboxes.shape[0] == 0:
            return 0
        else:
            bboxes = bboxes / scale

        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
            
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]

        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp detected faces using affine transformation.

        This method aligns detected faces with a face template using landmarks, warps them 
        using affine transformation, and optionally saves the cropped faces.

        Args:
            save_cropped_path (str, optional): Path to save cropped face images. If None,
                faces are not saved. The path will be appended with face index.
                Defaults to None.
            border_mode (str, optional): Border mode for warping. One of:
                - 'constant': Fill borders with constant gray value
                - 'reflect101': Mirror padding without repeating edge pixels 
                - 'reflect': Mirror padding with edge pixel repetition
                Defaults to 'constant'.

        Note:
            - Requires face landmarks to be detected first via get_face_landmarks_5()
            - Stores results in self.affine_matrices and self.cropped_faces
            - Uses gray border value (135,133,132) for constant border mode
        """
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Calculate inverse affine transformation matrices for face restoration.

        This method computes the inverse of each affine transformation matrix used for face
        alignment, scales it by the upscale factor, and optionally saves the matrices.

        Args:
            save_inverse_affine_path (str, optional): Path to save inverse affine matrices.
                If provided, matrices are saved as PyTorch tensors with face index appended.
                Defaults to None.

        Note:
            - Requires align_warp_face() to be called first to generate affine matrices
            - Stores results in self.inverse_affine_matrices
            - Matrices are scaled by self.upscale_factor for high-res restoration
        """
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, restored_face, input_face=None):
        """Add a restored face to the collection.

        Args:
            restored_face (ndarray): The restored face image to add
            input_face (ndarray, optional): Original input face for potential color
                transfer. Currently unused. Defaults to None.

        Note:
            Stores the restored face in self.restored_faces for later processing
        """
        # if self.is_gray:
        #     restored_face = bgr2gray(restored_face) # convert img into grayscale
        #     if input_face is not None:
        #         restored_face = adain_npy(restored_face, input_face) # transfer the color
        self.restored_faces.append(restored_face)

    def paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None):
        """Paste restored faces back into the original image.

        This method transforms restored faces back to their original positions using inverse
        affine matrices, blends them seamlessly with the background, and optionally draws
        face bounding boxes.

        Args:
            save_path (str, optional): Path to save final image. If None, image is not saved.
                Defaults to None.
            upsample_img (ndarray, optional): Pre-upsampled background image. If None,
                input image is upsampled using linear interpolation. Defaults to None.
            draw_box (bool, optional): Whether to draw green bounding boxes around faces.
                Defaults to False.
            face_upsampler (object, optional): Face-specific upsampler for enhanced quality.
                Must have an enhance() method. Defaults to None.

        Returns:
            ndarray: Final image with restored faces pasted in

        Note:
            - Requires restored faces and inverse affine matrices to be prepared
            - Handles both RGB and RGBA images
            - Uses Gaussian blending for seamless face integration
            - Supports 8-bit and 16-bit images
            - If using face parsing (self.use_parse=True), generates refined masks
        """
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            # simply resize the background
            # upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        
        inv_mask_borders = []
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            if face_upsampler is not None:
                restored_face = face_upsampler.enhance(restored_face, outscale=self.upscale_factor)[0]
                inverse_affine /= self.upscale_factor
                inverse_affine[:, 2] *= self.upscale_factor
                face_size = (self.face_size[0]*self.upscale_factor, self.face_size[1]*self.upscale_factor)
            else:
                # Add an offset to inverse affine matrix, for more precise back alignment
                if self.upscale_factor > 1:
                    extra_offset = 0.5 * self.upscale_factor
                else:
                    extra_offset = 0
                inverse_affine[:, 2] += extra_offset
                face_size = self.face_size
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

            # always use square mask
            mask = np.ones(face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)  # // 3
            # add border
            if draw_box:
                h, w = face_size
                mask_border = np.ones((h, w, 3), dtype=np.float32)
                border = int(1400/np.sqrt(total_face_area))
                mask_border[border:h-border, border:w-border,:] = 0
                inv_mask_border = cv2.warpAffine(mask_border, inverse_affine, (w_up, h_up))
                inv_mask_borders.append(inv_mask_border)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            if len(upsample_img.shape) == 2:  # upsample_img is gray image
                upsample_img = upsample_img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]

            # parse mask
            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                parse_mask = np.zeros(out.shape)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(MASK_COLORMAP):
                    parse_mask[out == idx] = color
                #  blur the mask
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                # remove the black borders
                thres = 10
                parse_mask[:thres, :] = 0
                parse_mask[-thres:, :] = 0
                parse_mask[:, :thres] = 0
                parse_mask[:, -thres:] = 0
                parse_mask = parse_mask / 255.

                parse_mask = cv2.resize(parse_mask, face_size)
                parse_mask = cv2.warpAffine(parse_mask, inverse_affine, (w_up, h_up), flags=3)
                inv_soft_parse_mask = parse_mask[:, :, None]
                # pasted_face = inv_restored
                fuse_mask = (inv_soft_parse_mask<inv_soft_mask).astype('int')
                inv_soft_mask = inv_soft_parse_mask*fuse_mask + inv_soft_mask*(1-fuse_mask)

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        # draw bounding box
        if draw_box:
            # upsample_input_img = cv2.resize(input_img, (w_up, h_up))
            img_color = np.ones([*upsample_img.shape], dtype=np.float32)
            img_color[:,:,0] = 0
            img_color[:,:,1] = 255
            img_color[:,:,2] = 0
            for inv_mask_border in inv_mask_borders:
                upsample_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_img
                # upsample_input_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_input_img

        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img

    def clean_all(self):
        """Reset all internal face processing state.

        Clears all stored landmarks, faces, transformation matrices and other internal
        state to prepare for processing a new image.
        """
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []