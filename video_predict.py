import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet


# ===================== CONFIG =====================
MODEL_PATH       = "checkpoints/checkpoint_epoch100.pth"
VIDEO_PATH       = "../videoTracking/video_with_marker2.mp4"
OUTPUT_VIDEO_PATH = "../videoTracking/u_net_output_mask_2.mp4"

SCALE_FACTOR     = 1.0       # must match what you used in training (e.g. 1.0 or 0.5)
MASK_THRESHOLD   = 0.5       # for binary (sigmoid) case if n_classes == 1
N_CHANNELS       = 3         # 3 if RGB, 1 if grayscale model
N_CLASSES        = 2         # 2 for binary segmentation (background + dot)
BILINEAR         = False      # set according to how you trained UNet

SHOW_WINDOW      = True      # set False if you don't want a live window
# ==================================================


def predict_frame(net, frame_bgr, device, scale_factor=1.0, out_threshold=0.5):
    """
    frame_bgr: numpy array (H,W,3) in BGR (from OpenCV)
    returns: mask (H,W) as uint8 (0 or 255)
    """
    net.eval()

    # Convert BGR -> RGB -> PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Use same preprocessing as training
    img = torch.from_numpy(
        BasicDataset.preprocess(None, pil_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,C,H,W]

    with torch.no_grad():
        output = net(img).cpu()   # [1, n_classes, H', W']

        # Resize network output back to original frame size (H,W)
        H, W = frame_bgr.shape[:2]
        output = F.interpolate(
            output, size=(H, W), mode='bilinear', align_corners=False
        )

        if net.n_classes > 1:
            # multi-class: take argmax over channel dimension
            mask = output.argmax(dim=1)  # [1, H, W], values in {0..n_classes-1}
            mask = (mask == 1).to(torch.uint8)  # for binary: class 1 as foreground
        else:
            # binary with single channel: sigmoid + threshold
            probs = torch.sigmoid(output)[0, 0]    # [H, W]
            mask = (probs > out_threshold).to(torch.uint8)

    # Convert to 0/255 uint8 image
    mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
    return mask_np


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    net = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=BILINEAR)
    net.to(device=device)

    print(f"Loading model from {MODEL_PATH} ...")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # Remove mask_values key if present (from training script)
    state_dict.pop('mask_values', None)
    net.load_state_dict(state_dict)
    print("Model loaded.")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video {VIDEO_PATH}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps_in <= 0:
        fps_in = 30.0

    print(f"Video: {VIDEO_PATH} ({width}x{height} @ {fps_in:.2f} FPS)")

    # Output video writer (for mask video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_in, (width, height))
    if not out.isOpened():
        print(f"Error: cannot open VideoWriter {OUTPUT_VIDEO_PATH}")
        cap.release()
        return

    frame_idx = 0
    prev_time = time.time()
    fps_smoothed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_idx += 1
        t0 = time.time()

        # Predict mask for this frame
        mask = predict_frame(
            net, frame, device,
            scale_factor=SCALE_FACTOR,
            out_threshold=MASK_THRESHOLD
        )

        # Convert mask to 3-channel BGR for video writing
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Write mask frame
        out.write(mask_bgr)

        # Compute FPS (simple exponential smoothing)
        t1 = time.time()
        inst_fps = 1.0 / max(t1 - t0, 1e-6)
        if fps_smoothed == 0.0:
            fps_smoothed = inst_fps
        else:
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst_fps

        # Prepare display: side-by-side input | mask
        if SHOW_WINDOW:
            combined = np.hstack([frame, mask_bgr])
            # Put FPS text on top-left
            cv2.putText(
                combined,
                f"FPS: {fps_smoothed:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Input | Predicted Mask", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Interrupted by user.")
                break

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames, FPS ~ {fps_smoothed:.1f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output mask video saved to {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
