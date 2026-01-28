
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from lightglue.aliked import ALIKED

def setup_aliked_masking():
    if getattr(ALIKED, "_is_patched_for_masking", False):
        return
    original_forward = ALIKED.forward
    def masked_forward(self, data):
        image = data["image"]
        if image.shape[1] == 1:
            from kornia.color import grayscale_to_rgb
            image = grayscale_to_rgb(image)
        feature_map, score_map = self.extract_dense_map(image)
        if "mask" in data and data["mask"] is not None:
            mask = data["mask"]
            if mask.shape[-2:] != score_map.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask, size=score_map.shape[-2:], mode="nearest"
                )
            mask = mask.to(score_map.device)
            score_map = score_map.masked_fill(mask > 0.5, float("-inf"))
        keypoints, kptscores, scoredispersitys = self.dkd(
            score_map, image_size=data.get("image_size")
        )
        descriptors, offsets = self.desc_head(feature_map, keypoints)
        _, _, h, w = image.shape
        wh = torch.tensor([w - 1, h - 1], device=image.device)
        return {
            "keypoints": wh * (torch.stack(keypoints) + 1) / 2.0,
            "descriptors": torch.stack(descriptors),
            "keypoint_scores": torch.stack(kptscores),
        }
    ALIKED.forward = masked_forward
    ALIKED._is_patched_for_masking = True

def run():
    img_path = Path("datasets/voiture_dure/images/frame_00366_video_1.png")
    mask_path = Path("datasets/voiture_dure/masks/window/frame_00366_video_1.png")
    
    setup_aliked_masking()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ALIKED(model_name="aliked-n16", max_num_keypoints=2048).eval().to(device)
    
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0
    inputs = inputs.unsqueeze(0).to(device)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Mask not found")
        return
    
    # Resize mask if needed
    if mask.shape != img.shape[:2]:
         mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_t = torch.from_numpy(mask).float()/255.0
    mask_t = (mask_t > 0.5).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model({"image": inputs, "mask": mask_t})
    
    kpts = pred["keypoints"][0].cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.imshow(mask, cmap='jet', alpha=0.3) # Overlay mask (Redish/Blueish)
    plt.scatter(kpts[:, 0], kpts[:, 1], c='lime', s=3, label='Keypoints')
    plt.title(f"Frame 366: {len(kpts)} Keypoints (Lime) | Mask (Overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vis_366.jpg", dpi=150)
    print("Saved vis_366.jpg")

if __name__ == "__main__":
    run()
