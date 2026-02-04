import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Important for Flask - non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sys

class BoneSegmentationModel:
    def __init__(self, model_path='model/bone_parts_segmentation.pth'):
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.colors = {}
        
        # Initialize colors for visualization
        self._init_colors()
        
        # Load model on initialization
        self.load_model()
        
    def _init_colors(self):
        """Initialize colors for bone parts"""
        cmap = plt.cm.tab20
        self.colors[0] = np.array([0, 0, 0], dtype=np.uint8)  # Background
        for i in range(1, 9):
            color = np.array(cmap((i * 3) % 20)[:3]) * 255
            self.colors[i] = color.astype(np.uint8)
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"✗ Model file not found: {self.model_path}")
                print("Please ensure the model is trained and saved to this location.")
                return False
            
            # Import SimpleUNet here to avoid circular imports
            class SimpleUNet(torch.nn.Module):
                def __init__(self, in_channels=3, num_classes=9):
                    super().__init__()
                    self.enc1 = self._conv_block(in_channels, 32)
                    self.enc2 = self._conv_block(32, 64)
                    self.enc3 = self._conv_block(64, 128)
                    self.pool = torch.nn.MaxPool2d(2)
                    self.bottleneck = self._conv_block(128, 256)
                    self.up3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                    self.dec3 = self._conv_block(256, 128)
                    self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                    self.dec2 = self._conv_block(128, 64)
                    self.up1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                    self.dec1 = self._conv_block(64, 32)
                    self.output = torch.nn.Conv2d(32, num_classes, kernel_size=1)
                
                def _conv_block(self, in_channels, out_channels):
                    return torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True)
                    )
                
                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    b = self.bottleneck(self.pool(e3))
                    d3 = self.up3(b)
                    d3 = torch.cat([d3, e3], dim=1)
                    d3 = self.dec3(d3)
                    d2 = self.up2(d3)
                    d2 = torch.cat([d2, e2], dim=1)
                    d2 = self.dec2(d2)
                    d1 = self.up1(d2)
                    d1 = torch.cat([d1, e1], dim=1)
                    d1 = self.dec1(d1)
                    return self.output(d1)
            
            self.model = SimpleUNet(in_channels=3, num_classes=9).to(self.device)
            
            # Load model weights
            print(f"Loading model from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            print(f"✓ Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _clean_xray_image(self, img_array):
        """Clean and preprocess X-ray image"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
            elif img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        
        # Normalize
        img_array = img_array.astype(np.float32)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert back to 3-channel RGB for model input
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        return img_array
    
    def predict(self, image_path, img_size=256):
        """Predict bone parts in image"""
        # Load image
        img = Image.open(image_path)
        original_size = img.size
        
        # Handle RGBA images
        if img.mode == 'RGBA':
            white_bg = Image.new('RGBA', img.size, 'WHITE')
            white_bg.paste(img, (0, 0), img)
            img = white_bg.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Clean the image
        img_cleaned = self._clean_xray_image(img_array)
        
        # Preprocess
        img_processed = cv2.resize(img_cleaned, (img_size, img_size))
        img_processed = img_processed.astype(np.float32) / 255.0
        
        # To tensor
        img_tensor = torch.from_numpy(img_processed).float().permute(2, 0, 1).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            output = self.model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Resize back to original
        pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred_mask, img_array
    
    def generate_visualizations(self, original_img, pred_mask, output_dir, filename):
        """Generate visualization images and save statistics"""
        
        # Create output filenames
        base_name = os.path.splitext(filename)[0]
        
        # 1. Save colored segmentation mask
        colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_idx, color in self.colors.items():
            colored_mask[pred_mask == class_idx] = color
        
        mask_filename = f"mask_{base_name}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        Image.fromarray(colored_mask).save(mask_path)
        
        # 2. Generate overlay visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(original_img.shape) == 3:
            axes[0].imshow(original_img)
        else:
            axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(colored_mask)
        axes[1].set_title('Bone Parts Segmentation')
        axes[1].axis('off')
        
        # Overlay
        if len(original_img.shape) == 3:
            axes[2].imshow(original_img)
        else:
            axes[2].imshow(original_img, cmap='gray')
        
        for class_idx in range(1, 9):
            part_mask = (pred_mask == class_idx)
            if part_mask.any():
                color = self.colors[class_idx] / 255.0
                overlay = np.zeros((*pred_mask.shape, 4))
                overlay[part_mask] = [*color, 0.6]
                axes[2].imshow(overlay)
        
        axes[2].set_title('Segmentation Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        overlay_filename = f"overlay_{base_name}.png"
        overlay_path = os.path.join(output_dir, overlay_filename)
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Calculate statistics
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        
        statistics = {
            'filename': filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_size': original_img.shape,
            'total_pixels': int(total_pixels),
            'bone_parts': {},
            'summary': {}
        }
        
        bone_pixels = 0
        for cls, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            if cls == 0:
                statistics['summary']['background_percentage'] = float(percentage)
            else:
                statistics['bone_parts'][f'part_{cls}'] = {
                    'pixels': int(count),
                    'percentage': float(percentage)
                }
                bone_pixels += count
        
        statistics['summary']['bone_pixels'] = int(bone_pixels)
        statistics['summary']['bone_percentage'] = float((bone_pixels / total_pixels) * 100)
        statistics['summary']['num_bone_parts'] = len(statistics['bone_parts'])
        
        # Save statistics to JSON
        stats_filename = f"stats_{base_name}.json"
        stats_path = os.path.join(output_dir, stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        return {
            'mask_path': mask_path,
            'overlay_path': overlay_path,
            'stats_path': stats_path,
            'statistics': statistics
        }

# Global model instance
bone_model = BoneSegmentationModel('model/bone_parts_segmentation.pth')