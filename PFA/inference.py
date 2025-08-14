import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from .utils import util
from .utils.tools import MyArgumentParser
from .model.basic_model import MixTrModel

if __name__ == "__main__":

    ArgParser = MyArgumentParser()
    parser = ArgParser.get_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(os.path.join("./PFA", args.config if args.config else "voc.yaml"))
    args = OmegaConf.merge(OmegaConf.create(args.__dict__), cfg)  #
    util.setup_seed(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_prototype = torch.zeros(
         args.dataset.num_classes, 
         args.prototype.global_num, 
         args.prototype.dim)

    test_loader = util.get_loader(is_train=False, args=cfg)

    model = MixTrModel(args=args)
    model = model.to(device)

    checkpoint = torch.load('./PFA/weights/checkpoints/test1/best_ckpt.pt', weights_only=False) 
    model.load_state_dict(checkpoint['model_state_dict'])  
    model.eval()

    output_dir = os.path.join('./PFA',args.work_dir.ckpt_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
            tbar = tqdm(test_loader)
            for batch_id, batch in enumerate(tbar):
                img, label, class_label, img_path = batch
                input_size = img.size()[2:4]

                img_ = img.to(device)
                label_ = label.to(device)
                class_label_ = class_label.to(device)

                class_labels = torch.cat((torch.ones((class_label_.size(0), 1)).to(device), class_label_), dim=1)

                feat, pred = model(img_)

                pred_ = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)

                prob_map = torch.softmax(pred_, dim=1)          # shape: [B, 2, H, W]
                binary_mask = torch.argmax(prob_map, dim=1)     # shape: [B, H, W], values 0 or 1

                for i in range(binary_mask.size(0)):
                    mask_np = binary_mask[i].cpu().numpy().astype(np.uint8) * 255
                    img_name = os.path.basename(img_path[i]).replace('.jpg', '_mask.png')
                    save_path = os.path.join(output_dir, img_name)
                    Image.fromarray(mask_np).save(save_path)

    print(f"Inference completed. Masks saved to {output_dir}")