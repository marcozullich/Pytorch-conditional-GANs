import os
import torch
import argparse
from conditional_dcgan import ModelG
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument("--weights", default="models_def/model_g_epoch_25.pth", type=str, help="Path to generator's parameters")
    parser.add_argument("--nz", default=100, type=int, help="Number of dimensions for input noise")
    parser.add_argument("--num_images", default=5, type=int, help="Number of images to generate")
    parser.add_argument("--categories", default=None, type=int, nargs="+", help="Categories to generate from. If not specified, all categories are randomly chosen. If one category is specified, only that category is chosen. If the number of categories is different from num_images, error is thrown.")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument("--device", default=None, type=str, help="Device to use. Leave None to use cuda if available")
    parser.add_argument("--save_dir", default="generated", type=str, help="Path to save the generated images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if (args.categories is not None):
        if len(args.categories) != args.num_images and len(args.categories) > 1:
            raise ValueError(f"Number of categories ({args.categories}) must be equal to number of images ({args.num_images}) or 1")
        if any(c < 0 or c > 9 for c in args.categories):
            raise ValueError(f"Categories must be between 0 and 9 (found {args.categories})")
    
    if (seed:=args.seed) is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    device = args.device if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    net = ModelG(args.nz)
    net.load_state_dict(torch.load(args.weights)["state_dict"])
    net.to(device)

    if args.categories is None:
        categories = torch.randint(0, 10, (args.num_images,), dtype=torch.long).to(device)
    elif len(args.categories) == 1:
        categories = torch.full((args.num_images,), args.categories[0], dtype=torch.long).to(device)
    else:
        categories = torch.Tensor(args.categories).long().to(device)
    categories_onehot = torch.nn.functional.one_hot(categories, num_classes=10).float().to(device)

    noise = torch.randn(args.num_images, args.nz, device=device)
    output = net(noise, categories_onehot)
    output = [(t*255).to(torch.uint8).squeeze().numpy() for t in output.detach().cpu().tensor_split(args.num_images, dim=0)]

    for i, (img, cl) in enumerate(zip(output, categories)):
        Image.fromarray(img).save(os.path.join(args.save_dir, f"generated_{i}_class_{cl.item()}.png"))
    print(f"Images saved to {args.save_dir}")
