import torch
from tqdm import tqdm
import os
from conch.open_clip_custom import create_model_from_pretrained
from PIL import Image
import argparse

def main(image_path, output_path):
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    # List of all image files in the input directory
    image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]

    # Process each image
    for image_file in tqdm(image_files):
        # Construct the full path to the image
        image_full_path = os.path.join(image_path, image_file)
        # Open the image
        image = Image.open(image_full_path)
        # Preprocess the image
        image = preprocess(image).unsqueeze(0)

        # Generate the image embedding
        with torch.no_grad():  # No gradient calculation for inference
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)

        # Save the embedding to a file
        embedding_path = os.path.join(output_path, f"{os.path.splitext(image_file)[0]}.pt")
        torch.save(image_embs, embedding_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for images using a pretrained model.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the folder to save embeddings.')

    args = parser.parse_args()
    main(args.image_path, args.output_path)
