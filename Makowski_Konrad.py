from licensePlateOCR.main import loadImages, imageSearch
import argparse
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Process license plates from images.')
    parser.add_argument('input_directory', type=str, help='The directory containing input images')
    parser.add_argument('output_file', type=str, help='The file to save the output results')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_file = args.output_file

    # Load images {'img': np.array, 'num': str, 'path': file_path}
    data = loadImages(input_directory)
    output_data = {}
    for d in tqdm(data, desc="Processing images"):
        img = d['img']
        filename = d['filename']

        image_search = imageSearch(img, epsilon=0.03, emergency_number='PO2137')
        detected_num = image_search.getLicenseNumber()
        output_data[filename] = detected_num

    # Write to json
    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


if __name__ == "__main__":
    main()
