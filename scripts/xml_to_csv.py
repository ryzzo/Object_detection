import os
import csv
import argparse
import xml.etree.ElementTree as ET

def parse_annotation(xml_file):
    """
    Parse the XML file and extract bounding box annotations.
    Returns a list of dicts with keys: filename, xmin, ymin, xmax, ymax, label
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.findtext('filename')
    if filename:
        filename = filename.strip()
    else:
        # fallback to using the XML filename
        filename = os.path.basename(xml_file).replace('.xml', '.jpg')

    boxes = []
    for obj in root.findall('object'):
        label = obj.findtext('name')
        if label:
            label = label.strip()
        else:
            label = ''
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = bndbox.findtext('xmin')
        ymin = bndbox.findtext('ymin')
        xmax = bndbox.findtext('xmax')
        ymax = bndbox.findtext('ymax')
        # ensure values are stripped and valid
        try:
            xmin = xmin.strip()
            ymin = ymin.strip()
            xmax = xmax.strip()
            ymax = ymax.strip()
        except AttributeError:
            continue
        boxes.append({
            'filename': filename,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'label': label
        })
    return boxes


def generate_csv(xml_dir, output_csv):
    """
    Generate a CSV file from all XML annotation files in the specified directory.
    """
    fieldnames = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for root_dir, _, files in os.walk(xml_dir):
            for fname in files:
                if not fname.lower().endswith('.xml'):
                    continue
                xml_path = os.path.join(root_dir, fname)
                try:
                    boxes = parse_annotation(xml_path)
                    for box in boxes:
                        writer.writerow(box)
                except ET.ParseError as e:
                    print(f"Error parsing {xml_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert XML VOC annotations to CSV format.'
    )
    parser.add_argument('xml_dir', type=str, help='Directory containing XML files')
    parser.add_argument('output_csv', type=str, help='Output CSV file path')
    args = parser.parse_args()

    if not os.path.isdir(args.xml_dir):
        print(f"Error: {args.xml_dir} is not a valid directory.")
        return
    generate_csv(args.xml_dir, args.output_csv)
    print(f"CSV file generated at {args.output_csv}")

if __name__ == '__main__':
    main()
