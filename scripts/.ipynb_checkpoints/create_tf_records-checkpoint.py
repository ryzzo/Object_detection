import os
import io
import argparse
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/test TFRecord files using a pre-made label_map.pbtxt."
    )
    parser.add_argument(
        "--annotation_csv", required=True,
        help="Path to CSV file with columns: filename,xmin,ymin,xmax,ymax,class"
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Folder where all images live (filenames in CSV must match files inside)."
    )
    parser.add_argument(
        "--train_record", default="train.record",
        help="Where to write the train TFRecord file."
    )
    parser.add_argument(
        "--test_record", default="test.record",
        help="Where to write the test TFRecord file."
    )
    parser.add_argument(
        "--label_map", required=True,
        help="Path to the existing label_map.pbtxt. This provides string→int mapping."
    )
    parser.add_argument(
        "--test_split", type=float, default=0.2,
        help="Fraction of examples reserved for testing. (default: 0.2)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for splitting train/test. (default: 42)"
    )
    
    return parser.parse_args()


def load_label_map(label_map_path):
    """
    Uses the Object Detection API's label_map_util to read a label_map.pbtxt,
    returning a dict: {string_label: integer_id}.
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    return label_map_dict


def create_tf_example(filename, group, image_dir, class_to_id):
    """
    Builds a tf.train.Example from one image’s group (all rows in CSV
    that share the same filename), using the class_to_id mapping for
    matched integer label IDs.
    """
    image_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    image = Image.open(io.BytesIO(encoded_image))
    width, height = image.size

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes_id = [], []

    for _, row in group.iterrows():
        # Normalize bounding‐box coords to [0,1]
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)

        class_name = row['label']  # must match exactly a key in class_to_id
        classes_text.append(class_name.encode('utf8'))
        # Look up the integer ID from label_map.pbtxt
        classes_id.append(class_to_id[class_name])

    feature_dict = {
        'image/height':               dataset_util.int64_feature(height),
        'image/width':                dataset_util.int64_feature(width),
        'image/filename':             dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded':              dataset_util.bytes_feature(encoded_image),
        'image/format':               dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin':     dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':     dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':     dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':     dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':    dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':   dataset_util.int64_list_feature(classes_id),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def write_record(df, record_path, image_dir, class_to_id):
    """
    Groups the DataFrame by 'filename' and writes a single TFRecord file
    containing one tf.train.Example per distinct image.
    """
    grouped = df.groupby('filename')
    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, group in grouped:
            tf_example = create_tf_example(filename, group, image_dir, class_to_id)
            writer.write(tf_example.SerializeToString())
    print(f"Wrote {record_path} with {len(grouped)} examples.")


def main():
    args = parse_args()

    # Load label_map.pbtxt
    class_to_id = load_label_map(args.label_map)
    print("Loaded label_map.pbtxt:", class_to_id)

    # Read the CSV of annotations into a DataFrame
    df = pd.read_csv(args.annotation_csv)
    if 'label' not in df.columns:
        raise KeyError(f"Expected a column 'class' in CSV, but found {df.columns.tolist()}")

    # Split into train/test
    train_df = df.sample(frac=1 - args.test_split, random_state=args.random_state)
    test_df  = df.drop(train_df.index)

    # Write out TFRecords using the same class_to_id mapping
    write_record(train_df, args.train_record, args.image_dir, class_to_id)
    write_record(test_df,  args.test_record,  args.image_dir, class_to_id)

    print("✅ TFRecord creation complete.")


if __name__ == "__main__":
    main()
