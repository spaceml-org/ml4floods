import torch
import argparse
from ml4floods.data import create_gt
from ml4floods.scripts.inference import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run inference on S2 and Landsat-8/9 images')
    parser.add_argument("--image", required=True, help="Path to folder with tif files or tif file to make prediction")
    parser.add_argument("--model_path",
                        help="Path to experiment folder. Inside this folder there should be a config.json file and  a model weights file model.pt",
                        required=True)
    parser.add_argument("--output_folder",
                        help="Path to save the files. The name of the prediction will be the same as the S2 image."
                             "If not provided it will be saved in dirname(s2)/basename(model_path)/collection_name/",
                        required=False)
    parser.add_argument("--max_tile_size", help="Size to tile the GeoTIFFs", type=int, default=1_024)
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help="Overwrite the prediction if exists")
    parser.add_argument('--distinguish_flood_traces', default=False, action='store_true',
                        help="Use MNDWI to distinguish flood traces")
    parser.add_argument("--th_water", help="Threshold water used in v2 models (multioutput binary)",
                        type=float, default=.5)
    parser.add_argument("--th_brightness", help="Threshold brightness used to get cloud predictions",
                        type=float, default=create_gt.BRIGHTNESS_THRESHOLD)
    parser.add_argument('--device_name', default="cuda", help="Device name")
    parser.add_argument("--collection_name", choices=["Landsat", "S2"], default="S2")

    args = parser.parse_args()

    if args.device_name != "cpu" and not torch.cuda.is_available():
        raise NotImplementedError("Cuda is not available. run with --device_name cpu")

    main(model_path=args.model_path, s2folder_file=args.image, device_name=args.device_name,
         output_folder=args.output_folder, max_tile_size=args.max_tile_size, th_water=args.th_water,
         overwrite=args.overwrite, th_brightness=args.th_brightness, collection_name=args.collection_name,
         distinguish_flood_traces=args.distinguish_flood_traces)




