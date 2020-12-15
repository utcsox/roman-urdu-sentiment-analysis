"""Module to perform hyper-parameter tuning on the MLP models"""
from typing import Any, Tuple
from src.data.make_dataset import load_roman_urdu_sentiment_analysis_dataset
from src.models.train_mlp_model import train_mlp_model
import argparse
import datetime


def tune_mlp_model(data: Tuple[Any, Any], output_dir: str):

    num_layers = [1, 3]
    num_units = [8, 64, 128]

    hparam = {
        'layers': [],
        'units': [],
        'accuracy': []
    }

    for layer in num_layers:
        for unit in num_units:
            hparam['layers'].append(layer)
            hparam['units'].append(unit)

            accuracy, loss = train_mlp_model(data=data, output_dir=output_dir, layers=layer, units=unit)
            print(f'Accuracy: {accuracy}, Parameters: (layers: {layer}, (units: {unit}')

            hparam['accuracy'].append(accuracy)

    print(hparam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../data/raw/',
        help='input data directory'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default='Roman Urdu DataSet.csv',
        help='input file name'
    )
    parser.add_argument(
        '--header_names',
        default= ['comment', 'sentiment', 'nan'],
        help='headers of columns of interest'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_',
        help='input data directory'
    )

    args, unparsed = parser.parse_known_args()
    dataset = load_roman_urdu_sentiment_analysis_dataset(args.data_dir, args.file_name, args.header_names)
    tune_mlp_model(dataset, args.output_dir)
