import click
import pandas as pd
import numpy as np
import os
from .representer import InfoAlignRepresenter

DEFAULT_MODEL_PATH = "infoalign_model/pretrain.pt"

@click.command()
@click.option('--input', type=click.Path(exists=True), required=True, help='Path to input CSV file with SMILES')
@click.option('--output', type=click.Path(), required=True, help='Path to output NPY file for representations')
@click.option('--batch-size', type=int, default=32, help='Batch size for prediction (default: 32)')
@click.option('--output-to-input-column', is_flag=True, default=False, help='Add representation vector to input CSV file')
@click.option('--model-path', type=click.Path(), default=DEFAULT_MODEL_PATH, help='Path to the trained model')
def predict_cli(input, output, batch_size, output_to_input_column, model_path):
    representer = InfoAlignRepresenter(model_path=model_path)
    print('??')
    df = pd.read_csv(input)
    smiles_list = df['SMILES'].tolist()  # Assuming the column name is 'SMILES'
    
    representations = representer.predict(smiles_list, batch_size=batch_size)
    
    if output_to_input_column:
        # Convert representations to a list of strings
        rep_strings = [','.join(map(str, rep.astype(np.float32))) for rep in representations]
        df['Representation'] = rep_strings
        df.to_csv(input, index=False)  # Overwrite the input file
        click.echo(f"Representations added to input file: {input}")

    np.save(output, representations)
    click.echo(f"Predictions saved to {output}")

if __name__ == '__main__':
    predict_cli()