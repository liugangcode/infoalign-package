import pytest
import os
import pandas as pd
import numpy as np
from click.testing import CliRunner
from src.cli import predict_cli, DEFAULT_MODEL_PATH
from unittest.mock import patch, MagicMock

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        'SMILES': ['CC', 'CCC', 'CCCC']
    })
    csv_path = tmp_path / "test_input.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def mock_representer():
    with patch('src.cli.InfoAlignRepresenter') as mock:
        instance = mock.return_value
        instance.predict.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        yield mock  # Return the mock class, not the instance

@pytest.fixture
def mock_model_path(tmp_path):
    model_path = tmp_path / DEFAULT_MODEL_PATH
    model_path.touch()
    return str(model_path)

def test_predict_cli_basic(runner, sample_csv, tmp_path, mock_representer, mock_model_path):
    output_path = str(tmp_path / "output.npy")
    result = runner.invoke(predict_cli, [
        '--input', sample_csv,
        '--output', output_path,
        '--model-path', mock_model_path
    ])
    assert result.exit_code == 0
    assert os.path.exists(output_path)
    mock_representer.assert_called_once_with(model_path=mock_model_path)
    mock_representer.return_value.predict.assert_called_once()

def test_predict_cli_with_batch_size(runner, sample_csv, tmp_path, mock_representer, mock_model_path):
    output_path = str(tmp_path / "output.npy")
    result = runner.invoke(predict_cli, [
        '--input', sample_csv,
        '--output', output_path,
        '--batch-size', '2',
        '--model-path', mock_model_path
    ])
    assert result.exit_code == 0
    assert os.path.exists(output_path)
    mock_representer.return_value.predict.assert_called_once()
    args, kwargs = mock_representer.return_value.predict.call_args
    assert kwargs['batch_size'] == 2

def test_predict_cli_output_to_input_column(runner, sample_csv, mock_representer, mock_model_path):
    result = runner.invoke(predict_cli, [
        '--input', sample_csv,
        '--output', 'dummy_output.npy',
        '--output-to-input-column',
        '--model-path', mock_model_path
    ])
    assert result.exit_code == 0
    df = pd.read_csv(sample_csv)
    assert 'Representation' in df.columns
    assert isinstance(df['Representation'].iloc[0], str)

def test_predict_cli_invalid_input(runner, tmp_path, mock_model_path):
    invalid_input = str(tmp_path / "nonexistent.csv")
    result = runner.invoke(predict_cli, [
        '--input', invalid_input,
        '--output', 'output.npy',
        '--model-path', mock_model_path
    ])
    assert result.exit_code != 0

if __name__ == '__main__':
    pytest.main()