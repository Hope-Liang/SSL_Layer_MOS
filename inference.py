import argparse

from extract_ssl_features import process_audio, get_ssl_model

parser = argparse.ArgumentParser(description='Path to model ckpt and audio')
parser.add_argument('--audio_path', type=str, help='Path to the test audio.')
parser.add_argument('--ssl_model', type=str, help='SSL feature extractor model name.')
parser.add_argument('--ssl_layer', type=int, help='SSL model layer, 0-indexed.')
parser.add_argument('--ckpt', type=str, help='Path to the model checkpoint.')

args = parser.parse_args()

def main():
	audio_path = args.audio_path
	ssl_model = args.ssl_model
	ssl_layer = args.ssl_layer
	ckpt_path = args.ckpt

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	ssl_model, ssl_model_sr = get_ssl_model(model_name)
	audio_tensor = process_audio(audio_path, model_sr, signal_duration=8.0).to(device)

	with torch.inference_mode():
		features, _ = model.extract_features(audio_tensor)

	selected_feature = features[i].squeeze().T.cpu().numpy()
	