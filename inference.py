import argparse

from extract_ssl_features import process_audio

parser = argparse.ArgumentParser(description='Path to model ckpt and audio')
parser.add_argument('--ckpt', type=str, help='Path to the model checkpoint.')
parser.add_argument('--audio_path', type=str, help='Path to the test audio.')

args = parser.parse_args()

def main():
	audio_path = args.audio_path
	ckpt_path = args.ckpt

	audio_tensor = process_audio(audio_path)
	