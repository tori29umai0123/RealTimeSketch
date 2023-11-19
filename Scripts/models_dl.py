import os
import requests

def download_file(url, output_file):
    # URLからファイルをダウンロードしてoutput_fileに保存する関数
    response = requests.get(url)
    with open(output_file, "wb") as f:
        f.write(response.content)

def download_files(repo_id, subfolder, files, cache_dir):
    # リポジトリから指定されたファイルをダウンロードする関数
    for file in files:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{subfolder}/{file}"
        output_file = os.path.join(cache_dir, file)
        if not os.path.exists(output_file):
            print(f"{file} を {url} から {output_file} にダウンロードしています...")
            download_file(url, output_file)
            print(f"{file} のダウンロードが完了しました！")
        else:
            print(f"{file} は既にダウンロードされています")

def check_and_download_model(model_dir, model_id, sub_dirs, files):
    # モデルディレクトリが存在しない場合、モデルをダウンロードする
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"モデルを {model_dir} にダウンロードしています。モデルID: {model_id}")

        # サブディレクトリごとにファイルをダウンロードする
        for sub_dir, sub_dir_files in sub_dirs:
            sub_dir_path = os.path.join(model_dir, sub_dir)
            if not os.path.exists(sub_dir_path):
                os.makedirs(sub_dir_path)
            download_files(model_id, sub_dir, sub_dir_files, sub_dir_path)

        # ルートディレクトリのファイルをダウンロードする
        for file in files:
            url = f"https://huggingface.co/{model_id}/resolve/main/{file}"
            output_file = os.path.join(model_dir, file)
            if not os.path.exists(output_file):
                print(f"{file} を {url} から {output_file} にダウンロードしています...")
                download_file(url, output_file)
                print(f"{file} のダウンロードが完了しました！")
            else:
                print(f"{file} は既にダウンロードされています")

        print("モデルのダウンロードが完了しました。")
    else:
        print("モデルは既にダウンロード済みです。")

def download_diffusion_model(model_dir):
    MODEL_ID = "852wa/SDHK"
    SUB_DIRS = [
        ("feature_extractor", ["preprocessor_config.json"]),
        ("safety_checker", ["config.json", "model.safetensors"]),
        ("scheduler", ["scheduler_config.json"]),
        ("text_encoder", ["config.json", "pytorch_model.bin"]),
        ("tokenizer", ["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]),
        ("unet", ["config.json", "diffusion_pytorch_model.bin"]),
        ("vae", ["config.json", "diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.bin"]),
    ]
    FILES = ["model_index.json"]

    check_and_download_model(model_dir, MODEL_ID, SUB_DIRS, FILES)


def download_vae_model(model_dir):
    MODEL_ID = "hakurei/waifu-diffusion-v1-4"
    SUB_DIRS = [
        ("vae", ["kl-f8-anime2.ckpt"]),
    ]
    check_and_download_model(model_dir, MODEL_ID, SUB_DIRS,[])

def download_contolnet_canny_model(model_dir):
    MODEL_ID = "lllyasviel/sd-controlnet-canny"
    FILES = ["config.json","diffusion_pytorch_model.bin"]
    check_and_download_model(model_dir, MODEL_ID, [], FILES)

def download_contolnet_scribble_model(model_dir):
    MODEL_ID = "lllyasviel/sd-controlnet-scribble"
    FILES = ["config.json","diffusion_pytorch_model.bin"]
    check_and_download_model(model_dir, MODEL_ID, [], FILES)

def download_lcm_lora_model(model_dir):
    MODEL_ID = "latent-consistency/lcm-lora-sdv1-5"
    FILES = ["pytorch_lora_weights.safetensors"]
    check_and_download_model(model_dir, MODEL_ID, [], FILES)


if __name__ == "__main__":
    stable_diffusion_path = "Models/SD"
    download_diffusion_model(stable_diffusion_path)

    contolnet_canny_path = "Models/controlnet/canny"
    download_contolnet_canny_model(contolnet_canny_path)

    contolnet_scribble_path = "Models/controlnet/scribble"
    download_contolnet_scribble_model(contolnet_scribble_path)

    lcm_lora_path = "Models/lcm_lora"
    download_lcm_lora_model(lcm_lora_path)
