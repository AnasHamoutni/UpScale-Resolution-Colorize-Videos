{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "v2Jp-vSEeoaO"
      },
      "source": [
        "# ◢ UpScale Resolution & Colorize your own videos!\n",
        "\n",
        "\n",
        "My name is **ANAS HAMOUTNI**, a Data Scientist from Morocco, and this is my second Data Science Project on Github and is intended as a tool to UpScale Resolution and Colorize gifs and short videos, if you are trying to convert longer video on Colab you may hit the limit on processing space. Running the Jupyter notebook on your own machine is recommended (and faster) for larger video sizes.\n",
        "\n",
        "**Credits** are given to all the people who worked in this field."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "2UnCvVYqONil",
        "outputId": "52b72878-2da7-499c-83fa-992d43d797b8"
      },
      "source": [
        "#@title ##**Install all the necessary libraries** { display-mode: \"form\" }\n",
        "\n",
        "!pip install youtube_dl\n",
        "!pip install ffmpeg\n",
        "!pip install ffmpeg-python\n",
        "!pip install torchvision==0.5\n",
        "!pip install torch==1.4\n",
        "\n",
        "from IPython.display import clear_output\n",
        "from google.colab import files\n",
        "import imageio\n",
        "import youtube_dl\n",
        "import cv2\n",
        "import os\n",
        "import torch\n",
        "import fastai\n",
        "import ffmpeg\n",
        "import os.path as osp\n",
        "import logging\n",
        "import shutil\n",
        "import re\n",
        "import gc\n",
        "from PIL import Image\n",
        "from tqdm import tqdm\n",
        "from os import path\n",
        "import numpy as np\n",
        "import moviepy.editor as mpy\n",
        "from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter\n",
        "from pathlib import Path\n",
        "import sys\n",
        "import glob\n",
        "from IPython import display as ipythondisplay\n",
        "from IPython.display import Image as ipythonimage\n",
        "torch.backends.cudnn.benchmark=True\n",
        "%matplotlib inline\n",
        "\n",
        "!rm -rf sample_data\n",
        "#@title ##**Clone the repository and download the necessary components** { display-mode: \"form\" }\n",
        "%cd /content\n",
        "!git clone https://github.com/xinntao/ESRGAN.git\n",
        "!cp -r video.mp4 /content/ESRGAN/\n",
        "%cd /content/ESRGAN\n",
        "!git checkout tags/old-arch\n",
        "model_url = \"https://www.dropbox.com/s/vouc15j8jjp2o5n/RRDB_ESRGAN_x4_old_arch.pth?dl=0\"\n",
        "!wget $model_url --content-disposition -P models\n",
        "import architecture as arch\n",
        "import os.path\n",
        "!mkdir frames\n",
        "!rm -rf results/baboon_ESRGAN.png\n",
        "#@title ##**Clone the repository and install the necessary dependencies** { display-mode: \"form\" }\n",
        "%cd /content\n",
        "!git clone https://github.com/jantic/DeOldify.git\n",
        "%cd /content/DeOldify\n",
        "!pip install -r colab_requirements.txt\n"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting youtube_dl\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/87/5e/fddd736a5ded17db328b0157508d935980c07d0fc2c5b5c6105519c5fb6f/youtube_dl-2020.12.7-py2.py3-none-any.whl (1.8MB)\n",
            "\r\u001b[K     |▏                               | 10kB 21.0MB/s eta 0:00:01\r\u001b[K     |▍                               | 20kB 28.9MB/s eta 0:00:01\r\u001b[K     |▌                               | 30kB 33.0MB/s eta 0:00:01\r\u001b[K     |▊                               | 40kB 34.1MB/s eta 0:00:01\r\u001b[K     |▉                               | 51kB 35.8MB/s eta 0:00:01\r\u001b[K     |█                               | 61kB 38.3MB/s eta 0:00:01\r\u001b[K     |█▎                              | 71kB 29.1MB/s eta 0:00:01\r\u001b[K     |█▍                              | 81kB 24.9MB/s eta 0:00:01\r\u001b[K     |█▋                              | 92kB 26.3MB/s eta 0:00:01\r\u001b[K     |█▊                              | 102kB 24.7MB/s eta 0:00:01\r\u001b[K     |██                              | 112kB 24.7MB/s eta 0:00:01\r\u001b[K     |██▏                             | 122kB 24.7MB/s eta 0:00:01\r\u001b[K     |██▎                             | 133kB 24.7MB/s eta 0:00:01\r\u001b[K     |██▌                             | 143kB 24.7MB/s eta 0:00:01\r\u001b[K     |██▋                             | 153kB 24.7MB/s eta 0:00:01\r\u001b[K     |██▉                             | 163kB 24.7MB/s eta 0:00:01\r\u001b[K     |███                             | 174kB 24.7MB/s eta 0:00:01\r\u001b[K     |███▏                            | 184kB 24.7MB/s eta 0:00:01\r\u001b[K     |███▍                            | 194kB 24.7MB/s eta 0:00:01\r\u001b[K     |███▌                            | 204kB 24.7MB/s eta 0:00:01\r\u001b[K     |███▊                            | 215kB 24.7MB/s eta 0:00:01\r\u001b[K     |████                            | 225kB 24.7MB/s eta 0:00:01\r\u001b[K     |████                            | 235kB 24.7MB/s eta 0:00:01\r\u001b[K     |████▎                           | 245kB 24.7MB/s eta 0:00:01\r\u001b[K     |████▍                           | 256kB 24.7MB/s eta 0:00:01\r\u001b[K     |████▋                           | 266kB 24.7MB/s eta 0:00:01\r\u001b[K     |████▉                           | 276kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████                           | 286kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████▏                          | 296kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████▎                          | 307kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████▌                          | 317kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████▊                          | 327kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████▉                          | 337kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████                          | 348kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████▏                         | 358kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████▍                         | 368kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████▋                         | 378kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████▊                         | 389kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████                         | 399kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████                         | 409kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████▎                        | 419kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████▌                        | 430kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████▋                        | 440kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████▉                        | 450kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████                        | 460kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████▏                       | 471kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████▍                       | 481kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████▌                       | 491kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████▊                       | 501kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████▉                       | 512kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████                       | 522kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████▏                      | 532kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████▍                      | 542kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████▋                      | 552kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████▊                      | 563kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████                      | 573kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████                      | 583kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████▎                     | 593kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████▌                     | 604kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████▋                     | 614kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████▉                     | 624kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████                     | 634kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████▏                    | 645kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████▍                    | 655kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████▌                    | 665kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████▊                    | 675kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████▉                    | 686kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████                    | 696kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████▎                   | 706kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████▍                   | 716kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████▋                   | 727kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████▊                   | 737kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████                   | 747kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████▏                  | 757kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████▎                  | 768kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████▌                  | 778kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████▋                  | 788kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████▉                  | 798kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████                  | 808kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████▏                 | 819kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████▍                 | 829kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████▌                 | 839kB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████▊                 | 849kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████                 | 860kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████                 | 870kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████▎                | 880kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████▍                | 890kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████▋                | 901kB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████▉                | 911kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████                | 921kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████▏               | 931kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████▎               | 942kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████▌               | 952kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████▊               | 962kB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████▉               | 972kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████               | 983kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████▏              | 993kB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████▍              | 1.0MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████▌              | 1.0MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████▊              | 1.0MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████              | 1.0MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████              | 1.0MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████▎             | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████▍             | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████▋             | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████▉             | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████             | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████▏            | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████▎            | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████▌            | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████▊            | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████▉            | 1.1MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████            | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████▏           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████▍           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████▋           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████▊           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████           | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████▎          | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████▌          | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████▋          | 1.2MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████▉          | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████          | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████▏         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████▍         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████▌         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████▊         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████▉         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████         | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████▎        | 1.3MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████▍        | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████▋        | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████▊        | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████        | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████▏       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████▎       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████▌       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████▋       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████▉       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████       | 1.4MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████▏      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████▍      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████▌      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████▊      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████      | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████▎     | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████▍     | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████▋     | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████▊     | 1.5MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████     | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████▏    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████▎    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████▌    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████▋    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████▉    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████    | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████▏   | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████▍   | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████▌   | 1.6MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████▊   | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████   | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████   | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████▎  | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████▍  | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████▋  | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |█████████████████████████████▉  | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████  | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████▏ | 1.7MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████▎ | 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████▌ | 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████▊ | 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |██████████████████████████████▉ | 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████ | 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████▏| 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████▍| 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████▋| 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |███████████████████████████████▊| 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████████| 1.8MB 24.7MB/s eta 0:00:01\r\u001b[K     |████████████████████████████████| 1.9MB 24.7MB/s \n",
            "\u001b[?25hInstalling collected packages: youtube-dl\n",
            "Successfully installed youtube-dl-2020.12.7\n",
            "Collecting ffmpeg\n",
            "  Downloading https://files.pythonhosted.org/packages/f0/cc/3b7408b8ecf7c1d20ad480c3eaed7619857bf1054b690226e906fdf14258/ffmpeg-1.4.tar.gz\n",
            "Building wheels for collected packages: ffmpeg\n",
            "  Building wheel for ffmpeg (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for ffmpeg: filename=ffmpeg-1.4-cp36-none-any.whl size=6083 sha256=4325428ebcc3273fa63d756591a88c7ff09e670529c02853a1d4a9c5fc7a0230\n",
            "  Stored in directory: /root/.cache/pip/wheels/b6/68/c3/a05a35f647ba871e5572b9bbfc0b95fd1c6637a2219f959e7a\n",
            "Successfully built ffmpeg\n",
            "Installing collected packages: ffmpeg\n",
            "Successfully installed ffmpeg-1.4\n",
            "Collecting ffmpeg-python\n",
            "  Downloading https://files.pythonhosted.org/packages/d7/0c/56be52741f75bad4dc6555991fabd2e07b432d333da82c11ad701123888a/ffmpeg_python-0.2.0-py3-none-any.whl\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from ffmpeg-python) (0.16.0)\n",
            "Installing collected packages: ffmpeg-python\n",
            "Successfully installed ffmpeg-python-0.2.0\n",
            "Collecting torchvision==0.5\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/7e/90/6141bf41f5655c78e24f40f710fdd4f8a8aff6c8b7c6f0328240f649bdbe/torchvision-0.5.0-cp36-cp36m-manylinux1_x86_64.whl (4.0MB)\n",
            "\u001b[K     |████████████████████████████████| 4.0MB 20.5MB/s \n",
            "\u001b[?25hRequirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5) (1.15.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5) (1.18.5)\n",
            "Requirement already satisfied: pillow>=4.1.1 in /usr/local/lib/python3.6/dist-packages (from torchvision==0.5) (7.0.0)\n",
            "Collecting torch==1.4.0\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/24/19/4804aea17cd136f1705a5e98a00618cb8f6ccc375ad8bfa437408e09d058/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl (753.4MB)\n",
            "\u001b[K     |████████████████████████████████| 753.4MB 22kB/s \n",
            "\u001b[?25hInstalling collected packages: torch, torchvision\n",
            "  Found existing installation: torch 1.7.0+cu101\n",
            "    Uninstalling torch-1.7.0+cu101:\n",
            "      Successfully uninstalled torch-1.7.0+cu101\n",
            "  Found existing installation: torchvision 0.8.1+cu101\n",
            "    Uninstalling torchvision-0.8.1+cu101:\n",
            "      Successfully uninstalled torchvision-0.8.1+cu101\n",
            "Successfully installed torch-1.4.0 torchvision-0.5.0\n",
            "Requirement already satisfied: torch==1.4 in /usr/local/lib/python3.6/dist-packages (1.4.0)\n",
            "Imageio: 'ffmpeg-linux64-v3.3.1' was not found on your computer; downloading it now.\n",
            "Try 1. Download from https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg-linux64-v3.3.1 (43.8 MB)\n",
            "Downloading: 8192/45929032 bytes (0.0%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b3784704/45929032 bytes (8.2%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b7766016/45929032 bytes (16.9%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b11993088/45929032 bytes (26.1%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b16089088/45929032 bytes (35.0%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b20201472/45929032 bytes (44.0%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b24379392/45929032 bytes (53.1%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b28475392/45929032 bytes (62.0%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b32636928/45929032 bytes (71.1%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b36708352/45929032 bytes (79.9%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b40894464/45929032 bytes (89.0%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b44892160/45929032 bytes (97.7%)\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b45929032/45929032 bytes (100.0%)\n",
            "  Done\n",
            "File saved as /root/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1.\n",
            "/content\n",
            "Cloning into 'ESRGAN'...\n",
            "remote: Enumerating objects: 216, done.\u001b[K\n",
            "remote: Total 216 (delta 0), reused 0 (delta 0), pack-reused 216\u001b[K\n",
            "Receiving objects: 100% (216/216), 24.86 MiB | 45.86 MiB/s, done.\n",
            "Resolving deltas: 100% (80/80), done.\n",
            "cp: cannot stat 'video.mp4': No such file or directory\n",
            "/content/ESRGAN\n",
            "Note: checking out 'tags/old-arch'.\n",
            "\n",
            "You are in 'detached HEAD' state. You can look around, make experimental\n",
            "changes and commit them, and you can discard any commits you make in this\n",
            "state without impacting any branches by performing another checkout.\n",
            "\n",
            "If you want to create a new branch to retain commits you create, you may\n",
            "do so (now or later) by using -b with the checkout command again. Example:\n",
            "\n",
            "  git checkout -b <new-branch-name>\n",
            "\n",
            "HEAD is now at aceb857 cuda version (issues #30)\n",
            "--2020-12-06 23:35:01--  https://www.dropbox.com/s/vouc15j8jjp2o5n/RRDB_ESRGAN_x4_old_arch.pth?dl=0\n",
            "Resolving www.dropbox.com (www.dropbox.com)... 162.125.6.1, 2620:100:601a:1::a27d:701\n",
            "Connecting to www.dropbox.com (www.dropbox.com)|162.125.6.1|:443... connected.\n",
            "HTTP request sent, awaiting response... 301 Moved Permanently\n",
            "Location: /s/raw/vouc15j8jjp2o5n/RRDB_ESRGAN_x4_old_arch.pth [following]\n",
            "--2020-12-06 23:35:01--  https://www.dropbox.com/s/raw/vouc15j8jjp2o5n/RRDB_ESRGAN_x4_old_arch.pth\n",
            "Reusing existing connection to www.dropbox.com:443.\n",
            "HTTP request sent, awaiting response... 302 Found\n",
            "Location: https://uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com/cd/0/inline/BEln1xUrkMgBGrkykmSXVyqXZ9UV6WSinxxfhY6m-bEvzC1pQrTxZ8cBEXhc9RSndeIr53i8luTW66XyADVH-vNsOe9joJ_X1BLis5fPb0YiMQ/file# [following]\n",
            "--2020-12-06 23:35:02--  https://uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com/cd/0/inline/BEln1xUrkMgBGrkykmSXVyqXZ9UV6WSinxxfhY6m-bEvzC1pQrTxZ8cBEXhc9RSndeIr53i8luTW66XyADVH-vNsOe9joJ_X1BLis5fPb0YiMQ/file\n",
            "Resolving uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com (uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com)... 162.125.6.15, 2620:100:601c:15::a27d:60f\n",
            "Connecting to uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com (uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com)|162.125.6.15|:443... connected.\n",
            "HTTP request sent, awaiting response... 302 Found\n",
            "Location: /cd/0/inline2/BElaBXzEQUjLlOkGEOBzotMkJCWCHa5bWQLYWK6CrUJmaIz1Tr9V9TV3XVfDqhfIw4UOvNlCcHEeSZlO2O-nm-env-MievV7RIXRO-x6KPXxN83AcOcd_OpZHOoNNWuANZ-hsngU6DNr-s2JP7UeddrwSQRrQUXaqGUz_GRem3EktyA_JHSlSs9xPZjP8ZQO2a0WKa6mWGYcR0Sy-ZAMDfJq2P7Nvffko23q8ZMhyhRebRdYdaEOqiY5iLvsDUvHWxSBKyaxGuECGtZbzr5R5AYNa2DLAJnkT0JjtrVAO2txntd6AEuE5zzFXVc0022p71Lk7uXW0hPaVMrU4jCKswM7/file [following]\n",
            "--2020-12-06 23:35:02--  https://uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com/cd/0/inline2/BElaBXzEQUjLlOkGEOBzotMkJCWCHa5bWQLYWK6CrUJmaIz1Tr9V9TV3XVfDqhfIw4UOvNlCcHEeSZlO2O-nm-env-MievV7RIXRO-x6KPXxN83AcOcd_OpZHOoNNWuANZ-hsngU6DNr-s2JP7UeddrwSQRrQUXaqGUz_GRem3EktyA_JHSlSs9xPZjP8ZQO2a0WKa6mWGYcR0Sy-ZAMDfJq2P7Nvffko23q8ZMhyhRebRdYdaEOqiY5iLvsDUvHWxSBKyaxGuECGtZbzr5R5AYNa2DLAJnkT0JjtrVAO2txntd6AEuE5zzFXVc0022p71Lk7uXW0hPaVMrU4jCKswM7/file\n",
            "Reusing existing connection to uc8d9b6457fc0f2bf80087103973.dl.dropboxusercontent.com:443.\n",
            "HTTP request sent, awaiting response... 200 OK\n",
            "Length: 66954395 (64M) [application/octet-stream]\n",
            "Saving to: ‘models/RRDB_ESRGAN_x4_old_arch.pth’\n",
            "\n",
            "RRDB_ESRGAN_x4_old_ 100%[===================>]  63.85M   195MB/s    in 0.3s    \n",
            "\n",
            "2020-12-06 23:35:04 (195 MB/s) - ‘models/RRDB_ESRGAN_x4_old_arch.pth’ saved [66954395/66954395]\n",
            "\n",
            "/content\n",
            "Cloning into 'DeOldify'...\n",
            "remote: Enumerating objects: 2187, done.\u001b[K\n",
            "remote: Total 2187 (delta 0), reused 0 (delta 0), pack-reused 2187\u001b[K\n",
            "Receiving objects: 100% (2187/2187), 69.39 MiB | 33.59 MiB/s, done.\n",
            "Resolving deltas: 100% (996/996), done.\n",
            "/content/DeOldify\n",
            "Collecting fastai==1.0.51\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/44/cc/dcc702cf43bb8c908d172e5be156615928f962366a20834c320cbca2b9d0/fastai-1.0.51-py3-none-any.whl (214kB)\n",
            "\u001b[K     |████████████████████████████████| 215kB 13.2MB/s \n",
            "\u001b[?25hCollecting tensorboardX==1.6\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/5c/76/89dd44458eb976347e5a6e75eb79fecf8facd46c1ce259bad54e0044ea35/tensorboardX-1.6-py2.py3-none-any.whl (129kB)\n",
            "\u001b[K     |████████████████████████████████| 133kB 29.6MB/s \n",
            "\u001b[?25hRequirement already satisfied: ffmpeg in /usr/local/lib/python3.6/dist-packages (from -r colab_requirements.txt (line 3)) (1.4)\n",
            "Collecting ffmpeg-python==0.1.17\n",
            "  Downloading https://files.pythonhosted.org/packages/3d/10/330cbc8e63d072d40413f4d470444a6a1e8c8c6a80b2a4ac302d1252ca1b/ffmpeg_python-0.1.17-py3-none-any.whl\n",
            "Requirement already satisfied: youtube-dl>=2019.4.17 in /usr/local/lib/python3.6/dist-packages (from -r colab_requirements.txt (line 5)) (2020.12.7)\n",
            "Collecting jupyterlab\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/31/7b/cd66f306c31a84a53c6a3a86e296586e8664f407a6ac5b7cfe6a433aa8c4/jupyterlab-2.2.9-py3-none-any.whl (7.9MB)\n",
            "\u001b[K     |████████████████████████████████| 7.9MB 37.2MB/s \n",
            "\u001b[?25hRequirement already satisfied: opencv-python>=3.3.0.10 in /usr/local/lib/python3.6/dist-packages (from -r colab_requirements.txt (line 7)) (4.1.2.30)\n",
            "Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from -r colab_requirements.txt (line 8)) (7.0.0)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.1.4)\n",
            "Requirement already satisfied: spacy>=2.0.18 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.2.4)\n",
            "Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.18.5)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.23.0)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (3.2.2)\n",
            "Requirement already satisfied: torch>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.4.0)\n",
            "Requirement already satisfied: numexpr in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.7.1)\n",
            "Collecting typing\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/05/d9/6eebe19d46bd05360c9a9aae822e67a80f9242aabbfc58b641b957546607/typing-3.7.4.3.tar.gz (78kB)\n",
            "\u001b[K     |████████████████████████████████| 81kB 12.3MB/s \n",
            "\u001b[?25hRequirement already satisfied: torchvision in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (0.5.0)\n",
            "Requirement already satisfied: nvidia-ml-py3 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (7.352.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (20.4)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.4.1)\n",
            "Requirement already satisfied: fastprogress>=0.1.19 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.0.0)\n",
            "Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (3.13)\n",
            "Requirement already satisfied: dataclasses; python_version < \"3.7\" in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (0.8)\n",
            "Requirement already satisfied: bottleneck in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.3.2)\n",
            "Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.6/dist-packages (from fastai==1.0.51->-r colab_requirements.txt (line 1)) (4.6.3)\n",
            "Requirement already satisfied: protobuf>=3.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorboardX==1.6->-r colab_requirements.txt (line 2)) (3.12.4)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from tensorboardX==1.6->-r colab_requirements.txt (line 2)) (1.15.0)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from ffmpeg-python==0.1.17->-r colab_requirements.txt (line 4)) (0.16.0)\n",
            "Requirement already satisfied: notebook>=4.3.1 in /usr/local/lib/python3.6/dist-packages (from jupyterlab->-r colab_requirements.txt (line 6)) (5.3.1)\n",
            "Requirement already satisfied: tornado!=6.0.0,!=6.0.1,!=6.0.2 in /usr/local/lib/python3.6/dist-packages (from jupyterlab->-r colab_requirements.txt (line 6)) (5.1.1)\n",
            "Collecting jupyterlab-server<2.0,>=1.1.5\n",
            "  Downloading https://files.pythonhosted.org/packages/b4/eb/560043dcd8376328f8b98869efed66ef68307278406ab99c7f63a34d4ae2/jupyterlab_server-1.2.0-py3-none-any.whl\n",
            "Requirement already satisfied: jinja2>=2.10 in /usr/local/lib/python3.6/dist-packages (from jupyterlab->-r colab_requirements.txt (line 6)) (2.11.2)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.6/dist-packages (from pandas->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.8.1)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2018.9)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.0.4)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (0.8.0)\n",
            "Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.0.4)\n",
            "Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.1.3)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (3.0.4)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (4.41.1)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (50.3.2)\n",
            "Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.0.0)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.0.4)\n",
            "Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (7.4.0)\n",
            "Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (0.4.1)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.51->-r colab_requirements.txt (line 1)) (3.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2020.11.8)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.24.3)\n",
            "Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.4.7)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.51->-r colab_requirements.txt (line 1)) (1.3.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->fastai==1.0.51->-r colab_requirements.txt (line 1)) (0.10.0)\n",
            "Requirement already satisfied: jupyter-core>=4.4.0 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (4.7.0)\n",
            "Requirement already satisfied: jupyter-client>=5.2.0 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (5.3.5)\n",
            "Requirement already satisfied: nbformat in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (5.0.8)\n",
            "Requirement already satisfied: nbconvert in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (5.6.1)\n",
            "Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.9.1)\n",
            "Requirement already satisfied: Send2Trash in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (1.5.0)\n",
            "Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.2.0)\n",
            "Requirement already satisfied: traitlets>=4.2.1 in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (4.3.3)\n",
            "Requirement already satisfied: ipykernel in /usr/local/lib/python3.6/dist-packages (from notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (4.10.1)\n",
            "Collecting json5\n",
            "  Downloading https://files.pythonhosted.org/packages/2b/81/22bf51a5bc60dde18bb6164fd597f18ee683de8670e141364d9c432dd3cf/json5-0.9.5-py2.py3-none-any.whl\n",
            "Collecting jsonschema>=3.0.1\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/c5/8f/51e89ce52a085483359217bc72cdbf6e75ee595d5b1d4b5ade40c7e018b8/jsonschema-3.2.0-py2.py3-none-any.whl (56kB)\n",
            "\u001b[K     |████████████████████████████████| 61kB 9.1MB/s \n",
            "\u001b[?25hRequirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from jinja2>=2.10->jupyterlab->-r colab_requirements.txt (line 6)) (1.1.1)\n",
            "Requirement already satisfied: importlib-metadata>=0.20; python_version < \"3.8\" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (2.0.0)\n",
            "Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.6/dist-packages (from jupyter-client>=5.2.0->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (20.0.0)\n",
            "Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.3)\n",
            "Requirement already satisfied: testpath in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.4.4)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (2.6.1)\n",
            "Requirement already satisfied: defusedxml in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.6.0)\n",
            "Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.8.4)\n",
            "Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (1.4.3)\n",
            "Requirement already satisfied: bleach in /usr/local/lib/python3.6/dist-packages (from nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (3.2.1)\n",
            "Requirement already satisfied: ptyprocess; os_name != \"nt\" in /usr/local/lib/python3.6/dist-packages (from terminado>=0.8.1->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.6.0)\n",
            "Requirement already satisfied: decorator in /usr/local/lib/python3.6/dist-packages (from traitlets>=4.2.1->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (4.4.2)\n",
            "Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.6/dist-packages (from ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (5.5.0)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from jsonschema>=3.0.1->jupyterlab-server<2.0,>=1.1.5->jupyterlab->-r colab_requirements.txt (line 6)) (20.3.0)\n",
            "Requirement already satisfied: pyrsistent>=0.14.0 in /usr/local/lib/python3.6/dist-packages (from jsonschema>=3.0.1->jupyterlab-server<2.0,>=1.1.5->jupyterlab->-r colab_requirements.txt (line 6)) (0.17.3)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy>=2.0.18->fastai==1.0.51->-r colab_requirements.txt (line 1)) (3.4.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.6/dist-packages (from bleach->nbconvert->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.5.1)\n",
            "Requirement already satisfied: pexpect; sys_platform != \"win32\" in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (4.8.0)\n",
            "Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (1.0.18)\n",
            "Requirement already satisfied: pickleshare in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.7.5)\n",
            "Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.6/dist-packages (from ipython>=4.0.0->ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.8.1)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.6/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0->ipykernel->notebook>=4.3.1->jupyterlab->-r colab_requirements.txt (line 6)) (0.2.5)\n",
            "Building wheels for collected packages: typing\n",
            "  Building wheel for typing (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for typing: filename=typing-3.7.4.3-cp36-none-any.whl size=26307 sha256=c8f7b5b0219cf939310f043ed56026d1651f0dfd295d9d4504a599bb87e0bad4\n",
            "  Stored in directory: /root/.cache/pip/wheels/2d/04/41/8e1836e79581989c22eebac3f4e70aaac9af07b0908da173be\n",
            "Successfully built typing\n",
            "\u001b[31mERROR: nbclient 0.5.1 has requirement jupyter-client>=6.1.5, but you'll have jupyter-client 5.3.5 which is incompatible.\u001b[0m\n",
            "Installing collected packages: typing, fastai, tensorboardX, ffmpeg-python, json5, jsonschema, jupyterlab-server, jupyterlab\n",
            "  Found existing installation: fastai 1.0.61\n",
            "    Uninstalling fastai-1.0.61:\n",
            "      Successfully uninstalled fastai-1.0.61\n",
            "  Found existing installation: ffmpeg-python 0.2.0\n",
            "    Uninstalling ffmpeg-python-0.2.0:\n",
            "      Successfully uninstalled ffmpeg-python-0.2.0\n",
            "  Found existing installation: jsonschema 2.6.0\n",
            "    Uninstalling jsonschema-2.6.0:\n",
            "      Successfully uninstalled jsonschema-2.6.0\n",
            "Successfully installed fastai-1.0.51 ffmpeg-python-0.1.17 json5-0.9.5 jsonschema-3.2.0 jupyterlab-2.2.9 jupyterlab-server-1.2.0 tensorboardX-1.6 typing-3.7.4.3\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "fastai",
                  "ffmpeg",
                  "typing"
                ]
              }
            }
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "a_jZfhQDQIyn"
      },
      "source": [
        "#@title ##**Upload video** { display-mode: \"form\" }\n",
        "#@markdown *Enter a link to the video below (for example, YouTube or Twitter), or leave the **source_url** field blank (in this case, you will be asked to upload the video from your computer).*\n",
        "source_url = 'https://www.youtube.com/watch?v=ZB-HI9fQBE0' #@param {type:\"string\"}\n",
        "\n",
        "if source_url == '':\n",
        "  uploaded = files.upload()\n",
        "  for fn in uploaded.keys():\n",
        "    print('User uploaded file \"{name}\" with length {length} bytes'.format(\n",
        "        name=fn, length=len(uploaded[fn])))\n",
        "  os.rename(fn, fn.replace(\" \", \"\"))\n",
        "  fn = fn.replace(\" \", \"\")\n",
        "  file_name = \"downloaded_video.\" + fn.split(\".\")[-1]\n",
        "  !mv -f $fn $file_name\n",
        "\n",
        "else:\n",
        "  try:\n",
        "    ydl_opts = {\n",
        "        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',\n",
        "        'outtmpl': 'downloaded_video.mp4',\n",
        "        }\n",
        "    with youtube_dl.YoutubeDL(ydl_opts) as ydl:\n",
        "      ydl.download([source_url])\n",
        "    file_name = 'downloaded_video.mp4'\n",
        "  \n",
        "  except BaseException:\n",
        "    !wget $source_url\n",
        "    fn = source_url.split('/')[-1]\n",
        "    os.rename(fn, fn.replace(\" \", \"\"))\n",
        "    fn = fn.replace(\" \", \"\")\n",
        "    file_name = \"downloaded_video.\" + fn.split(\".\")[-1]\n",
        "    !mv -f $fn $file_name\n",
        "\n",
        "!cp -r downloaded_video.mp4 video.mp4\n",
        "clear_output()\n",
        "fps_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FPS))\n",
        "frames_of_video = int(cv2.VideoCapture(file_name).get(cv2.CAP_PROP_FRAME_COUNT))\n",
        "#@markdown *Downloading videos lasting longer than one minute is not recommended. In addition, don't upload video that have a \"space\" or a \"dot\" in the title.*\n",
        "\n",
        "#@markdown *If there is an error during execution, then run this block again*"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Os_TObrMN_LR",
        "outputId": "0adebbd2-1012-49b3-cd44-ffbe7fac3aa4"
      },
      "source": [
        "#@title ##**Split the video into frames** { display-mode: \"form\" }\n",
        "%cd /content/ESRGAN/\n",
        "!mkdir frames\n",
        "\n",
        "frames_of_video = int(cv2.VideoCapture(\"video.mp4\").get(cv2.CAP_PROP_FRAME_COUNT))\n",
        "fps_of_video = int(cv2.VideoCapture(\"video.mp4\").get(cv2.CAP_PROP_FPS))\n",
        "vidcap = cv2.VideoCapture('video.mp4')\n",
        "success,image = vidcap.read()\n",
        "count = 0\n",
        "success = True\n",
        "while success:\n",
        "  cv2.imwrite(\"frames/frame%09d.jpg\" % count, image)\n",
        "  success,image = vidcap.read()\n",
        "  count += 1\n"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/ESRGAN\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 248
        },
        "id": "mC9wy0sMNoPJ",
        "outputId": "49b520b8-bf4e-4392-f2e6-96af2e7ecadf"
      },
      "source": [
        "#@title ##**Upscale resolution** { display-mode: \"form\" }\n",
        "#@markdown **How many times to upscale the resolution:**\n",
        "upscale = 4 #@param {type: \"slider\", min: 4, max: 8}\n",
        "%cd /content/ESRGAN\n",
        "%env CUDA_VISIBLE_DEVICES=0\n",
        "device = torch.device('cuda')\n",
        "model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu', \\\n",
        "                        mode='CNA', res_scale=1, upsample_mode='upconv')\n",
        "model.load_state_dict(torch.load('models/{:s}'.format('RRDB_ESRGAN_x4_old_arch.pth')), strict=True)\n",
        "model.eval()\n",
        "for k, v in model.named_parameters():\n",
        "    v.requires_grad = False\n",
        "model = model.to(device)\n",
        "\n",
        "count_frames = 0\n",
        "\n",
        "for path in glob.glob('frames/*'):\n",
        "    base = os.path.splitext(os.path.basename(path))[0]\n",
        "    img = cv2.imread(path, cv2.IMREAD_COLOR)\n",
        "    img = img * 1.0 / 255\n",
        "    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()\n",
        "    img_LR = img.unsqueeze(0)\n",
        "    img_LR = img_LR.to(device)\n",
        "\n",
        "    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()\n",
        "    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))\n",
        "    output = (output * 255.0).round()\n",
        "    path = 'results/{:s}_rlt.png'.format(base)\n",
        "    cv2.imwrite(path, output)\n",
        "    count_frames += 1\n",
        "    clear_output()\n",
        "    print(\"Processed: {} из {}\".format(str(count_frames), str(frames_of_video)))\n",
        "clear_output()"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Processed: 2 из 10368\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-14-3add901d4d1d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     26\u001b[0m     \u001b[0moutput\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0moutput\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0;36m255.0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mround\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     27\u001b[0m     \u001b[0mpath\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m'results/{:s}_rlt.png'\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbase\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 28\u001b[0;31m     \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moutput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     29\u001b[0m     \u001b[0mcount_frames\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     30\u001b[0m     \u001b[0mclear_output\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hFF0Ho-JSDHn"
      },
      "source": [
        "#@title ##**Collecting a video file** { display-mode: \"form\" }\n",
        "import os\n",
        "import shutil\n",
        "import imageio\n",
        "shutil.rmtree('/content/ESRGAN/frames', ignore_errors=True)\n",
        "%cd /content/ESRGAN/\n",
        "\n",
        "frames = []\n",
        "img = os.listdir('results/')\n",
        "img.sort()\n",
        "for i in img:\n",
        "  frames.append(imageio.imread(\"results/\"+i))\n",
        "frames = np.array(frames)\n",
        "imageio.mimsave(\"upscaled_video.mp4\", frames, fps=30)\n",
        "\n",
        "print('Сборка завершена')\n",
        "!cp -r upscaled_video.mp4 /content/upscaled_video.mp4\n",
        "clear_output()"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XJx3SmZvSko_"
      },
      "source": [
        "#@title ##**Get result** { display-mode: \"form\" }\n",
        "what_next = 'play' #@param [\"play\", \"download\"]\n",
        "if what_next == \"play\":\n",
        "  display(mpy.ipython_display(\"/content/upscaled_video.mp4\", height=400, autoplay=1, loop=1, maxduration=600))\n",
        "else:\n",
        "  !zip -r /content/results.zip /content/ESRGAN/results\n",
        "  from google.colab import files\n",
        "  files.download(\"/content/results.zip\")\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SDH27RoAQxy5"
      },
      "source": [
        "#@title ##**Upload the pre-trained model** { display-mode: \"form\" }\n",
        "from deoldify.visualize import *\n",
        "!mkdir 'models'\n",
        "try:\n",
        "  !rm -rf models/ColorizeVideo_gen.pth\n",
        "  !gdown https://drive.google.com/uc?id=1-3ONnPYcX9fOqnY-pUKYGJOd6XeFon9X -O ./models/ColorizeVideo_gen.pth\n",
        "except BaseException:\n",
        "  !rm -rf models/ColorizeVideo_gen.pth\n",
        "  !wget https://www.dropbox.com/s/336vn9y4qwyg9yz/ColorizeVideo_gen.pth?dl=0 -O ./models/ColorizeVideo_gen.pth\n",
        "colorizer = get_video_colorizer()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 384
        },
        "id": "fUffPJCxQ-NK",
        "outputId": "c835613b-d6da-4974-f61c-f5aebfe6124c"
      },
      "source": [
        "#@title ##**Colorize the video** { display-mode: \"form\" }\n",
        "#@markdown **21, 35** *- optimal values*\n",
        "!wget https://data.deepai.org/deoldify/ColorizeVideo_gen.pth -O ./models/ColorizeVideo_gen.pth\n",
        "!rm -rf video\n",
        "!mkdir 'video'\n",
        "!mkdir 'video/source'\n",
        "!cp -r /content/video.mp4 video/source/video.mp4\n",
        "colorizer = get_video_colorizer()\n",
        "render_factor = 21  #@param {type: \"slider\", min: 5, max: 44}\n",
        "video_path = colorizer.colorize_from_file_name('video.mp4', render_factor)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "--2020-12-06 23:49:50--  https://data.deepai.org/deoldify/ColorizeVideo_gen.pth\n",
            "Resolving data.deepai.org (data.deepai.org)... 138.201.36.183\n",
            "Connecting to data.deepai.org (data.deepai.org)|138.201.36.183|:443... connected.\n",
            "HTTP request sent, awaiting response... 200 OK\n",
            "Length: 874066230 (834M) [application/octet-stream]\n",
            "Saving to: ‘./models/ColorizeVideo_gen.pth’\n",
            "\n",
            "els/ColorizeVideo_g   9%[>                   ]  77.68M  23.8MB/s    eta 33s    ^C\n",
            "cp: cannot stat '/content/video.mp4': No such file or directory\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-17-5e79954ccbb0>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msystem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"mkdir 'video/source'\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msystem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'cp -r /content/video.mp4 video/source/video.mp4'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m \u001b[0mcolorizer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_video_colorizer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      9\u001b[0m \u001b[0mrender_factor\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m21\u001b[0m  \u001b[0;31m#@param {type: \"slider\", min: 5, max: 44}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mvideo_path\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcolorizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolorize_from_file_name\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'video.mp4'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrender_factor\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'get_video_colorizer' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KN1RNS5Kekba"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-2UhHnWLRT8X"
      },
      "source": [
        "#@title ##**Get result** { display-mode: \"form\" }\n",
        "!cp -r video/result/video.mp4 /content/colorized_video.mp4\n",
        "what_next = 'play' #@param [\"play\", \"download\"]\n",
        "if what_next == \"play\":\n",
        "  display(mpy.ipython_display(\"/content/colorized_video.mp4\", height=400, autoplay=1, loop=1, maxduration=600))\n",
        "else:\n",
        "  files.download('/content/colorized_video.mp4')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bzQEN7FIRq4O"
      },
      "source": [
        "#@title ###**Optional block!** { display-mode: \"form\" }\n",
        "#@markdown *If it seems to you that the quality of coloring is insufficient, then run this block, wait until it is complete and look at what value of **render_factor** you like the picture more. Next, run the previous coloring block again, setting the desired value.*\n",
        "for i in range(10,45,2):\n",
        "    colorizer.vis.plot_transformed_image('video/bwframes/video/00001.jpg', render_factor=i, display_render_factor=True, figsize=(8,8))"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}