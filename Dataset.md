## Dataset

This project uses the **MultiWOZ 2.2** dataset for training and evaluation.

To reproduce the experiments, create a Kaggle account, generate your personal **`kaggle.json`** API token, and place it in:

```text
/root/.kaggle/kaggle.json
```

when using Google Colab (or `~/.kaggle/kaggle.json` on a local machine).

Then run the notebook to automatically download, extract, and preprocess the dataset.

> **Note:** The `kaggle.json` file contains your personal Kaggle API credentials and is **not included** in this repository. Please generate your own API token from your Kaggle account. The dataset remains the property of its original authors; refer to the official MultiWOZ documentation and licensing information for usage terms.
