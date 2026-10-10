# Dual-Path Brain Tumor Understanding: Segmentation + Text Generation

A PyTorch research notebook that explores two complementary tasks on
FLAIR brain MRI data from **BraTS 2020**, paired with radiology-style
text from **TextBraTS 2020**:

-   **Path A --- Text-guided tumor segmentation:** predicts a binary
    tumor mask from an MRI slice, conditioned on a precomputed text
    embedding.
-   **Path B --- Image-to-text generation:** generates a text
    description from an MRI slice.
-   **Cross-path analysis:** investigates whether per-sample
    segmentation quality (Dice) is associated with text-generation
    quality (ROUGE-L).


## Pipeline overview

``` text
FLAIR MRI volume + segmentation volume + associated text/report
                         |
                 Slice preparation
                         |
           +-------------+--------------+
           |                            |
           v                            v
   PATH A: SEGMENTATION          PATH B: TEXT GENERATION
   MRI + 768-D text embedding    MRI slice
           |                            |
   ResNet34 U-Net backbone       ResNet34 image encoder
   + text-feature fusion         + projection + LSTM decoder
           |                            |
   Binary tumor mask             Generated text description
           |                            |
   Dice, IoU, precision,          ROUGE-1/2/L, BLEU-4
   recall, Hausdorff distance              |
           +---------------+--------------+
                           |
                 Spearman correlation
                   (Dice vs. ROUGE-L)
```

The notebook builds slice-level examples from the available MRI and mask
volumes. It converts the multi-region mask into a binary foreground mask
by treating any non-background tumor channel as tumor, and keeps slices
with at least `MIN_TUMOR_PIXELS` foreground pixels.

## Models

### Path A: Text-guided segmentation

-   **Backbone:** U-Net with an ImageNet-pretrained ResNet34 encoder,
    implemented with `segmentation-models-pytorch`.
-   **Input image:** a FLAIR slice converted to three channels by
    repeating the grayscale image.
-   **Text input:** a precomputed 768-dimensional embedding loaded from
    a `.npy` file and averaged across its sequence dimension.
-   **Fusion:** the embedding is projected through fully connected
    layers to 256 dimensions, broadcast spatially, concatenated with the
    U-Net's one-channel segmentation output, and passed through
    convolutional layers to produce the final mask logits.
-   **Loss:** a weighted combination of binary cross-entropy with logits
    and Dice loss.
-   **Optimization:** AdamW, a `ReduceLROnPlateau` scheduler, mixed
    precision, best-validation-Dice checkpointing, and early stopping.

### Path B: image-to-text generation

-   **Encoder:** ResNet34 encoder initialized from ImageNet; matching
    encoder weights are transferred from the trained segmentation model.
-   **Feature projection:** global average pooling followed by a learned
    linear projection.
-   **Decoder:** an LSTMCell-based autoregressive decoder that predicts
    tokens one at a time.
-   **Tokenizer:** Hugging Face GPT-2 tokenizer; the notebook uses the
    tokenizer's EOS token as the padding token and uses its BOS/EOS
    token IDs for sequence handling.
-   **Training objective:** token-level cross-entropy with teacher
    forcing, ignoring the configured padding token.
-   **Generation:** greedy decoding up to the configured maximum
    sequence length.

## Datasets and expected layout

The notebook uses Kaggle-style paths by default. You need both the
preprocessed FLAIR image/mask dataset and the TextBraTS text/embedding
data.

The image split is expected to resemble:

``` text
FLAIR_BRATS2020_split/
├── train/
│   ├── images/
│   │   └── image_<id>.npy
│   └── masks/
│       └── mask_<id>.npy
└── val/
    ├── images/
    └── masks/
```

The text data is expected to resemble:

``` text
TextBraTSData/
└── BraTS20_Training_<id>/
    ├── BraTS20_Training_<id>_flair_text.npy
    └── BraTS20_Training_<id>_flair_text.txt
```

The notebook derives a patient identifier from the numeric image
filename and expects the corresponding embedding file. Examples without
a matching mask or text embedding are skipped. Text files are used as
reference descriptions for evaluation; if a text file is missing, the
notebook falls back to an empty string.

**Dataset preparation matters:** the notebook expects `.npy` volumes and
masks already arranged in this layout. It does not download or convert
the raw BraTS datasets automatically. Confirm that image IDs, mask IDs,
patient IDs, and text records match your data before training.

## Requirements

The notebook installs some packages in its first code cell:

``` bash
pip install segmentation-models-pytorch scikit-learn rouge_score nltk transformers
```

It also imports PyTorch, torchvision, NumPy, Matplotlib, and SciPy.
Install compatible versions of these packages for your environment. A
CUDA-capable GPU is recommended for practical training, although the
code selects CPU if CUDA is unavailable.

## Running the notebook

1.  Open the notebook in Jupyter or Kaggle.
2.  Make the BraTS image/mask and TextBraTS datasets available in the
    expected directories.
3.  Update the paths in the `Config` class to match your environment.
4.  Ensure the tokenizer and pretrained ResNet34 encoder weights can be
    loaded. The first run may require internet access or cached model
    files.
5.  Run the notebook cells in order.
6.  Inspect the training curves, qualitative predictions, generated
    descriptions, validation metrics, and correlation output.

The notebook's configuration currently specifies 30 segmentation epochs
and 20 captioning epochs, with a batch size of 32; early stopping can
end training sooner. Training can take substantial time and memory
depending on the dataset and hardware.

## Evaluation

### Segmentation

The notebook reports these metrics against the binary ground-truth mask:

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  Dice coefficient                    Overlap between predicted and
                                      ground-truth tumor regions

  IoU                                 Intersection divided by union

  Precision                           Fraction of predicted tumor pixels
                                      that are true positives

  Recall                              Fraction of ground-truth tumor
                                      pixels recovered

  Hausdorff distance                  Worst-case directed
                                      boundary/point-set distance,
                                      combined symmetrically
  -----------------------------------------------------------------------

It also visualizes the MRI slice, ground-truth mask, predicted mask, and
an overlay. The ablation compares validation predictions using the
supplied text embedding with predictions using a zeroed embedding.

### Text generation

Generated descriptions are compared with the available reference text
using:

-   **ROUGE-1:** unigram overlap.
-   **ROUGE-2:** bigram overlap.
-   **ROUGE-L:** longest-common-subsequence-based overlap.
-   **BLEU-4:** n-gram precision-based corpus score up to four-grams.

These metrics measure text overlap, not clinical correctness or factual
grounding.

### Cross-path correlation

For each validation sample, the notebook pairs segmentation Dice with
generated-text ROUGE-L and calculates Spearman's rank correlation
coefficient and its p-value. This tests whether the two metrics tend to
move together; correlation alone does **not** establish that both
pathways understand the tumor or that the generated text is medically
accurate.

## Configuration

Important parameters are defined in the notebook's `Config` class:

  ------------------------------------------------------------------------
  Parameter                                  Default Purpose
  --------------------- ---------------------------- ---------------------
  `IMG_SIZE`                                   `128` Intended image size
                                                     setting; verify where
                                                     it is applied in
                                                     preprocessing

  `TEXT_DIM`                                   `768` Precomputed
                                                     text-embedding
                                                     dimension

  `EPOCHS`                                      `30` Maximum segmentation
                                                     epochs

  `CAP_EPOCHS`                                  `20` Maximum captioning
                                                     epochs

  `BATCH_SIZE`                                  `32` DataLoader batch size

  `LR`                                        `1e-4` Learning rate

  `WEIGHT_DECAY`                              `1e-4` AdamW weight decay

  `PATIENCE`                                     `7` Early-stopping
                                                     patience

  `MIN_TUMOR_PIXELS`                            `50` Minimum tumor pixels
                                                     required for a slice

  `MAX_SEQ_LEN`                                 `64` Maximum tokenized
                                                     description length

  `LOAD_CHECKPOINT`                          `False` Whether to load
                                                     existing checkpoints
                                                     where supported
  ------------------------------------------------------------------------

## Implementation notes and limitations

-   **Check the vocabulary-size setting:** the notebook loads the GPT-2
    tokenizer and derives `VOCAB_SIZE` from it, while the `Config` class
    also declares `VOCAB_SIZE = 3000`. Ensure the model is instantiated
    with the tokenizer's actual vocabulary size consistently; otherwise,
    token IDs and the decoder output dimension may not match.
-   **Review padding IDs:** the caption decoder's embedding layer has a
    default `padding_idx=0`, while the notebook assigns GPT-2's EOS
    token ID as `PAD_IDX`. Align these settings if you modify the
    implementation.
-   **Attention is not fully implemented in the current decoder:** an
    attention layer is declared, but the forward/generation logic feeds
    the same projected global image feature at each time step. Do not
    describe this as spatial attention unless you implement and use
    attention over spatial encoder features.
-   **Verify the data split:** the shown configuration includes training
    and validation directories. The final evaluation functions shown in
    the notebook evaluate on the validation loader; a separate held-out
    test set is not defined in the documented workflow.
-   **Check slice-level leakage:** multiple slices can come from the
    same patient. Ensure patient-level separation between training and
    validation sets to avoid leakage if the supplied split was not
    already created at patient level.
-   **Text association is patient-level in the dataset code:** a
    patient's text embedding and report are reused for each retained
    slice from that patient. Interpret results with this in mind.
-   **Not a diagnostic tool:** generated text and predicted masks
    require expert review. Do not use outputs for clinical decisions
    without rigorous external validation.

