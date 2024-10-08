# AI Fashion Stylist for H&M

## Setup

### Run Time
* Open `AI Fashion Stylist.ipynb` in [Google Colab](https://colab.research.google.com/). Select Nvidia A100 GPU as the runtime backend (this requires the Colab Pro subscription). Restart the runtime, if at anypoint, we exhaused the GPU VRAM.

### H&M Dataset
* creates a `HM` directory under Google Drive 
* download the [articles.csv](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=articles.csv) and place it under the Google Drive `HM` directory, as: /content/drive/MyDrive/HM/articles.csv
* download the [images.zip](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=images), unzip and place it under the Google Drive `HM` directory, as: /content/drive/MyDrive/HM/images/
* `runs` the `H&M Dataset` section in `AI Fashion Stylist.ipynb. This will mount the Google Drive to the Colab environment's file system. Makes sure to allow the current Colab notebook to access your Google Drive, following the prompts (e.g., login to Google account etc). Once done, the output should be:
```
-rw-------  1 root root 36127865 Oct  5 20:02 articles.csv
drwx------ 88 root root     4096 Oct  6 13:38 images
```

### Dependency and Imports
* run the `Dependency and Import` section of the Colab to install all the libraries. It will also login to the HuggingFace. For this, makes sure you have a valid HF access token (e.g., for me it's `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`). The HF access token should include agreements to use the following models: [`Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), [PaliGemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224). Restart the runtime, if the import throws build errors. Note: after restarting the runtime, you need to return the following 3 sections: `H&M Dataset`, `Dependency and Imports`, and `Initialize Models`.


### Initialize Models
* run the `Initialize Model section` of the Colab. This should load LLM models from pre-trained checkpoints, process the H&M dataset (articles.csv), generate embeddings, and build a lightweight article query pipeline. At highlevel, there these LLMs for the Fashion Stylist system:
  - a frontend LLM (do fashion conversation with user) 
  - a retriever LLM (take frontend LLM's answer and retrive article id from the H&M database)
  - a multimodal VLM (process images given article id, and add additional descriptions)
  - a preference based PEFT component that can finetune the frontend LLM's behavior. We constantly collect user feedbacks to create such a preference dataset.
  - a T2I (text2image) diffusion models that is conditioned on brand (H&M) database to generate offline images. And also a VTO model to generate TryOn images on this user's uploaded photos.

![arch](./arch_v2.png)

### Frontend LLM

run the `Frontend LLM` section of the Colab. Adjust the `max_conversation_round` to the number of interactions you like. This should look like the following:
![frontend LLM2](https://github.com/tz-proactive/pail-aifs/blob/main/Frontend%20LLM2.png)

It uses `Mistral-7B-Instruct-v0.1` that takes all previous chat context (including a few warmup context), and generate answer for each new user's instruction. I added a warmup conversation as context, therefore the model will generate consistent text styling (e.g, inserting Emoji). This warm up history context can be viewed as a lightweight "PEFT". In the future, we can collect user feedbacks and apply more involved LoRA SFT or DPO.

### Retriever LLM
run the `Retriever LLM` section of the Colab. It uses `Mistral-7B-Instruct-v0.1` together with a `ConversationalRetrieval` pipeline. Adjust the `max_conversation_round` to the number of interactions you like. Adjust the `max_hm_clothing_articles_to_load` to the number clothings of the H&M database to use. The larger the number, the higher the article retrieval recall. The smaller the number the faster to compute the embedding for the database. The `use_front_end_llm` will combine the recommendation from the frontend LLM, as the retrievel conditioned model loses styling creativity. This should look like the following:
![retrieval LLM2](./retrieval4.png)
![retrieval LLM5](./retrieval5.png)

### Multi-modal (VLM)
run the `Multi-modal` section of the Colab. It uses a VLM (`google/paligemma-3b-mix-224`) to generate image caption/description. Then this info is concatenated to the prompt. Adjust the `image_size` to include more high-resolution information for the VLM model. Note: In a production system, we should use a more powerful VLM (e.g., `Pixtral 12B`, `Llama-3.2-vision-11B`).
![VLM](./VLM2.png)

### Event based Recommendation
The frontend LLM is already conditioning on the chat history context. However, the LLM model doesn't always utilize this information. To medigate this, I explicitly add a prompt engineering to make sure the model generate response using the user preference. The prompt engineering is `, based my preference`. This should look like the following:
![EVENT](./event-based.png) Note: In production, we should increase the database rows used by the Retrieval LLM, so we get more accurate product from H&M.

### Open-ended Multi-modal Recommendation
The Multi-modal (frontend LLM + retrieval LLM + VLM) is also already conditioning on the chat history context. The same prompt engineering `, based my preference` works:
![EVENT](./vlm-recommendation.png) ![EVENT](./vlm-recommendation2.png) Note: In a production system, we should use a more powerful VLM (e.g., `Pixtral 12B`, `Llama-3.2-vision-11B`). And increase the database rows used by the Retrieval LLM, so we get more accurate product from H&M.
 
### Style Matching & Complete Your Look
To add the ability to complete the look, we need two fashion outfit matching pipeline: one based on text, and one based on visual. They compliment each other, images are more accurate/closer to fashion designer's native expression. For the text branch, we need a crawler to read fashion blogs, and output matching outfit into a CSV (each row of the CSV is a text description of the outfit, separated into standardized: top, bottom, shoes, accessories column). For the visual branch, we need a Stable Diffusion T2I model, speciallized in Fashion outfit generation (e.g., LoRA finetuned on fashion outfits). Here are the added components to our exisiting Multi-modal system:
![Complete](./complete-look2.png)
![Complete](./complete.png)

### Virtual TryOn
Makese sure you have access and mounted the following directory: https://drive.google.com/drive/folders/13zXSlJlqI9xynbivGvIoV_s0Y4x3iCWT?usp=sharing. It includes the human models (e.g., Taylor Swift).
Runs `VTO Imports` and then `Virtual TryOn H&M`. Due to GPU RAM constraints, I only show this feature standalone, without integrate with the rest of the LLMs. In production, to improve quality, we need to use stronger garment and person parsers. See: [TryOnDiffusion](https://tryondiffusion.github.io/) for more details.   
![TryOn](./tryon.png)
