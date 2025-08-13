# Open-Set Image Tagging with Multi-Grained Text Supervision

Xinyu Huang1,2 $Yi-JieHuang2$  Youcai Zhang2Weiwei Tian4 $\text {RuiFeng}^{1,4}$ 

Yuejie Zhang1Yanchun $\mathbf {Xie}^{2}$  Yaqian $\mathbf {Li}^{2}$  Lei Zhang3

1Shanghai Key Lab of Intell. Info. Processing, School of Computer Science, Fudan University

2OPPO Research Institute3International Digital Economy Academy (IDEA)

4Academy for Engineering and Technology, Fudan University

## Abstract

In this paper, we introduce the Recognize Anything Plus Model $(RAM++)$ , an open-set image tagging model effec tively leveraging multi-grained text supervision. Previous approaches (e.g., CLIP) primarily utilize global text su pervision paired with images, leading to sub-optimal per formance in recognizing multiple individual semantic tags.In contrast, $RAM++$  seamlessly integrates individual tag supervision with global text supervision, all within a uni fed alignment framework.This integration not only en sures effcient recognition of predefned tag categories, but also enhances generalization capabilities for diverse open set categories. Furthermore, RAM++ employs large lan guage models (LLMs) to convert semantically constrained tag supervision into more expansive tag description super vision, thereby enriching the scope of open-set visual de scription concepts. Comprehensive evaluations on various image recognition benchmarks demonstrate RAM++ ex ceeds existing state-of-the-art (SOTA) open-set image tag ging models on most aspects. Specifcally, for predefned commonly used tag categories, RAM++ showcases 10.2mAP and 15.4 mAP enhancements over CLIP on OpenIm ages and ImageNet. For open-set categories beyond pre defned, RAM++ records improvements of 5.0 mAP and 6.4 mAP over CLIP and RAM respectively on OpenImages.For diverse human-object interaction phrases, RAM++achieves 7.8 mAP and 4.7 mAP improvements on the HICO benchmark.Code, datasets and pre-trained models are available at https://github.com/xinyu1205/recognize-anything.

## 1. Introduction

Image recognition remains a fundamental research area in computer vision, necessitating machines to output various semantic contents based on the given images. To this end,visual models with text supervision, such as CLIP [43],

<!-- (a) Visualization Example A dog sits on a  Text couch near a table. Tagdog, couch, table DescriptionDog is a four-legged  mammal with a tail … Couch is a cushioned  seat with armrests … -->
![](https://web-api.textin.com/ocr_image/external/86d66cea6cbf2740.jpg)

<!-- (b) Comparison of Multi-Grained Text Supervision Corgi is a small  siszheodr dt olegg wsi…th couch dog A dog sits  on a couch near a table.  Global Text Supervision Individual Tag Supervisiontable Tag Description Supervision Open-Set Inference -->
![](https://web-api.textin.com/ocr_image/external/192f5b2c369bedf9.jpg)

Figure 1.Illustration of multi-grained text supervision.(i)Global text supervision entangles multiple semantics, leading to sub-optimal performance in recognizing multiple individual se mantic tags. (ii) Our model leverages both individual tag super vision and global text supervision, enhancing tagging capacity on both predefned and open-set categories. (iii) We further convert tag supervision into more expansive tag description supervision via the LLMs, facilitating the recognition of diverse open-set cat egories with visual concepts.

ALIGN [22], and Florence [56], leverage large-scale image text pairs from the Internet to learn comprehensive vi sual concepts.These models demonstrate notable open set recognition in single-label image classifcation [10], fa cilitating their application across diverse domain-specifc datasets with arbitrary visual concepts [16, 49].

Despite such advances, these models predominantly rely on global text supervision, which directly align global

<!-- 100Tag-CommonTag-UncommonPhrase-HOI 9086.586.8 ) % 8076.671.472.470.268.875.4 ( 70   P m 60 A57.056.655.0 5048.4 4037.6 30 29.832.9 OpenImagesImageNetOpenImagesImageNetHICO SupervisedCLIPRAMRAM++ -->
![](https://web-api.textin.com/ocr_image/external/cfcf4c5b0115c1ae.jpg)

Figure 2. Comparison of zero-shot image recognition perfor mance on various benchmarks. Our RAM++ model outperforms existing SOTA open-set image tagging models (CLIP [43] and RAM [59]), in terms of common tag categories of OpenImages and ImageNet, uncommon tag categories of OpenImages and Im ageNet, and human-object interaction phrases of HICO.

text embeddings with corresponding global visual features.Such supervision is sub-optimal for more complex multi tag recognition tasks. Due to the global text supervision en tangles multiple semantics, the infuence of individual tag semantics is signifcantly weakened. As illustrated in Fig ure 1, the text “a dog sits on a touch near a table” encom passes the concepts of “dog”, “couch” and “table”. How ever, its global embedding exhibits partial divergence from these individual semantics.

By contrast, image tagging models with individual tag supervision, primarily utilize manually annotated image tags of limited scale [13, 28]. Despite recent studies [20,21, 59] signifcantly expand the scale of image tags us ing image-text pairs, image tagging models still fall short in recognizing tag categories beyond their predefned label system. This limitation highlights the constrained semantic generalization capabilities of tag supervision with fxed cat egories, consequently hindering their broader applicability.For instance, it is challenging to generalize the tag of “dog”or “drinks” to more specifc subcategories such as “corgi”or “Coca Cola”. Moreover, the numerous phrase categories like “meteor shower” further poses this challenge.

To address the aforementioned limitations, our study proposes an open-set image tagging model leveraging multi-grained text supervision, integrating both global text supervision and individual tag supervision. The image tags are automatically parsed from the texts, offering more fne grained supervision which ensures the competent recog nition on predefned tag categories.Simultaneously, the diverse text supervision enables the model to learn a broader range of textual semantics far beyond fxed tag categories, extending generalization capabilities for open set categories. Specifcally, we incorporate image-tag-text triplets within a unifed alignment framework. The multi-grained text supervision interacts with visual spatial fea tures through an effcient alignment decoder [51]. Com pared with other prevalent alignment paradigms, our ap proach demonstrates superior tagging performance with high effciency.

Furthermore, considering the insuffcient visual concepts of tag supervision, we convert tag supervision into more ex pansive tag description supervision through large language models (LLMs) [1, 37]. LLMs are employed to automati cally generate multiple visual descriptions for each tag cat egory. These descriptions are subsequently integrated into tag embedding via a novel automatic re-weighting mech anism, enhancing the relevance with corresponding image features. This approach enriches the scope of visual con cepts for the image tagging model, enhancing its capability to incorporate visual descriptions for open-set recognition during inference. For instance, the tag “corgi” can be ex panded to a more descriptive “a small-sized dog with short legs ...”, which aids in determining its presence in images.

Consequently,buildinguponourproposedap proaches, we introduce the Recognize Anything Plus Model (RAM++), an open-set image tagging model with an exceptional capability in recognizing diverse tag categories.As depicted in Figure 2, RAM++ exceeds existing SOTA open-set image tagging models (CLIP [43] and RAM [59])across various benchmarks. Notably, RAM++ showcases 10.2 mAP and 15.4 mAP enhancements over CLIP on predefned commonly used categories of OpenImages [25]and ImageNet [10]. Moreover, RAM++ also achieves 5.0mAP and 6.4 mAP improvements over CLIP and RAM on open-set uncommon categories of OpenImages. For diverse human-object interaction phrases, RAM++ achieves 7.8mAP and 4.7 mAP improvements on HICO [6] against CLIP and RAM, respectively.

Our key contributions can be summarized as follows:

• We integrate the image-tag-text triplets within a unifed alignment framework, achieving superior performance on predefned tag categories and augmenting recognition ca pabilities on open-set categories.

• To the best of our knowledge, our work is the frst ef fort to incorporate LLM’s knowledge into image tagging training stage, allowing the model to integrate visual de scription concepts for open-set category recognition dur ing inference.

 Evaluations on OpenImages, ImageNet, HICO bench marks demonstrate that RAM++ exceeds existing SOTA open-set image tagging models on most aspects. Com prehensive experiments provide evidence highlighting the effectiveness of multi-grained text supervision.

<!-- Exist…A coant a s sleoefpai.ng Parsecat, sofaExist…OOfnflliinnee Image-TextImage-TagsFrozen Alignment DecoderAlignment Decoder Alignment Decoder 2 x ……Feed Forward Cross-Attention text1 text2 text3 text4cat sofa dog person Label System TextImageTextdocagt, ,p seorfsao,n ,  EncoderEncoderEncodercar, flower… ChatGPT ◼A cat sleeping on a sofa.◼Cat is a small mammal with soft fur, a tail, sharp  ◼A dog next to flowers. claws, and ears that are pointed and upright. ◼Pepper the aussie pup.◼Sofa consists of a cushioned seat with a backrest  ◼…and armrests, which are comfortable to recline. ◼…… Image Batch TextsTag Descriptions -->
![](https://web-api.textin.com/ocr_image/external/2b0639f21cc3541a.jpg)

Figure 3. Illustration of RAM++ training framework. With image-tag-text triplets, RAM++ adopts a shared alignment decoder to align image-text and image-tags simultaneously. The individual tag supervision ensures effcient recognition of predefned tag categories, and the diverse text supervision signifcantly enhances the open-set tagging abilities. In addition, RAM++ employs a LLM to generate multiple visual descriptions for each category within the label system, thereby enriching the scope of open-set visual concepts.

## 2. Related Works

Tag Supervision. Image tagging, also known as multi-label recognition, involves assigning multiple tags to an image.Traditional methods primarily depend on limited manually annotated datasets [8, 13, 28], leading to poor generaliza tion capabilities. DualCoop [50] and MKT [17] employ pre trained vision-language models to boost open-set capabili ties, but they are constrained by the scale of training dataset.Tag2Text [21] and RAM [59] obtain large-scale image tags based on image-text pairs, demonstrating advanced zero shot capabilities on predefned categories. Nonetheless, all these models rely on tag supervision with closed-set seman tic scope, limiting their ability to recognize more diverse range of open-set tag categories. Our RAM++ seamlessly integrate diverse text supervision with tag supervision, ef fectively enhancing the open-set tagging abilities.

Text Supervision.Visual models with text supervi sion can recognize open-set categories by aligning visual linguistic features. Pioneering models like CLIP [43] and ALIGN [22], which collect millions of image-text pairs,demonstrate remarkable performance in single-label image classifcation [10]. However, their reliance on global text supervision present challenges in multi-tag tasks of indi vidual semantics [59]. Although other studies (e.g., AL BEF [26] and BLIP [27]) adopt deep visual-linguistic fea ture fusion, our analysis indicates their limitations of eff ciency and capacity in extensive-category tagging tasks. In contrast, RAM++ align multiple texts and individual tags within a unifed alignment framework, demonstrating supe rior tagging performance with high effciency.

Description Supervision.Several prior works demon strate the effectiveness of leveraging text-based category descriptions for enhancing image recognition performance.However, all these previous studies rely on external nat ural language databases such as handcraft [18, 19, 44],Wikipedia [12, 39] or WordNet [4, 14, 49, 54].With LLMs [3, 37] demonstrating powerful knowledge compres sion capabilities, recent works incorporate LLM’s knowl edge at the inference stage of CLIP to improve perfor mance [9, 29, 36, 41, 45] and interpretability [35]. Different from these approaches, our work pioneers the integration of LLM knowledge into the training process of image tagging,which is natural and effective to enhance the open-set capa bility of tagging models.

## 3. Approaches

### 3.1. Overview Framework

This section details RAM++, an open-set image tagging model capitalizes from multi-grained text supervision, en compassing both global text supervison and individual tag description supervison.As depicted in Figure 3, the ar chitecture of RAM++ comprises an image encoder, a text encoder, and an alignment decoder. The training data are image-tag-text triplets, comprising image-text pairs and im-age tags parsed from the texts. During the training process,the input into the model consists of images accompanied with variable batch texts and fxed tag descriptions. Then the model outputs alignment probability scores correspond ing to each image-tag/text pair, which are optimized by the alignment loss [46].

### 3.2. Multi-Grained Text Alignment

Unifed Image-Tag-Text Alignment Paradigm.With image-tag-text triplets, RAM++ adopts a shared alignment decoder to align image-text and image-tags simultaneously.Figure 3 splits the framework into two segments for clarity.The left segment illustrates the process of image-text align ment, where texts from the current training batch are passed through the text encoder to extract global text embeddings.These text embeddings are subsequently aligned with the image features via cross-attention layers in the alignment decoder, where text embedding serves as the Query, and image features as the Key & Value. Conversely, the right segment emphasizes the process of image tagging, where the image features interact with fxed tag categories using the same text encoder and alignment decoder.

The alignment decoder is a two-layer attention de coder [30, 51], each layer comprising a cross-attention layer and a feed-forward layer. This lightweight design ensures the effciency for image tagging involving extensive cate gories.Critically, it eliminates the mutual infuence be tween tag embeddings without self-attention layers, thus al lowing the model to recognize any quantity of tag categories without affecting performance.

Alignment Paradigm Comparison. In Figure 4, we com pare our Image-Tag-Text Alignment (ITTA) with other prevalent alignment paradigms:Image-Text Contrastive Learning (ITC) adopted by CLIP [43] and ALIGN [22],and Image-Text Matching (ITM) adopted by ALBEF [26]and BLIP [27]. On the one hand, ITC aligns the global fea tures of multiple images and texts simultaneously through dot product with high effciency. Nonetheless, its reliance on global text supervision with shallow interaction presents challenges for image tagging requiring localized recogni tion of multiple individual tags. On the other hand, ITM in volves in-depth visual-linguistic feature fusions with a deep alignment decoder. However, it only perform one single image-text pair, leading to signifcant computational costs when aligning the images with multiple texts or tags in both training and inference. Figure 6 demonstrates that both CLIP with ITC and BLIP with ITM fall short in image tag ging tasks with sub-optimal performance.

As such, our ITTA addresses these shortcomings by in corporating both global text supervision and individual tag supervision, ensuring robust tagging performance for both predefned and open-set categories. Additional, the adopted

<!-- Global  Spatial/Token Feature Feature Decoder x2 Single-ImageMulti-Texts/Tags ITTA -->
![](https://web-api.textin.com/ocr_image/external/11516e1bd789ebfa.jpg)

<!-- Decoder x x6 Single-ImageSingle-Text ITM -->
![](https://web-api.textin.com/ocr_image/external/d606e105cd342f36.jpg)

<!-- Dot Multi-ImagesMulti-Texts ITC -->
![](https://web-api.textin.com/ocr_image/external/4af1a57217e0a46a.jpg)

Figure 4.Comparison of different image-text alignment paradigms: Image-Text Contrastive Learning (ITC) adopted by CLIP [43], Image-Text Matching (ITM) adopted by BLIP [27]and Image-Tag-Text Alignment (ITTA). Our ITTA unifes image text alignment with image tagging framework, achieving a balance between effciency and performance.

effcient alignment decoder utilizes the image spatial fea ture instead of image global features, taking into account the fact that tags frequently correspond to various image regions. As a result, ITTA establishes a balance between performance and effciency, capable of aligning the images with thousands of tag categories with high effciency. For the comparison of inference times across different align ment paradigms, please refer to Figure 7.

### 3.3. LLM-Based Tag Description

Another innovative approach is LLM-based tag description,which involves leveraging the knowledge of the LLM to convert semantically constrained tag supervision into ex pansive semantic tag descriptions, thereby enriching the scope of open-set visual concepts that can be described.

LLM Prompt Design. To obtain descriptions for each tag category within the label system, prompt design for LLMs is essential. We anticipate that the tag descriptions gener ated by LLMs predominantly exhibit two characteristics: (i)as diverse as possible to cover a broader range of scenarios;(ii) as relevant as possible to image features for ensuring high relevance.

Drawing inspiration from [41], we design a total of fve LLM prompts for each tag category, as follows: (1) “De scribe concisely what a(n) {} looks like”; (2) “How can you identify a(n) {} concisely?”; (3) “What does a(n) {} look like concisely?”; (4) “What are the identifed characteris tics of a(n) {}”; (5) “Please provide a concise description of the visual characteristics of {}”.

Tag Description Generation. Based on the designed LLM prompts, we automatically generate descriptions for each tag category by calling the LLM API. Specifcally, we em ploy the “GPT-35-turbo” model [1], and set max token $s=$ 77 which is the same tokenizer length of the text encoder.To promote the diversity of the LLM responses, we set temperatur $e=0.$ $.99.$  Consequently, we acquire 10 unique responses for each LLM prompt, amassing a total of 50 tag descriptions per category. Comparison in Appendix E indi-

<!-- SpatialAlignment Decoder Feature … catsofa dog person Reweight Global… Feature Dot0.1 0.4 0.3 Image Text Encoder Encoder Cat a long tail, sharp claws …  Cat whiskered face, almond-shaped eyes … Cat a furry coat in a variety of colors … ··· Image Multiple Tag Descriptions -->
![](https://web-api.textin.com/ocr_image/external/c82c141adc8fc659.jpg)

Figure 5. Automatic re-weighting of multiple tag descriptions.

cates the superiority of the GPT-3.5 over GPT-3.

Automatic Re-weighting of Multiple Tag Descriptions.The multiple descriptions of each category requires to be integrated into one tag embedding for image tagging. A straightforward strategy is prompt ensemble, which aver ages multiple tag descriptions within the textual represen tation space. This strategy aligns with prevalent works of evaluating on open-set tagging model [41, 43]. However,the averaged embeddings can be sub-optimal for the train ing process, due to the ignorance of different similarities between the image and multiple candidate tag descriptions.

To enable selective learning from multiple candidate tag descriptions, we design an automatic re-weighting module for handling multiple tag descriptions, as illustrated in Fig ure 5. The probability scores for the i-th tag category are calculated as follows:

$$\text {Output}_{i}=\text {Decoder}\left[\left\{V_{1},\cdots ,V_{k}\right\},\sum _{j=1}^{50}\text {Softmax}\left(τ·g_{v}\left(V_{\text {global}}\right)·g_{w}\left(\mathbf {d}_{ij}\right)\right)·\mathbf {d}_{ij}\right]$$

(1)

Where Decoder represents the alignment decoder, $V_{\text {global}}$  refers to the image global features and $\left\{V_{1},\ldots ,V_{k}\right\}$ denotes the image spatial features. The term $\mathbf {d}_{ij}$  signifes the embedding of the $j-thtag$  description. The functions $g_{v}$ and $g_{w}$  are projector heads that map inputs into the same dimension, while $T$  is a learnable temperature parameter.

### 3.4. Online/Offine Design

Our approach also incorporates an online/offine design for different steps, ensuring seamless integration of the image text alignment and image tagging processes. In the con text of image tagging, the number of tag descriptions are fxed but of large volume (e $\text {g.,}4,500\text {tag}x50\text {des})$ . Al though extracting embeddings for all tag descriptions is time-consuming, the description embeddings can be pre processed offine using an off-the-shelf text encoder [43].In contrast, image-text alignment deals with variable text inputs, where the volume determined by batch size is rel atively modest.Therefore, text embeddings can be ex-


| Type  | Dataset  | #Images | #Categories |
| --- | --- | --- | --- |
| Tag-Common  | OpenImages  | 57,224  | 214  |
| Tag-Common  | ImageNet  | 5,000  | 492  |
| Tag-Uncommon  | OpenImages  | 21,991  | 200  |
| Tag-Uncommon  | ImageNet  | 5,000  | 508  |
| Phrase-HOI  | HICO  | 9,658  | 600  |


Table 1. The statistics of evaluation benchmarks.

tracted online for individual batches, circumventing sub stantial computational cost overhead.

## 4. Experiment

### 4.1. Experimental Settings

Training Datasets. We utilize the same training datasets as that employed by Tag2Text [21] and RAM [59]. The datasets are based on open-source image-text pair datasets and include two settings: a 4-million (4M) image dataset and a 14-million (14M) image dataset.The 4M setting comprises two human-annotated datasets (COCO [28] and VG [24]), as well as two web datasets (SBU Caption [38]and CC-3M [48]). The 14M setting extends the 4M by in corporating CC-12M [5]. Our label system includes 4,585categories that are commonly used in texts. For Tag2Text,the image tags are automatically extracted from their paired texts using a parser [52]. For RAM, both tags and texts are further augmented via an automated data engine [59]. We train RAM++ using the RAM datasets, and perform addi tional validations on the Tag2Text datasets in Appendix F,to substantiate the effectiveness of our proposed methods.

Implementation Details. We employ the $\text {Swin}_{\text {Base}}$  [32]pre-trained on ImageNet [10] as the image encoder, and select base-scale models across other comparative methods for fair comparison. We leverage the off-the-shelf text en coder from CLIP [43] to extract text and tag description embeddings. We adopt the robust alignment loss function of ASL [46] for both image-text alignment and image tag ging. The comparison of different alignment loss functions is available in Appendix G. Following [21, 26, 27, 59], our model further fne-tunes on the COCO dataset after pre trianing to augment its performance. Benefting from the fast convergence characteristic, the 4M and 14M versions of RAM++ necessitate only 1 and 3 days respectively for training, using 8 A100 GPUs.

Evaluation Benchmarks. We employ mean Average Pre cision (mAP) as the evaluation metric, which is well established for evaluating multi-tag recognition perfor mance [30, 46, 47, 59]. Additional metrics, including F1scores, precision, and recall, are provided in Appendix D.

We assess the image tagging capabilities on various out of-domain evaluation benchmarks. Specifcally, we utilize the widely used benchmarks OpenImages [25] and Ima-


| Methods  | Training #Images | Inference<br>Prompt  | Tag-Common  | Tag-Common  | Tag-Uncommon  | Tag-Uncommon  | Phrase-HOI HICO  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Methods  | Training #Images | Inference<br>Prompt  | OpenImages  | ImageNet-Multi  | OpenImages  | ImageNet-Multi  | Phrase-HOI HICO  |
| Closed-Set Models: | Closed-Set Models: |  |  |  |  |  |  |
| RelVit [34] | 4K  | - | ✗ | ✗ | ✗ | $\times$ | 39.4  |
| Swin [32]<br>ML-Decoder [47] | 1.3M  | - | $\times$ | 78.1  | ✗ | $79.0$ | $\times$ |
| Swin [32]<br>ML-Decoder [47] | 9M  | - | 85.8  | ✗ | 79.5  | ✗ | ✗ |
| Tag2Text [21] | 4M  | - | 82.9  | ✗ | ✗ | $\times$ | ✗ |
| Tag2Text [21] | 14M  | - | 83.4  | $\times$ | ✗ | ✗ | ✗ |
| Open-Set Models:MKT∗[17] | 162K  | Hand-Written  | 77.8  | 54.7  | 63.5  | 45.2  | 25.5  |
| BLIPIT C [27]<br>BLIPIT M [27] | 129M  | Hand-Written  | 75.7  | 56.2  | 61.1  | 36.4  | 33.5  |
| BLIPIT C [27]<br>BLIPIT M [27] | 129M  | Hand-Written  | 71.7  | 50.8  | 62.9  | 37.9  | 38.0  |
| DiHT [42] | 438M  | Hand-Written  | 71.3  | 67.7  | 62.4  | 66.8  | 36.7  |
| CLIP [43] | 400M  | Hand-Written  | 73.6  | 56.6  | 66.2  | 58.6  | 26.8  |
| CLIP [43] | 400M  | LLM Tag Des  | 76.6  | 57.0  | 70.2  | 56.6  | 29.8  |
| RAM∗[59] | 4M  | Hand-Written  | 86.0  | 70.2  | 66.7  | 47.3  | 32.8  |
| RAM∗[59] | 14M  | Hand-Written  | 86.5  | 71.4  | 68.8  | 48.4  | 32.9  |
| RAM∗[59] | 14M  | LLM Tag Des  | 82.2  | 62.8  | 65.9  | 43.2  | 29.6  |
| $RAM++*$ | 4M  | LLM Tag Des  | 86.5  | 71.6  | 73.9  | 51.3  | 37.8  |
| $RAM++*$ | 14M  | LLM Tag Des  | 86.6  | 72.4  | 75.4  | 55.0  | 37.7  |


Table 2. Zero-shot performance comparison of SOTA open-set image tagging models on mAP. Green refers to fully supervised learning with vertical domain training datasets. Inference prompt refers to the category prompt during model inference, $e.g.$ ., Hand-Written:“A photo of a cat”; LLM Tag Description: “Cat is a small general with soft fur ...”. $\text {BLIP}_{ITM}$  requires more than $1000\times$  inference time of CLIP and $RAM++$  in recognizing thousands of tag categories (see Figure 7).∗indicates the models leveraging the off-the-shelf CLIP. 

geNet [10]. Given that ImageNet is single-labeled and has missing labels in its test set [2, 57], we resort to ImageNet Multi [2], where each image in the test set possesses multi ple labels for a more comprehensive annotation. The cate gories of these benchmarks are categorized into “common”and “uncommon” categories based on the inclusion within the RAM++ label system.For more evaluations on the phrase categories, we resort to the HICO [6] benchmark,a prevalent standard on human object interactions (HOI).HICO encompasses 80 object categories, 177 action cate gories, resulting in a total of 600 “human-act-object” phrase combinations. The statistics of the evaluation benchmarks are presented in Table 1. It is worth noting that for RAM and $RAM++$ , apart from Tag-Common which are consid ered as predefned categories, all other benchmarks refer to unseen categories in an open-set confguration.

### 4.2. Comparison with State-of-the-Arts

Quantitative Results. Table 2 presents the $zero-shot$  per formance comparison between $RAM++$ and SOTA open-set image tagging models. On the one hand, text-supervised models such as BLIP and CLIP, exhibit sub-optimal perfor mance across both common and uncommon categories on multi-tag recognition. On the other hand, the tag-supervised model RAM notably boosts performance on common cate gories, but falls short on uncommon categories compared

†Zero-shot refers to the model does not utilize the training dataset of the corresponding vertical domain.

to CLIP. Moreover, the performance of CLIP can be signif icantly enhanced when utilizing the LLM tag descriptions for inference, which is consistent with the fndings of [41].Conversely, RAM does not beneft from LLM tag descrip tions, indicating its limited open-set generalization potential due to the constrained semantics of tag supervision.

Our RAM++ model, which utilizes both text supervi sion and tag description supervision, establishes a new SOTA zero-shot performance across various benchmarks.Specifcally, RAM++ outperforms CLIP by 10.0 mAP and 15.4 mAP on the common categories of OpenImages and ImageNet, respectively.In terms of open-set cate gories, $RAM++$  signifcantly outperforms RAM on both Tag-Uncommon and Phrase-HOI, underscoring the effec tiveness of our approach. Remarkably, $RAM++$  achieves an improvement of 6.6 mAP and 5.2 mAP over RAM and CLIP on OpenImages-uncommon, and 8.0 mAP and 4.9mAP over RAM and CLIP on HICO, respectively.

Despite RAM++ slightly behind CLIP on the uncommon categories of ImageNet, we attribute to that the 14M dataset scale of $RAM++$  is inadequate for covering these rare cate gories. It is noteworthy that the data expansion from 4M to 14M for RAM++ result in a 3.7 mAP performance improve ment on ImageNet-Uncommon. We contend that further scaling up the training dataset could potentiate the open-set recognition effcacy of $RAM++$ .

Distribution of Probability Scores. In Figure 6, we ana lyze the distribution of probability scores for positive and


| Text Supervision  | Text Supervision  | Tag Supervision $\text {Text+Des}$ Supervision  |
| --- | --- | --- |
| CLIP  | BLIP $ITM$ | RAM $RAM++$ |
| $多$<img src="https://web-api.textin.com/ocr_image/external/b7f4ba69354eb346.jpg"> | $\frac {2}{25}$Common <img src="https://web-api.textin.com/ocr_image/external/4625434359c41683.jpg"> | $0.$ $0.15$Tag Categories<br><img src="https://web-api.textin.com/ocr_image/external/dbd92c666f419fb9.jpg"><img src="https://web-api.textin.com/ocr_image/external/7021be5552f15434.jpg"> |
| 1.0<br>Positive<br>0.8Negative<br>$多$ ( <br>n<br>o<br>0.6<br>i<br>t<br>or<br>p<br>o<br>0.4<br>r<br>P<br>0.2<br>0.00.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32 0.36 0.4<br>Probability Score<br><img src="https://web-api.textin.com/ocr_image/external/4db0c36b42a47a5c.jpg"> | 1.0<br>Positive<br>0.8Negative<br>$多$x<br>n<br>o<br>0.6<br>i<br>t<br>or<br>p<br>o<br>0.4<br>r<br>P<br>0.2<br>0.00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.91<br>Probability Score<br>Uncommon <img src="https://web-api.textin.com/ocr_image/external/198346aa9cab69f2.jpg"> | 1.01.0<br>PositivePositive<br>) )<br>0.8Negative0.8Negative<br>x P<br>x<br>$25$n n<br>o o<br>0.6 ( 0.6<br>i i<br>t t<br>or0.4or0.4p<br>p<br>o o<br>r r<br>P0.2P0.2<br>0.00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.910.00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.91<br>Probability ScoreProbability Score<br><img src="https://web-api.textin.com/ocr_image/external/2f5d0ab3e1121f46.jpg"><img src="https://web-api.textin.com/ocr_image/external/34abe93c459b51a6.jpg"> |
| 1.0<br>Positive<br>0.8Negative<br>$多$ ( <br>n<br>o<br>0.6<br>i<br>t<br>or<br>p<br>o<br>0.4<br>r<br>P<br>0.2<br>0.00.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32 0.36 0.4<br>Probability Score<br><img src="https://web-api.textin.com/ocr_image/external/4db0c36b42a47a5c.jpg"> | 1.0<br>Positive<br>0.8Negative<br>$多$x<br>n<br>o<br>0.6<br>i<br>t<br>or<br>p<br>o<br>0.4<br>r<br>P<br>0.2<br>0.00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.91<br>Probability Score<br>Uncommon <img src="https://web-api.textin.com/ocr_image/external/198346aa9cab69f2.jpg"> |  Tag Categories  |


Figure 6. Distribution of probability scores for positive and negative tags on the OpenImages benchmark. On the one hand, text supervised models, such as CLIP and BLIP, exhibit challenges in predicting high probability scores for positive tags, leading to sub-optimal performance for multi-tag recognition. On the other hand, the tag-supervised model RAM falls short in recognizing open-set categories.As such, our $RAM++$ , which leverages both text and tag description supervision, demonstrates robust performance across both predefned common and open-set uncommon tag categories.


| Case  | Text<br>Supervision  | Tag<br>Supervision  | Tag Description Supervision  | Automatic Weighting  | Inference<br>Prompt  | Tag-Common  | Tag-Common  | Tag-Uncommon  | Tag-Uncommon  | Phrase-HOI HICO  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Case  | Text<br>Supervision  | Tag<br>Supervision  | Tag Description Supervision  | Automatic Weighting  | Inference<br>Prompt  | OpenImages  | ImageNet  | OpenImages  | ImageNet  | Phrase-HOI HICO  |
| (a)  | ✓ |  |  |  | Hand-Written  | 77.4  | 47.0  | 69.6  | 38.5  | 31.9  |
| (b)  |  | ✓ |  |  | Hand-Written  | 86.0  | 70.2  | 66.7  | 47.3  | 32.8  |
| (c)  | ✓ | ✓ |  |  | Hand-Written  | 86.5  | 71.5  | 70.5  | 49.9  | 35.5  |
| (d)  | ✓ | ✓ |  |  | LLM Tag Des  | 83.1  | 67.2  | 71.6  | 47.7  | 35.6  |
| (e)  | ✓ |  | ✓ |  | LLM Tag Des  | 86.5  | 71.3  | 73.4  | 50.8  | 37.2  |
| (f)  | ✓ |  | ✓ | ✓ | LLM Tag Des  | 86.6  | 71.6  | 73.9  | 51.3  | 37.8  |


Table 3. Ablation study of multi-grained text supervision on various image tagging benchmarks.

)

x x x P x

n n n n

o o o o

i

t

or

p p p p

o r r r

r P

P

) ) )

i

t t t

or

o o o

 

i

i

o

negative tags across various models on the OpenImages benchmark. An effective model should clearly distinguish between positive and negative tags. Notably, $RAM++$ , wtih dual supervision from texts and tag descriptions, demon strates robust performance on both predefned and open-set tag categories.

Besides, we acknowledge the value of investigating the reasons behind the score distributions of different alignment paradigms, which we leave as future work. As an illustra tion, we consider the contrastive loss in CLIP may leading to its scores around 0.2. And the suboptimal distribution of the ITM model can be attributed to the insuffcient utiliza tion of negative samples during training.

Quantitative results of prediction probability compari son between RAM and $RAM++$  are provided in Figure 8.The descriptions depicted in the fgure represent those with high weight in automatic re-weighting. RAM++ demon strates a signifcant improvement in prediction probabilities on open-set categories.

### 4.3. Analysis of Multi-Grained Supervision

Evaluation on Multi-Grained Text Supervision. We con duct a comprehensive ablation study in Table 3 to evalu ate the impact of multi-grained text supervision. Case (a)and (b) refer to the two segments of Figure 3, which lever age solely text supervision and tag supervision through the alignment decoder. Text supervision maintains consistent performance across various benchmarks, whereas tag super vision enhances outcomes in common categories.

Case (c) demonstrates the superiority of integrating image-text alignment with image tagging, signifcantly en hances the model’s capability to recognize open-set cate gories, evidenced by a 3.8 mAP and 2.7 mAP improvement on OpenImages-Uncommon and HICO. This approach, in contrast to the tag-supervised RAM model referenced in Ta ble 2, avoids a sharp decline in performance when utilizing LLM tag descriptions as the inference prompts, suggesting an enhanced semantic concepts by text supervision.

<!-- $10^{2}$ 89.76 CLIP with ITC BLIP with ITM 22.44 ) s $10^{1}$ $RAM++withITTA$ ( e  2.24 m  Ti e $10^{0}$ n c0.24 e r fe n $10^{-1}$ I $8$ .02 0170.0170.0180.0190.024 $10^{-2}$ 0.0070.0070.0070.0070.007 $10^{0}$ $10^{1}$ $10^{2}$ $10^{3}$ $4\times 10^{3}$ Num of Tag Categories -->
![](https://web-api.textin.com/ocr_image/external/39d18f2a794bfcf7.jpg)

Figure 7. Inference time comparison between different align ment paradigms for an image with the number of tag categories increasing.


| Image Feature  | Feature<br>Fusion  | OpenImages  | OpenImages  | HICO  |
| --- | --- | --- | --- | --- |
| Image Feature  | Feature<br>Fusion  | Common  | Uncommon  | HICO  |
| Global  | Dot Product  | 85.0  | 68.9  | 34.5  |
| Spatial  | Align Decoder  | 85.5  | 73.8  | 37.8  |


Table 4. Performance comparison of image features with dif ferent granularities.

Case (e) underscores the effectiveness of incorporating LLM tag descriptions in the training stage. When also em ploying tag descriptions for open-set categories evaluation,our model records the 2.9 and 1.7 mAP improvements on OpenImage-Uncommon and HICO. Such results indicates that expanding the semantically restricted tag supervision into a wide range of descriptive concepts during both train ing and inference stage, can substantially yield benefts for open-set tagging recognition.

Building on this foundation, case (f) reveals the auto matic re-weighting of multiple tag descriptions further en hance the model’s capabilities. In Section 4.3, we show case our re-weighting module achieves more signifcant im provements with more specifc and diverse tag descriptions.

Inference Time Comparison. Figure 7 presents the com parison of inference time consumption across three align ment paradigms with the number of tag categories increas ing. This comparison utilizes the average inference time calculated over 1,000 iterations, conducted on an A100GPU. The fgure obviously reveals that inference time for ITM models, which align with a single image-text pair, in creases exponentially with the augmentation of categories.This trend poses challenges for the model when process ing a large array of tag categories. In contrast, the ITC and ITTA models maintain high inference effciency, even with a large increase on tag categories. For instance, in the sce nario of recognizing 4,000 categories, the ITM model re quires 86.76 seconds, whereas the ITC and ITTA models necessitate only 0.024 seconds and 0.007 seconds.

Triceratops

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/c91ef9f7f4a9e831.jpg)

A triceratops is a large, herbivorous dinosaur with a unique appearance characterized by its three- horned face, a bony frill on its skull, and a bulky body.

Meteor Shower

<!-- RAM $RAM++$ -->
![](https://web-api.textin.com/ocr_image/external/2645c6f701f9ee97.jpg)

A meteor shower is characterized by numerous shooting stars or meteors that streak across the night sky. The meteors are usually brief, bright, and fast- moving,leaving a trail of light behind them.

Figure 8. Visual comparison of probability scores from RAM and $RAM++$  for open-set category recognition based on tag de scriptions. The descriptions are those assigned the highest weight by the $RAM++$  re-weighting module.


| Description Type  | Multiple<br>Description  | ImageNet- | ImageNet- |
| --- | --- | --- | --- |
| Description Type  | Multiple<br>Description  | Common  | Uncommon  |
| Basic  | Ensemble  | 65.3  | 46.0  |
| Basic  | Reweight  | 65.5  | 46.5  |
| Specifc  | Ensemble  | 60.1  | 25.7  |
| Specifc  | Reweight  | 62.7  | 31.9  |


Table 5. Performance comparison of different integrated meth ods for multiple tag descriptions.

Comparison of Image Features with different granular ities. Table 2 demonstrates that RAM++ with ITTA con sistently outperforms CLIP with ITC across various bench marks. To further compare image features of different gran ularity, we conduct the evaluation of image spatial features with the alignment decoder, against image global features with dot product, under the same training dataset compris ing image-tag-text triplets. As indicated in Table 4, im age spatial features consistently outperform global features,particularly on OpenImages-Uncommon and HICO bench marks of open-set categories. These results highlight the signifcance of our ITTA, seamlessly integrates image-text alignment and image tagging within the fne-grained align ment decoder framework.

More Specifc and Diverse Descriptions. We observe that the diversity of LLM descriptions, controlled by temper ature, is mainly limited to rephrasing rather than offering true semantic variety. To further validate the effectiveness of our proposed automatic re-weighting of multiple tag de scriptions, we attempt to employ more specifc and diverse tag descriptions. Specifcally, we design the LLM prompt of “Describe 50 different possible appearances of what a(n) {}looks like” to generate descriptions. Table 5 illustrates that our automatic re-weighting module achieves more signif cant improvements with more specifc and diverse tag de scriptions, due to the proposed freedom to selectively learn from mutually different texts. However, there is also a sig-nifcant decline on the quality of these descriptions, leading to much lower overall performance than the basic version.

## 5. Conclusion

This paper introduces RAM++, an open-set image tagging model with robust generalization capabilities. By leverag ing multi-grained text supervision, RAM++ achieves ex ceptional performance across various open-set categories.Comprehensive evaluations demonstrate that RAM++ ex ceeds existing SOTA models on most aspects. Given the revolution in natural language process by LLMs, RAM++highlights that integrating the knowledge of natural lan guage can signifcantly empower visual models. We hope our efforts can provide some inspiration for other works.

## References

[1] OpenAI. ChatGPT.https://openai.com/blog/chatgpt/, 2023. 2, 4

[2] Lucas Beyer, Olivier J H´enaff, Alexander Kolesnikov, Xi aohua Zhai, and A¨aron van den Oord. Are we done with imagenet? arXiv preprint arXiv:2006.07159, 2020. 6

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakan tan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan guage models are few-shot learners. Advances in neural in formation processing systems, 33:1877–1901, 2020. 3

[4] Sebastian Bujwid and Josephine Sullivan. Large-scale zero shot image classifcation from rich and diverse textual de scriptions. arXiv preprint arXiv:2103.09669, 2021. 3

[5] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre training to recognize long-tail visual concepts. In Proceed ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021. 5

[6] Yu-Wei Chao, Zhan Wang, Yugeng He, Jiaxuan Wang, and Jia Deng. Hico: A benchmark for recognizing human-object interactions in images. In Proceedings of the IEEE inter national conference on computer vision, pages 1017–1025,2015. 2, 6

[7] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy,Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter:Universal image-text representation learning. In European conference on computer vision, pages 104–120. Springer,2020. 12

[8] Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhip ing Luo, and Yantao Zheng. Nus-wide: a real-world web im age database from national university of singapore. In Pro ceedings of the ACM international conference on image and video retrieval, pages 1–9, 2009. 3

[9] Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, and Yongbin Li. Exploring large language models for multi-modal out of-distribution detection. arXiv preprint arXiv:2310.08027,2023. 3

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 1, 2, 3, 5, 6

[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.Bert:Pre-training of deep bidirectional transformers for language understanding.arXiv preprint arXiv:1810.04805, 2018. 13

[12] Mohamed Elhoseiny, Yizhe Zhu, Han Zhang, and Ahmed Elgammal. Link the head to the” beak”: Zero shot learning from noisy text description at part precision. In Proceed ings of the IEEE conference on computer vision and pattern recognition, pages 5640–5649, 2017. 3

[13] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christo pher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. In ternational journal of computer vision, 111:98–136, 2015. 2,3

[14] Christiane Fellbaum.WordNet:An electronic lexical database. MIT press, 1998. 3, 13

[15] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng,and Jingjing Liu. Large-scale adversarial training for vision and-language representation learning. Advances in Neural Information Processing Systems, 33:6616–6628, 2020. 12

[16] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui.Open-vocabulary object detection via vision and language knowledge distillation.In International Conference on Learning Representations, 2021. 1

[17] Sunan He, Taian Guo, Tao Dai, Ruizhi Qiao, Xiujun Shu, Bo Ren, and Shu-Tao Xia. Open-vocabulary multi-label classi fcation via multi-modal knowledge transfer. In Proceedings of the AAAI Conference on Artifcial Intelligence, pages 808–816, 2023. 3, 6, 12

[18] Xiangteng He and Yuxin Peng. Fine-grained image classif cation via combining vision and language. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5994–6002, 2017. 3

[19] Siteng Huang, Min Zhang, Yachen Kang, and Donglin Wang. Attributes-guided and pure-visual attention alignment for few-shot recognition. In Proceedings of the AAAI Con ference on Artifcial Intelligence, pages 7840–7847, 2021. 3

[20] Xinyu Huang, Youcai Zhang, Ying Cheng, Weiwei Tian,Ruiwei Zhao, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Xiaobo Zhang.Idea: Increasing text diversity via online multi-label recognition for vision-language pre training. In Proceedings of the 30th ACM International Con ference on Multimedia, pages 4573–4583, 2022. 2

[21] Xinyu Huang, Youcai Zhang, Jinyu Ma, Weiwei Tian, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Lei Zhang. Tag2text: Guiding vision-language model via im age tagging. arXiv preprint arXiv:2303.05657, 2023. 2, 3,5, 6

[22] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh,Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representa tion learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,2021. 1, 3, 4

[23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White head, Alexander C Berg, Wan-Yen Lo, et al. Segment any thing. arXiv preprint arXiv:2304.02643, 2023. 12

[24] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson,Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalan tidis, Li-Jia Li, David A Shamma, et al.Visual genome:Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision,123(1):32–73, 2017. 5

[25] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unifed image classifcation, object detection, and visual relationship detection at scale. Interna tional Journal of Computer Vision, 128(7):1956–1981, 2020.2, 5

[26] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare,Shafq Joty, Caiming Xiong, and Steven Chu Hong Hoi.Align before fuse: Vision and language representation learn ing with momentum distillation. Advances in neural infor mation processing systems, 34:9694–9705, 2021. 3, 4, 5

[27] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.Blip: Bootstrapping language-image pre-training for uni fed vision-language understanding and generation. In In ternational Conference on Machine Learning, pages 12888–12900. PMLR, 2022. 3, 4, 5, 6, 12

[28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755.Springer, 2014. 2, 3, 5, 12

[29] Jiaxiang Liu, Tianxiang Hu, Yan Zhang, Xiaotang Gai, Yang Feng, and Zuozhu Liu. A chatgpt aided explainable frame work for zero-shot medical image diagnosis. arXiv preprint arXiv:2307.01981, 2023. 3

[30] Shilong Liu, Lei Zhang, Xiao Yang, Hang Su, and Jun Zhu.Query2label: A simple transformer way to multi-label clas sifcation. arXiv preprint arXiv:2107.10834, 2021. 4, 5

[31] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection.arXiv preprint arXiv:2303.05499, 2023. 12

[32] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer:Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. 5, 6

[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 12

[34] Xiaojian Ma, Weili Nie, Zhiding Yu, Huaizu Jiang, Chaowei Xiao, Yuke Zhu, Song-Chun Zhu, and Anima Anandkumar.Relvit: Concept-guided vision transformer for visual rela tional reasoning. arXiv preprint arXiv:2204.11167, 2022. 6

[35] Sachit Menon and Carl Vondrick. Visual classifcation via description from large language models.arXiv preprint arXiv:2210.07183, 2022. 3

[36] Zachary Novack, Julian McAuley, Zachary Chase Lipton,and Saurabh Garg.Chils: Zero-shot image classifcation with hierarchical label sets. In International Conference on Machine Learning, pages 26342–26362. PMLR, 2023. 3

[37] OpenAI. GPT-4 technical report. https://arxiv.org/abs/2303.08774, 2023. 2, 3

[38] Vicente Ordonez,Girish Kulkarni,and Tamara Berg.Im2text: Describing images using 1 million captioned pho tographs. Advances in neural information processing sys tems, 24, 2011. 5

[39] Tzuf Paz-Argaman, Yuval Atzmon, Gal Chechik, and Reut Tsarfaty.Zest: Zero-shot learning from text descriptions using textual similarity and visual summarization.arXiv preprint arXiv:2010.03276, 2020. 3

[40] Bryan A Plummer, Liwei Wang, Chris M Cervantes,Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazeb nik. Flickr30k entities: Collecting region-to-phrase corre spondences for richer image-to-sentence models.In Pro ceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015. 12

[41] Sarah Pratt, Ian Covert, Rosanne Liu, and Ali Farhadi. What does a platypus look like? generating customized prompts for zero-shot image classifcation.In Proceedings of the IEEE/CVF International Conference on Computer Vision,pages 15691–15701, 2023. 3, 4, 5, 6

[42] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian,Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen,Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distil lation, and hard negatives for vision-language pre-training.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6967–6977, 2023. 6

[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervi sion. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 4, 5, 6, 12

[44] Scott Reed, Zeynep Akata, Honglak Lee, and Bernt Schiele.Learning deep representations of fne-grained visual descrip tions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 49–58, 2016. 3

[45] Zhiyuan Ren, Yiyang Su, and Xiaoming Liu.Chatgpt powered hierarchical comparisons for image classifcation.arXiv preprint arXiv:2311.00206, 2023. 3

[46] Tal Ridnik, Emanuel Ben-Baruch, Nadav Zamir, Asaf Noy,Itamar Friedman, Matan Protter, and Lihi Zelnik-Manor.Asymmetric loss for multi-label classifcation. In Proceed ings of the IEEE/CVF International Conference on Com puter Vision, pages 82–91, 2021. 4, 5, 13

[47] Tal Ridnik, Gilad Sharir, Avi Ben-Cohen, Emanuel Ben Baruch, and Asaf Noy. Ml-decoder: Scalable and versatile classifcation head. In Proceedings of the IEEE/CVF Win ter Conference on Applications of Computer Vision, pages 32–41, 2023. 5, 6

[48] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, im age alt-text dataset for automatic image captioning. In Pro ceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 5

[49] Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Zhe Gan, Lijuan Wang, Lu Yuan,Ce Liu, et al. K-lite: Learning transferable visual models with external knowledge. Advances in Neural Information Processing Systems, 35:15558–15573, 2022. 1, 3

[50] Ximeng Sun, Ping Hu, and Kate Saenko. Dualcoop: Fast adaptation to multi-label recognition with limited annota tions. Advances in Neural Information Processing Systems,35:30569–30582, 2022. 3

[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2, 4

[52] Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, and Wei-Ying Ma. Unifed visual-semantic embeddings: Bridging vision and language with structured meaning representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,pages 6609–6618, 2019. 5

[53] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiao long Wang, and Shalini De Mello. Open-vocabulary panop tic segmentation with text-to-image diffusion models.In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 2955–2966, 2023. 12

[54] Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu.Detclip: Dictionary-enriched visual-concept paralleled pre training for open-world detection. Advances in Neural Infor mation Processing Systems, 35:9125–9138, 2022. 3

[55] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu,and Haifeng Wang. Ernie-vil: Knowledge enhanced vision language representations through scene graphs. In Proceed ings of the AAAI Conference on Artifcial Intelligence, pages 3208–3216, 2021. 12

[56] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella,Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang,Boxin Li,Chunyuan Li,et al.Florence:A new foundation model for computer vision.arXiv preprint arXiv:2111.11432, 2021. 1

[57] Sangdoo Yun, Seong Joon Oh, Byeongho Heo, Dongyoon Han, Junsuk Choe, and Sanghyuk Chun. Re-labeling im agenet: from single to multi-labels, from global to local ized labels.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2340–2350, 2021. 6

[58] Youcai Zhang, Yuhao Cheng, Xinyu Huang, Fei Wen, Rui Feng, Yaqian Li, and Yandong Guo. Simple and robust loss design for multi-label learning with missing labels. arXiv preprint arXiv:2112.07368, 2021. 13

[59] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li,Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo,Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model.arXiv preprint arXiv:2306.03514,2023. 2, 3, 5, 6, 12

[60] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba.Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641,2017. 12

## A. More Implementation Details

Our models are uniformly pre-trained 5 epochs with a batch size of 720, followed by a fne-tuning process through an additional epoch on the higher-quality COCO dataset [28].The optimizer is the AdamW [33] with a weight decay of 0.05. During the pre-training stage, the input images are resized to $224\times 224.$  The learning rate is warmed-up to $1e^{-4}$ over the frst 3,000 iterations, and then follows linear decay with a rate of 0.9. In the fne-tuning stage, the input images size increase to $384\times 384$  and the learning rate is set to $5e^{-6}$  Following [17, 59], we employ the CLIP image encoder paired with the frozen text encoder to distill image feature, making full use of its original image text alignment properties.

## B. Comparison with Open-Set Localization Models

This section provides a comparative analysis between RAM++ and other SOTA open-set localization models (de tection [31] and segmentation [53]). The SAM [23] model is not included in the comparison due to its lack of recogni tion capabilitiesa. Table 6 illustrates the zero-shot recogni tion performance of different models on ADE20K [60] (in cluding 143 categories).Notably, RAM++ demonstrates signifcant advantages on both precision and recall metrics.

More importantly, the effciency of these localization models exhibits a highly correlation with the quantity of categories need to be recognized. Specifcally, they can ef fectively locate the corresponding objects when provided with the correct image tags. However, their recognition and localization performance markedly decline when provided with a large number of indeterminate categories.

In contrast, RAM++ maintains the robust recognition ability across thousands of categories with high accuracy.This distinctive capability enables $RAM++$  can signif cantly empower localization models to develop a strong vi sual semantic analysis pipeline.


| Methods  | ADE20k  | ADE20k  |
| --- | --- | --- |
| Methods  | Precision  | Recall  |
| Open-Set Detection Model: | Open-Set Detection Model: | Open-Set Detection Model: |
| Grounding-DINO [31] | 35.6  | 26.0  |
| Open-Set Segmentation ODISE [53] | Model:<br>48.2  | 50.3  |
| Open-Set Recognition Models: | Open-Set Recognition Models: | Open-Set Recognition Models: |
| CLIP [43] | 31.0  | 5.5  |
| RAM++ | 54.0  | 52.4  |


Table 6. Tagging performance comparison of $RAM++$  with other SOTA open-set localization models.

## C. Evaluation on Image-Text Retrieval

We extend our evaluation on image-text retrieval task to as sess the model’s alignment ability with fne-grained text.Specifcally, we focus on text-to-image retrieval perfor mance of Flickr30K [40], given its prominent application in practical scenarios. As depicted in Table 7, RAM sub stantially underperforms compared to CLIP, which further substantiate the limited generalization ability of RAM for open-set semantics. Our $RAM++$ , which employs the same dataset as RAM, even outperforms CLIP on both R@5 and R@10 metrics, demonstrating the effectiveness of our pro posed approaches.In addition, although BLIP achieves the best performance among zero-shot models, it relies on $\text {ITC+ITM}$ , resulting in a considerable inference time — re markably longer than both CLIP and $RAM++$  by several magnitudes.


| Methods  | Time/query(ms)  | Text-Retrieval (Flickr30K)  | Text-Retrieval (Flickr30K)  | Text-Retrieval (Flickr30K)  |
| --- | --- | --- | --- | --- |
| Methods  | Time/query(ms)  | R@1 | R@5 | R@10 |
| Fine-tuned Models: | Fine-tuned Models: | Fine-tuned Models: | Fine-tuned Models: | Fine-tuned Models: |
| UNITER [7] | - | 75.6  | 94.1  | 96.8  |
| ERNIE-ViL [55] | - | 76.7  | 93.6  | 96.4  |
| VILLA [15] | - | 76.3  | 94.2  | 96.8  |
| Zero-Shot Models: | Zero-Shot Models: | Zero-Shot Models: | Zero-Shot Models: | Zero-Shot Models: |
| CLIP [43] | ∼0.6 | 68.7  | 90.6  | 95.2  |
| RAM [59] | 3.1  | 45.9  | 75.9  | 84.6  |
| $RAM++(Ours)$ | ∼∼3.1 | 66.8  | 92.0  | 95.8  |
| BLIP [27] | ∼402.4 | 85.0  | 96.8  | 98.6  |


Table 7. Text to image retrieval performance comparison.

## D. Additional Evaluation Metrics

In Table 8, we present additional evaluation metric results,including F1 score, precision and recall. We manually ad just the threshold of different models to ensure compara bility across evaluations. The results demonstrate that our $RAM++$  exceeds other open-set image tagging models in both predefned and open-set categories, further highlights the robust tagging capabilities of $RAM++$ .


| Methods  | OpenImages-CommonOpenImages-Uncommon<br>F1PrecisionRecallF1PrecisionRecall  | OpenImages-CommonOpenImages-Uncommon<br>F1PrecisionRecallF1PrecisionRecall  |
| --- | --- | --- |
| BLIP<br>CLIP<br>RAM<br>$RAM++$ | 64.878.655.1<br>63.077.952.9<br>77.679.575.9<br>77.679.975.4  | 53.954.753.1<br>63.855.873.7<br>54.053.854.3<br>64.856.376.2  |


Table 8. Zero-shot performance comparison with SOTA open-set image tagging models in various metrics.

## E. GPT3 vs. GPT3.5.

In Table 9, we compare the performance impact of using different LLMs to generate tag descriptions for $RAM++$ 

(LLM with consistent training and testing).Evaluation results suggest that GPT-3.5 offers superior performance compared to GPT-3, due to its enhanced accuracy and di versity in responses.

In addition to the LLMs, we also attempt to utilize Word Net descriptions [14]. However, their contribution to per formance was minimal, due to WordNet only provides one description or even no description for each category.


| LLM  | Tag-Uncommon  | Tag-Uncommon  |
| --- | --- | --- |
| LLM  | OpenImages  | ImageNet  |
| GPT-3  | 72.9  | 55.4  |
| GPT-3.5  | 73.8  | 55.5  |


Table 9. Performance comparison of different LLMs applied in RAM++.

## F. Validation on Different Training Datasets

We further validate our approaches on the 4M training dataset of Tag2Text. Tag2Text fully utilizes open-source image-text pairs.RAM further augments both tags and texts via an automated data engine. As shown in Table 10,RAM++ demonstrates notable improvements across various benchmarks on both training datasets, highlighting the eff cacy of our approaches.


| Training Dataset  | Method  | Tag-Common<br>OpenImages  | Tag-Uncommon OpenImages  | Phrase-HOI HICO  |
| --- | --- | --- | --- | --- |
| Image-Text Pairs  | Tag2Text  | 82.9  | ✗ | ✗ |
| Image-Text Pairs  | RAM  | 83.1  | 63.2  | 28.4  |
| Image-Text Pairs  | RAM++ | 83.5  | 70.4  | 35.6  |
| Image-Text Pairs<br>+Data Engine  | RAM  | 86.0  | 66.7  | 32.8  |
| Image-Text Pairs<br>+Data Engine  | RAM++ | 86.5  | 73.9  | 37.8  |


Table 10. Approaches validation on different training datasets.

## G. Alignment Loss Function Comparison

Image-Text Alignment Loss Function. In Table 11 and Table 12, we compare different alignment loss functions for image-text alignment and image tagging, including the Cross Entropy (CE) function employed by CLIP, and other robust tagging loss functions (BCE, ASL [46], Hill [58],SPLC [58]).The results indicate that ASL outperforms other loss functions, which alleviates the potential missing labels and imbalance between positive and negative sam ples.

## H. Model Architecture Comparison

Off-The-Shelf Text Encoder. In this section, we explore the impact of different off-the-shelf text encoders, includ ing pre-trained BERT [11] and CLIP text encoder. Table 13showcases that the text/tag embedding extracted by CLIP


| ITA Loss  | OpenImages- | OpenImages- |
| --- | --- | --- |
| ITA Loss  | Common  | Uncommon  |
| BCE  | 81.1  | 65.4  |
| CE  | 83.1  | 67.7  |
| Hill  | 82.7  | 69.2  |
| ASL  | 83.2  | 70.2  |


Table 11. Performance comparison of different alignment loss functions for image-text alignment.


| Tagging Loss  | OpenImages- | OpenImages- |
| --- | --- | --- |
| Tagging Loss  | Common  | Uncommon  |
| Hill  | 79.6  | 67.7  |
| SPLC  | 82.0  | 66.3  |
| ASL  | 83.2  | 70.2  |


Table 12. Performance comparison of different alignment loss functions for image tagging.

text encoder is much better than that extracted by BERT.This suggest the image aligned text features can effectively enhance the ability of image text alignment models, espe cially when the text encoder remains frozen.


| Text<br>Encoder  | ImageNet- | ImageNet- |
| --- | --- | --- |
| Text<br>Encoder  | Common  | Uncommon  |
| BERT  | 57.9  | 24.2  |
| CLIP  | 63.6  | 44.6  |


Table 13. Performance comparison of different off-the-shelf text encoders.

Larger Image Encoder. Table 14 presents the performance comparison of image encoders with different scales. While SwinLarge exhibits improvements on predefned categories,it reveals a decrease on performance for open-set categories.


| Image<br>Encoder  | Tag-Common  | Tag-Common  | Tag-Uncommon  | Tag-Uncommon  | Phrase HICO  |
| --- | --- | --- | --- | --- | --- |
| Image<br>Encoder  | Openimages  | ImageNet  | Openimages  | ImageNet  | Phrase HICO  |
| Swin-B  | 86.6  | 72.4  | 75.4  | 55.0  | 37.7  |
| Swin-L  | 86.4  | 74.0  | 75.0  | 53.4  | 39.2  |


Table 14. Performance comparison of different image encoder.

Depth of Alignment Decoder. Table 15 demonstrates that increasing the layer depth of the alignment decoder does not necessarily enhance the model’s recognition capabilities, al lowing ITA to achieve superior performance with minimal computational consumption.


| Decoder Depth  | OpenImages- | OpenImages- |
| --- | --- | --- |
| Decoder Depth  | Common  | Uncommon  |
| 2  | 82.4  | 61.7  |
| 6  | 80.2  | 58.5  |


Table 15. Performance comparison of different layer depth for alignment decoder.

Wire Fox Terrier


![](https://web-api.textin.com/ocr_image/external/71f401d617dd987f.jpg)

The Wire Fox Terrier has a distinctive rough and wiry coat. Their head is flat with a long muzzle and dark, oval-shaped eyes. They have pointed ears that are usually folded over.

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/1bc24534014a56d6.jpg)

Cockatoo


![](https://web-api.textin.com/ocr_image/external/66edd9c37c8b3e82.jpg)

A sulphur-crested cockatoo is a large, white cockatoo bird with a distinctive yellow crest on its head. It has a curved beak, black feet, and dark eyes.

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/54f1ada069ce578c.jpg)

Trampolining


![](https://web-api.textin.com/ocr_image/external/b423ff2030496390.jpg)

Trampolining involves individuals jumping and performing acrobatic movements on a trampoline. Participants use the trampoline's bouncing effect to propel themselves higher in the air, performing various flips, twists, and other tricks.

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/8c602e61ef91ee62.jpg)

### Kite (bird of prey)


![](https://web-api.textin.com/ocr_image/external/80829f2df7058230.jpg)

A kite is a medium-sized bird of prey with a long wingspan, slender body, and a forked tail. It has a distinctive shape in flight, with long, narrow wings and a buoyant and graceful flying style. 

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/b2f3fc7f114e9b72.jpg)

Domestic Rabbit

<!-- RAM -->
![](https://web-api.textin.com/ocr_image/external/014a73ae78025308.jpg)

Domestic rabbits typically have small,round bodies, short tails and long legs. They have large, expressive eyes and long ears that can be upright or droopy. 

RAM++

Sledding

<!-- RAM RAM++ -->
![](https://web-api.textin.com/ocr_image/external/7f25e29b9354bbf6.jpg)

Sledding is a recreational activity typically done in winter where individuals slide down a slope using a sled. The visual characteristics of sledding include snowy landscapes, people wearing winter clothing, and joyful expressions.

Figure 9. More visual comparison of model prediction probabilities between RAM and RAM++ for open-set category recognition.RAM++ effectively utilizes visual features derived from the descriptions, demonstrating a signifcant improvement on prediction probabil ities.

## I. Additional Qualitative Results

In Figure 9, we show more examples that RAM++ presents better robustness on open-set categories against RAM, by utilizing visual features derived from the tag descriptions.

## J. Evaluation Benchmark Details

In Figure 10, we present the word clouds of the categories in various evaluation benchmarks. The word size is pro portional to the category frequency. This visualization re veals that uncommon categories not included in the prede fned labeling systems are not necessarily rare categories.Instead, most of them are well-recognized and commonly understood categories.


![](https://web-api.textin.com/ocr_image/external/1514fcc81c62b0ea.jpg)


![](https://web-api.textin.com/ocr_image/external/b4bb8cc4774b9ade.jpg)

(a) OpenImages-Common (b)OpenImages-Uncommon


![](https://web-api.textin.com/ocr_image/external/2fb6315b2dfff431.jpg)


![](https://web-api.textin.com/ocr_image/external/1fe500faf48f4893.jpg)

(c) ImageNet-Common (d) ImageNet-Uncommon

Figure 10. Illustration of the categories in various evaluation benchmarks.

