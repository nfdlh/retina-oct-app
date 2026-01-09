Contents lists available at ScienceDirect
Computer Methods and Programs in Biomedicine
journal homepage: https://www.sciencedirect.com/journal/computer-methods-and-
programs-in-biomedicine
From segmentation to explanation: Generating textual reports from MRI with
LLMs
Alberto G. Valerio ∗, Katya Trufanova , Salvatore de Benedictis, Gennaro Vessio 1,
Giovanna Castellano 1
Department of Computer Science, University of Bari Aldo Moro, Bari, Italy
A R T I C L E I N F O
Keywords:
Explainability
Large language models
Medical imaging
Report generation
Semantic segmentation
A B S T R A C T
Background and Objective: Artificial Intelligence (AI) has significantly advanced medical imaging, yet the
opacity of deep learning models remains challenging, often reducing the trust of medical professionals towards
AI-driven diagnoses. As a result, there is a strong focus on making AI models more transparent and interpretable
to boost healthcare providers’ confidence in these technologies.
Methods: This paper introduces a novel approach to enhance AI explainability in critical medical tasks
by integrating state-of-the-art semantic segmentation models with atlas-based mapping and Large Language
Models (LLMs) to produce comprehensive, human-readable medical reports. The proposed framework ensures
that the generated outputs are factual and contextually rich. Our anti-hallucination design, which combines
structured JSON with prompt constraints, is a critical innovation compared to most naïve LLM report
generation methods, thereby enhancing the transparency and interpretability of AI systems.
Results: Experimental results show that the SegResNet model achieves high segmentation accuracy, while
LLMs (Gemma, Llama, and Mistral) demonstrate diverse strengths in generating explanatory reports. Numerous
metrics have been employed to assess the quality and effectiveness of generated textual explanations, such as
lexical diversity, readability, coherence, and information coverage.
Conclusions: The method has been specifically tested for brain tumor detection in glioma, one of the most
aggressive forms of cancer, and subsequently applied to multiple sclerosis lesion detection to further validate
its generalizability across various medical imaging scenarios, thereby contributing to the trustworthiness of
healthcare AI applications.
Reproducibility: The complete source code for implementing the framework and reproducing the results is
publicly available, along with full pipeline examples demonstrating each step – from segmentation to report
generation – at the following repository: https://github.com/albertovalerio/from-segmentation-to-explanation.
1. Introduction
Artificial Intelligence (AI) is playing an increasingly prominent role
in healthcare, particularly in medical imaging, where deep learning
models have achieved remarkable performance in tasks such as disease
detection, classification, and segmentation. However, despite these ad-
vances, the opacity of such models remains a critical barrier to clinical
adoption. The inability to understand how and why a model arrives
at a particular decision raises concerns among healthcare profession-
als, especially when diagnostic decisions can directly affect patient
outcomes [1,2].
Explainable AI (XAI) has thus emerged as a crucial research area,
aiming to enhance transparency and foster trust in AI-assisted diagnos-
tics. In medical imaging, visual explanation methods like Grad-CAM [3]
∗ Corresponding author.
E-mail address: a.valerio31@phd.uniba.it (A.G. Valerio).
1 Vessio/Castellano: equal last author contribution
have been widely used to highlight relevant regions in the input data.
Nevertheless, these methods often provide limited insight: heatmaps
can be ambiguous, emphasize clinically irrelevant features, or fail to
articulate the rationale behind a decision in a manner that is human-
understandable [4]. As a result, they may not align with the interpretive
practices of medical experts.
Recent developments in Large Language Models (LLMs) offer new
opportunities for generating textual explanations that resemble clinical
narratives. Such models have demonstrated the ability to produce
coherent and contextually rich descriptions. However, their outputs can
suffer from factual inaccuracies, especially when prompts are poorly
formulated or when the model lacks grounding in domain-specific
data [5]. This limitation is particularly problematic in the medical
https://doi.org/10.1016/j.cmpb.2025.108922
Received 28 October 2024; Received in revised form 30 May 2025; Accepted 17 June 2025
Computer Methods and Programs in Biomedicine 270 (2025) 108922
Available online 1 July 2025
0169-2607/© 2025 The Authors. Published by Elsevier B.V. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).
A.G. Valerio et al.
domain, where incorrect or hallucinated information can undermine
trust and compromise safety.
While visual and textual explainability have often been treated as
separate modalities, their integration holds the key to more compre-
hensive and clinically useful AI explanations. Visual cues can guide
attention to relevant image regions, while text can convey diagnostic
reasoning and contextual interpretations. When appropriately com-
bined, these two modalities can complement each other, reflecting the
way clinicians interpret and report findings.
This paper presents a novel framework that integrates visual expla-
nations and LLMs to generate clinically interpretable reports from 3D
brain MRI segmentations automatically. Unlike traditional approaches
that rely on pre-existing text corpora or external knowledge bases, our
methodology generates explanations from scratch, grounded directly in
the model’s outputs and anatomical context.
The framework is applied primarily to the detection of glioma
tumors in adult brain MRIs, and its generalizability is further evaluated
on a separate dataset focused on multiple sclerosis lesion segmentation.
The process begins with the application of state-of-the-art deep learning
models for the semantic segmentation of brain anomalies. The resulting
segmentation maps are then enriched with anatomical context through
a mapping module based on the Julich Brain Atlas, which localizes each
detected region within specific brain structures.
These anatomically grounded maps are transformed into structured
intermediate representations – formatted in JSON – that encode spa-
tial and diagnostic attributes in a format interpretable by LLMs. This
structured input constrains the language model’s generative process,
ensuring that the final textual reports remain factually consistent and
clinically relevant. The generated explanations are designed to emulate
the style and reasoning found in radiology reports, making the outputs
directly useful for medical professionals.
To evaluate the effectiveness of this framework, we introduce an
assessment strategy that combines quantitative metrics – such as read-
ability, coherence, and lexical diversity – with domain-specific criteria
related to diagnostic completeness and clinical accuracy. Through this
multi-faceted evaluation, we aim to validate both the interpretability
and the practical utility of the generated reports.
In doing so, this work contributes to the development of trustworthy
and human-centered AI systems in medical imaging. By integrating
anatomical knowledge, visual context, and controlled language gener-
ation, we offer a new solution to the challenge of explainability – one
that aligns closely with the cognitive and communicative practices of
clinicians.
The rest of this paper is structured as follows. Section 2 reviews
related work. Section 3 detail the materials and methods used in
this study, outlining the dataset, preprocessing techniques, and the
innovative framework proposed for integrating visual and textual ex-
planations. Section 4 presents the results of our experimental anal-
ysis. Section 5 concludes the paper and discusses potential future
developments of our research.
2. Background and objective
Explainable AI has gained significant traction in healthcare, par-
ticularly in medical image analysis. XAI techniques aim to enhance
the transparency and interpretability of deep learning models, enabling
medical professionals to understand the decision-making process and
build trust in AI-based systems [6].
A substantial body of research in XAI for medical imaging fo-
cused on generating visual explanations that elucidate the model’s
decision-making process. Common approaches include saliency maps,
gradient-based methods (e.g., Grad-CAM [3]), and relevance propa-
gation techniques (e.g., Layer-wise Relevance Propagation, LRP [7]).
These methods visually represent which regions of an input image are
most influential in the model’s predictions.
Specifically, in brain tumor detection and segmentation from MRI
data, several studies have integrated XAI techniques with deep learning
models to enhance trust in AI-based computer-aided diagnostic systems.
Zeineldin et al. [8] proposed the NeuroXAI framework, which combines
deep neural networks with various XAI methods to generate visual
explanation maps for brain tumor classification and segmentation tasks.
Similarly, different studies have also integrated XAI techniques to
provide visual explanations in the context of multiple sclerosis lesion
segmentation from MRI data. Sadeghibakhi et al. [9] utilized M3d-CAM
to generate 3D attention maps, visually highlighting the regions in the
input images that are most influential in the model’s predictions.
While these visual explanation methods have proven useful, they
have limitations in providing comprehensive clinical explanations. As
discussed in the introductory section, heatmaps and similar visual-
izations often fail to provide the contextual information and patho-
logical significance that medical professionals require for informed
decision-making [4].
Recent advances in Large Language Models have shown promise
in generating textual explanations in natural language for medical
imaging tasks. One notable example is ChatCAD, which integrates LLMs
with computer-aided diagnosis models for medical image analysis [5].
In the ChatCAD framework, trained models first process medical images
to obtain outputs such as disease classification probabilities. These
outputs are then translated into natural language prompts for an LLM,
which generates summarized reports and diagnostic conclusions and
engages in conversations about symptoms, diagnosis, and treatment.
However, the use of LLMs to generate medical reports and tex-
tual explanations has not been widely studied. One challenge is that
the quality and accuracy of the generated reports depend heavily
on how the input prompts are formulated. Furthermore, there is a
risk that LLMs may produce inaccurate information, emphasizing the
need to base their output on verified models and established medical
knowledge.
Although visual and textual explanations have been explored sepa-
rately, very few examples effectively integrate both approaches. Rad-
Former [10] is a notable exception, combining global attention mech-
anisms for regions of interest with local attention on bag-of-words
style feature embeddings. This approach allows RadFormer to generate
heatmap visualizations and textual explanations mapped to radiolog-
ical lexicons for tasks like gallbladder cancer detection in ultrasound
images.
Recent studies underscore the crucial role of contextual information
in generating meaningful diagnostic insights. Research, such as the
work on EgoCap and EgoFormer [11], explicitly demonstrates that
explanatory mechanisms must be rigorously contextualized to deliver
clinically substantive interpretations. This aligns with the need to in-
tegrate visual and textual explanations to enhance interpretability,
ensuring that AI-generated outputs provide relevant and actionable
insights for medical professionals.
Despite these advancements, a significant gap remains in research
that combines state-of-the-art visual explanation techniques with tex-
tual explanation generation in medical imaging. Our work aims to
address this gap by proposing an innovative approach that integrates vi-
sual and textual explanations, offering complementary information that
aligns more closely with physicians’ reasoning and reporting processes.
A key advantage of our approach is its ability to generate clinically
meaningful reports without relying on preexisting textual datasets or
external knowledge bases, thereby overcoming the limitations seen
in systems like ChatCAD and RadFormer, which heavily depend on
curated data or large-scale knowledge repositories.
To summarize, this work introduces a novel framework that, for the
first time, integrates semantic segmentation, brain atlas-based anatom-
ical mapping, and LLMs to generate structured and interpretable medi-
cal reports in the domain of neuroimaging. A central innovation lies
in the use of a structured JSON-based intermediate representation,
which mediates between the segmentation output and the language
Computer Methods and Programs in Biomedicine 270 (2025) 108922
2
A.G. Valerio et al.
Fig. 1. Illustration of the proposed methodology pipeline, which includes semantic segmentation, brain atlas mapping, JSON construction, and LLM prompting. The workflow
begins with an input MRI volume processed by the segmentation model. The resulting mask is then mapped onto the Julich Brain Atlas for anatomical region extraction. These
regions are structured into a standardized JSON format, which serves as input for the prompting phase of LLMs, ultimately generating clinically interpretable textual reports.
model, ensuring factual consistency and significantly reducing the risk
of hallucinations during text generation.
The framework also incorporates an error mitigation strategy that
prioritizes explanations based on the most affected brain regions,
thereby minimizing the influence of segmentation inaccuracies on the
final report. We further present a comprehensive evaluation of the
generated explanations using a range of linguistic metrics, such as
coherence, lexical diversity, and readability, applied across different
LLM configurations.
Building on these contributions, the following section presents a de-
tailed description of the proposed methodology, outlining how segmen-
tation, anatomical mapping, and language generation are seamlessly
integrated to produce clinically meaningful, interpretable outputs.
3. Methods
3.1. Framework overview
We propose an innovative and explainable approach that combines
advanced deep learning techniques with Large Language Models to gen-
erate interpretable medical reports in natural language. The proposed
approach differs from conventional techniques in the scientific litera-
ture, as it generates textual explanations without relying on existing
text related to medical images or external knowledge sources.
An essential advantage of LLMs, in fact, lies in their ability to
leverage the extensive knowledge gained through pre-training on vast
textual corpora. This training equips LLMs with a comprehensive un-
derstanding of medical knowledge, encompassing disease pathologies,
clinical manifestations, and treatment methods. Such a rich foundation
enables LLMs to effectively apply this understanding to specific med-
ical image-processing tasks, provided they are supplied with accurate
factual data. By utilizing this rich pre-trained knowledge, LLMs can gen-
erate accurate, relevant, and contextually appropriate textual explana-
tions that reduce hallucinations commonly associated with generative
models, thereby enhancing the overall interpretability and usefulness
of AI-generated insights in medical imaging and assisted diagnostics.
The proposed methodology comprises four stages, as shown in Fig.
1:
• Semantic segmentation: A segmentation model processes 3D brain
MRI volumes to segment pathological regions such as gliomas.
• Brain atlas mapping: The segmented tumor regions are registered
to the MNI152 standard space and mapped onto the Julich Brain
Atlas to determine the most affected anatomical areas.
• JSON construction: The extracted spatial and quantitative
information – including affected regions, tumor distribution, seg-
mentation confidence, and model metadata – is encoded into a
structured JSON file.
• LLM prompting: The JSON file is used to prompt advanced LLMs
(Gemma, Llama, Mistral), which generate natural language med-
ical reports describing the tumor’s location, extent, and potential
clinical relevance.
In this paper, our approach is applied explicitly first to brain tumors
and then to multiple sclerosis lesion segmentation in 3D brain MRI.
While these initial applications demonstrate promising results within
the domain of brain imaging, further validation on other organs and
diseases is necessary to confirm broader generalizability. Nonetheless,
the methodology is inherently adaptable and holds potential for ex-
tension to other anatomical contexts, provided suitable segmentation
models and structural atlases are available. The use of a detailed
human body atlas enables the extraction of semantically meaningful,
anatomically grounded information that can serve as factual input for
LLM-based report generation.
Computer Methods and Programs in Biomedicine 270 (2025) 108922
3
A.G. Valerio et al.
Fig. 2. MRI modalities in the BraTS 2023 dataset. From left to right: T1Gd, T1, T2-FLAIR, and T2 images. Each modality provides complementary anatomical and pathological
information, contributing to a comprehensive understanding of tumor characteristics and brain tissue differentiation.
For instance, when examining liver conditions like cirrhosis or
tumors, structural alterations such as nodules or masses help identify
an unhealthy liver [12]. By incorporating a human body atlas, structure
and tissue changes become readily identifiable and quantifiable. Addi-
tionally, many diseases have secondary effects on surrounding struc-
tures [13]. The human anatomy atlas, mapping the spatial relationships
between organs, blood vessels, and nerves, aids in understanding how
a localized disease can impact surrounding tissues and organs.
Beyond our current focus on glioma and MS lesions, our framework
can seamlessly incorporate established structural atlases or atlas-based
tools, such as 3D Slicer [14], for identifying anatomical regions across
various organs. For instance, the Couinaud classification system pro-
vides a structural framework for analyzing liver tumors. At the same
time, lung segmentation techniques aid in identifying abnormalities
within specific lung regions, such as lobes or bronchial trees. The
ability to integrate these anatomical references into our medical imag-
ing workflows further underscores the flexibility of our approach,
enhancing our framework’s applicability and generalizability, making
it a versatile tool in medical imaging and diagnosis across multiple
scenarios.
3.2. Data
We used the dataset associated with the first task of the BraTS 2023
challenge, organized by the Radiological Society of North America
(RSNA), the American Society of Neuroradiology (ASNR), and the
Medical Image Computing and Computer Assisted Interventions (MIC-
CAI) society. Specifically, the dataset comprises scans from 1470 adult
patients affected by brain glioma tumors [15–19].
Each MRI scan in the BraTS 2023 dataset is provided as a NIfTI file
and includes the following four MRI modalities:
• Post-contrast T1-weighted (T1Gd): T1 scan taken after a contrast
agent injection.
• Native (T1): Captures the brain’s natural state.
• T2 Fluid Attenuated Inversion Recovery (T2-FLAIR): Suppresses
the signal from free water, making it easier to see lesions and
other abnormalities.
• T2-weighted (T2): Highlights fluid-filled areas.
The imaging data underwent preprocessing steps, including co-
registration to a standard anatomical template, interpolation to a uni-
form resolution of 1 mm3, and skull stripping [15]. An example of the
MRI modalities in the BraTS 2023 dataset is shown in Fig. 2.
One to four expert raters have manually annotated all images in
the dataset following a standardized annotation protocol, with anno-
tations approved by experienced neuroradiologists. The annotations
include the GD-enhancing tumor (ET), the peritumoral edematous/in-
vaded tissue (ED), and the necrotic tumor core (NCR). The sub-regions
considered for evaluation were defined as follows:
• Enhancing tumor (ET): Areas showing hyper-intensity in the T1Gd
modality compared to T1 and healthy white matter in T1Gd.
• Tumor core (TC): Comprises the ET and NCR parts of the tumor,
typically resected during surgery.
• Whole tumor (WT): The complete extent of the disease, including
the TC and ED, typically represented by a hyper-intense signal in
the FLAIR modality.
3.3. Semantic segmentation
The first stage involves preprocessing the imaging data and per-
forming data augmentation to enhance the quality and diversity of the
training dataset. Intensity normalization was applied to mitigate differ-
ences in intensity distributions across scanners and imaging protocols.
The image intensities were normalized based on the mean and standard
deviation of the non-zero voxel values within each channel using the
formula:
Normalized Intensity = Image Intensity − Mean
Standard Deviation . (1)
Inspired by previous approaches in the literature [20,21], we also
applied data augmentation techniques, including random flipping along
the three spatial axes, random spatial cropping, and random intensity
scaling and shifting. These techniques increase the diversity of the
training data and improve the model’s generalization ability.
We explored different state-of-the-art deep learning models for se-
mantic segmentation:
• Residual U-Net [22]: This model enhances the classic U-Net archi-
tecture by incorporating residual units in both the encoding and
decoding paths. It uses parametric rectified linear unit (PReLU)
activations and instance normalization. The encoding path em-
ploys residual units with varying strides for downsampling, while
the decoding path uses transposed convolutions for learnable
upsampling.
• SegResNet [23]: Designed for 3D medical image segmentation,
SegResNet is a 3D CNN based on the residual network (ResNet)
design. It features an encoder–decoder structure with four down-
sampling blocks in the encoder and three upsampling blocks in
the decoder.
• SwinUNETR [24]: This model combines a hierarchical Swin
Transformer encoder with a CNN-based decoder in a U-shaped
architecture. Developed for multi-modal 3D brain tumor segmen-
tation, it achieved top performance in the BraTS 2021 challenge.
SwinUNETR processes 3D MRI data by dividing it into non-
overlapping patches and employs a shifted windowing approach
for efficient modeling of long-range dependencies.
To prevent data leakage, a stratified subject-wise splitting of the
dataset was employed, ensuring data from the same patient was present
in either the training or testing set but not in both [25,26].
This first stage of semantic segmentation provides an initial level
of visual explainability in medical imaging. By highlighting the affected
regions of interest (ROIs) within the MRI images, radiologists and
medical professionals have a clear and intuitive explanation of tumor
location, size, and morphology. This approach mitigates diagnostic
uncertainty arising from complex brain structures and the often subtle
differences between tumorous and healthy tissues in MRI images. A
properly trained segmentation algorithm demarcates these distinctions
Computer Methods and Programs in Biomedicine 270 (2025) 108922
4
A.G. Valerio et al.
Fig. 3. Visualization of affected brain regions identified through brain atlas mapping. The mapped regions are overlaid on a 3D brain model and displayed from three anatomical
perspectives: left (lateral), back (posterior), and top (superior).
by precisely outlining abnormal growths, effectively explaining areas of
concern with accurate boundaries. In addition, differentiating between
different tumor sub-regions with different colors clarifies which tumor
parts exhibit specific characteristics. This level of detail can explain
whether a tumor is growing aggressively or has spread based on the
specific segmentation patterns observed.
3.4. Brain atlas mapping
In the second stage, we mapped the segmented ROIs (i.e., the tumor
core) to specific brain areas using the Julich Brain Atlas [27], a unique
and comprehensive three-dimensional atlas resulting as a culmination
of 25 years of research. It provides detailed probabilistic maps of
over 200 cortical areas and subcortical nuclei in the human brain,
all based on cytoarchitectonic differences. This atlas stands out for
its ability to represent the structural and functional organization of
the brain, capturing inter-individual variations through the mapping
of postmortem brains.
The mapping process involved overlaying the ROIs onto the stan-
dard MNI152 brain space. From there, we generated a point set rep-
resentation of the tumor core voxels, which were then assigned to
the corresponding brain regions in the labeled atlas map. An example
of this brain atlas mapping process for brain tumor segmentation is
illustrated in Fig. 3.
Through this process, we identified the top five most affected brain
regions and quantified the tumor volume in each region. This stage is a
significant step in understanding the impact of pathological conditions
on specific brain areas.
3.5. JSON construction
We constructed a structured JSON file in the third stage to encapsu-
late the relevant information extracted from the semantic segmentation
and brain atlas mapping stages. This step is crucial for providing the
LLM with structured, factual information, thereby mitigating the risk of
hallucinations—a common issue language models face when generating
text without a factual basis. The JSON file serves as a source of truth,
ensuring that the LLM’s outputs remain consistent with the quantitative
analysis and do not deviate into unreliable or fictional content.
The JSON file contains comprehensive information extracted from
the previous stages for the brain tumor segmentation task. Specifically,
it includes:
• The top five affected brain regions as determined during the brain
atlas mapping stage, along with their respective percentages of
tumor occupation and the percentage of each region affected.
• The color-coding scheme used for the semantic segmentation
masks, associating specific colors with the tumor core, peritu-
moral edema, and GD-enhancing tumor.
• Performance metrics for the semantic segmentation, including
Dice scores and Hausdorff distances for enhancing tumor, tumor
core, whole tumor, and their averages.
• The name of the model used for semantic segmentation.
This structured representation ensures that the subsequent LLM
prompting stage can access accurate, quantitative data derived from the
image analysis, facilitating the generation of factual and trustworthy
reports. The JSON template for brain tumor segmentation is shown in
Listing 1.
Listing 1: Template for the JSON file given as input to the LLM in
conjunction with the prompt. Capitalized strings indicate placeholders
for the values.
1 {
2 " MRI_Scan " : {
3 " Tumor_Details " : {
4 " Spatial_Distribution " : [
5 {
6 " Region " : " REGION_AFFECTED " ,
7 " Percentage_of_Tumor " : " % _OF_TUMOR " ,
8 " Percentage_of_Region_Affected " : " %
_OF_REGION "
9 },
10 ...
11 ],
12 " Semantic_Segmentation " : {
13 " Tumor_Core " : {
14 " Color " : " red "
15 },
16 " Peritumoral_Edema " : {
17 " Color " : " yellow "
18 },
19 " GD_Enhancing_Tumor " : {
20 " Color " : " green "
21 }
22 },
23 " Segmentation_Confidence " : {
24 " Dice_Score " : {
25 " Enhancing_Tumor " : " DICE_SCORE_ET " ,
26 " Tumor_Core " : " DICE_SCORE_TC " ,
27 " Whole_Tumor " : " DICE_SCORE_WT " ,
28 " Average " : " DICE_SCORE_AVG "
29 },
30 ...
31 },
32 " Model_Used " : " SegResNet "
33 }
34 }
35 }
3.6. Large language model prompting
In the final stage, we employed three state-of-the-art LLMs to gener-
ate textual explanations (i.e., medical reports) based on the information
contained in the JSON file. The following LLMs were used:
• Gemma 9B-IT [28]: It is the 9 billion parameter instruction-tuned
version of Google DeepMind’s open-weight LLM, trained on a
Computer Methods and Programs in Biomedicine 270 (2025) 108922
5
A.G. Valerio et al.
Fig. 4. Prompt used for report generation.
Table 1
Computational details.
Component Model
CPU Intel(R) Xeon(R) Gold 5317, @ 3.00 GHz, 12 cores
GPU NVIDIA A100 PCIe 80 GB
RAM 90 GB
curated mix of web documents, code, and dialogue data. Based
on the Gemini research, it features a decoder-only Transformer
architecture with multi-query attention optimized for efficient
inference and safe conversational use.
• Llama 3 70B-IT [29]: It is the 70 billion parameter instruction-
tuned version of Meta AI’s LLM, which overcomes previous ver-
sions through innovations like grouped query attention, improved
tokenization, and extensive curation of the 15 trillion token pre-
training dataset across languages and domains.
• Mistral Saba 24B [30]: It is the 24 billion parameter instruction-
tuned LLM from Mistral AI, designed for high-quality multilingual
reasoning and tool use. Trained on a sophisticated data mix that
includes synthetic tools and multilingual tasks, it builds on Mis-
tral’s dense decoder-only architecture with optimized attention
mechanisms, yielding strong performance across benchmarks.
After careful prompt engineering, we designed an articulated
prompt to provide the LLMs with context and guidance, enabling them
to generate coherent and informative medical reports. The prompt
incorporated the structured data from the JSON file, allowing the LLMs
to integrate the quantitative information and generate human-readable
textual descriptions of the tumor location, extent, and potential impact
on brain functions. Integrating structured data into the prompt helps
mitigate the risk of hallucinations and ensures accurate and relevant
medical reports are generated. Fig. 4 shows the prompt used for the
brain tumor task.
The following section details the experimental setup and evaluation
results obtained by applying the proposed framework to brain MRI
datasets for glioma and MS lesion segmentation.
4. Experimental results
4.1. Setting
We used a robust hardware setup for the experiments, as detailed
in Table 1, and several specialized libraries: PyTorch [31] as the deep
learning framework; MONAI [32] for implementing semantic segmenta-
tion models, data preprocessing, and metric calculations; Nibabel [33]
and Nilearn [34] for reading, manipulating, and visualizing MRI data;
and Siibra [35] for all operations related to brain atlas mapping. The
Table 2
Segmentation model hyperparameters.
Hyperparameter Value
Optimizer Adam
Learning rate 1e−4
Weight decay 1e−5
Loss Dice loss
Scheduler Cosine Annealing LR
Epochs 100
Patience 10
Table 3
LLM hyperparameters.
Hyperparameter Value
Decoding strategy Greedy
Temperature 0.6
Max tokens 1024
Top p 0.9
Groq API [36] was used to access and interact with the LLMs for the
textual explainability component. A comprehensive overview of the
critical hyperparameters used during the segmentation model training
phase and the LLM prompting phase is provided in Tables 2 and 3,
respectively. All other parameters not listed follow the default settings
of the corresponding libraries.
4.1.1. Semantic segmentation
We employed the following experimental setup. The dataset was
divided into 80% for training and 20% for testing, with an additional
20% of the training set used for validation.
We employed state-of-the-art deep learning models for semantic
segmentation and assessed their performance using task-specific met-
rics for brain tumor segmentation, computed across the entire test
set and complemented by 95% confidence intervals estimated through
bootstrapping with 1000 iterations to capture statistical variability. The
performance of the employed models was assessed using two widely
adopted metrics:
• Dice Score: It measures the similarity between the predicted
segmentation masks and the ground truth masks, ranging from 0
to 1, with higher values indicating better segmentation accuracy.
The Dice Score is defined as:
Dice = 2|𝐴 ∩ 𝐵|
|𝐴| + |𝐵| , (2)
where 𝐴 and 𝐵 are the predicted and ground truth binary masks,
respectively.
• Hausdorff Distance: This metric quantifies the maximum distance
between the predicted segmentation masks and the ground truth
Computer Methods and Programs in Biomedicine 270 (2025) 108922
6
A.G. Valerio et al.
Table 4
Dice scores for the three segmentation models: Residual U-Net, SegResNet, and SwinUNETR. The scores are reported for enhancing
tumor (ET), tumor core (TC), whole tumor (WT), and the average score across these classes. For each metric, the corresponding 95%
confidence interval (CI) is also provided.
Model ET [CI 95%] TC [CI 95%] WT [CI 95%] Average [CI 95%]
Residual U-Net 0.8323 0.8878 0.9109 0.8770
[0.8074, 0.8558] [0.8673, 0.9048] [0.9012, 0.9204] [0.8615, 0.8908]
SegResNet 0.8585 0.9130 0.9289 0.9001
[0.8353, 0.8797] [0.8955, 0.9282] [0.9208, 0.9364] [0.8867, 0.9119]
SwinUNETR 0.8511 0.8992 0.9217 0.8907
[0.8262, 0.8740] [0.8802, 0.9164] [0.9125, 0.9305] [0.8758, 0.9049]
masks, with lower values indicating better boundary adherence.
The Hausdorff Distance is defined as:
𝑑𝐻 (𝐴, 𝐵) = max
{
sup
𝑎∈𝐴
inf
𝑏∈𝐵 𝑑(𝑎, 𝑏), sup
𝑏∈𝐵
inf
𝑎∈𝐴 𝑑(𝑎, 𝑏)
}
, (3)
where 𝑑(𝑎, 𝑏) is the Euclidean distance between points 𝑎 and 𝑏, and
𝐴 and 𝐵 are the sets of points on the boundaries of the predicted
and ground truth masks, respectively.
4.1.2. Textual explainability
To assess the quality and effectiveness of the generated medical
reports, we developed a comprehensive evaluation framework that
included various metrics: lexical diversity, readability, coherence, and
coverage of information. To support the calculation of these metrics,
we employed the Spacy library [37]. Specifically, the following metrics
were used to evaluate the output of LLMs:
:
• Lexical diversity:
– Type-Token Ratio (TTR): This measure of lexical diversity
is defined as the ratio of unique words (types) to the total
number of words (tokens) in a text [38]. A higher TTR value
indicates greater lexical richness and diversity in the output.
The formula for calculating TTR is:
TTR = Number of Unique Words
Total Number of Words . (4)
– Maas’ Index: This adaptation of the Type-Token Ratio aligns
the measure with a logarithmic scale [38]. A lower score
on Maas’ Index signifies greater diversity in vocabulary. The
index is calculated using the formula:
Maas’ = log(nTokens) − log(nTypes)
log(nTokens)2 . (5)
• Readability:
– Flesch Reading Ease Score (FRES): The score evaluates the
ease of reading a given text based on factors such as word
length and sentence complexity [39]. The formula for cal-
culating FRES is:
FRES = 206.835 − (1.015 × ASL) − (84.6 × ASW), (6)
where ASL represents the Average Sentence Length, and
ASW represents the Average Number of Syllables per Word.
• Coherence:
– Coherence Score (CohS): This newly proposed metric quan-
tifies the coherence and flow of the generated textual ex-
planations, assessing the logical structure and transitions
between ideas within the output. The coherence score is
calculated as follows:
CohS = 1
𝑁 − 1
𝑁−1∑
𝑖=1
cos(𝑆𝑖, 𝑆𝑖+1), (7)
where 𝑁 is the number of sentences and cos(𝑆𝑖, 𝑆𝑖+1) is the
cosine similarity between the embeddings of sentences 𝑖 and
𝑖 + 1.
• Coverage of Information:
– Embedding-based Coverage Score (ECS): This new metric,
introduced in this paper, measures the coverage of informa-
tion in the LLM’s output by comparing the embeddings of
the sentences in the prompt and the corresponding output.
The coverage score is calculated as follows:
ECS = 1
𝑁
𝑁∑
𝑖=1
cos(𝑆text, 𝑆ref𝑖 ), (8)
where:
∗ 𝑆text is the embedding of the LLM output.
∗ 𝑆ref𝑖 is the embedding of the 𝑖th sentence in the refer-
ence text (LLM prompt).
∗ 𝑁 is the total number of sentences in the reference
text.
∗ cos(𝑆text, 𝑆ref𝑖 ) is the cosine similarity between the
embedding of the LLM output and the reference text
(prompt) sentence.
– Token-based Coverage Score (TCS): This newly proposed
metric quantifies the overlap between the tokens in the
prompt and the output. It is calculated as the ratio of
the intersection of tokens to their union after removing
punctuation and stop words from both sets:
TCS = |textTokens ∪ refTokens|
|textTokens ∩ refTokens| . (9)
This metric evaluates how well the LLM incorporated rele-
vant information from the prompt into its generated output.
4.2. Results
4.2.1. Semantic segmentation
The Dice Scores obtained for the three segmentation models (Resid-
ual U-Net, SegResNet, and SwinUNETR) are presented in Table 4. These
scores reflect the performance of each model for the three segmentation
classes, enhancing tumor (ET), tumor core (TC), and whole tumor (WT),
and their averages. The Hausdorff Distances for the three models are
shown in Table 5. These distances indicate the boundary adherence of
each model for the three segmentation classes and their averages.
Based on the quantitative results, the SegResNet model emerged
as the best-performing architecture for the brain tumor segmenta-
tion task, achieving the highest values for the performance metrics
across all classes. The SwinUNETR model demonstrated competitive
performance, while the U-Net model lagged in both evaluation metrics.
Given these results, the SegResNet model was selected for use in
the subsequent stages of the pipeline. Examples of the segmentation
provided by SegResNet for brain tumors are shown in Fig. 5.
Computer Methods and Programs in Biomedicine 270 (2025) 108922
7
A.G. Valerio et al.
Fig. 5. Results provided by the SegResNet model for brain tumor segmentation. From left to right: 4-channel MRI input scan, ground truth mask for the whole tumor (WT), and
predicted segmentation mask.
Table 5
Hausdorff Distances for the three segmentation models: Residual U-Net, SegResNet, and
SwinUNETR. The distances are reported for enhancing tumor (ET), tumor core (TC),
whole tumor (WT), and the average distance across these classes.
Model ET TC WT Average
Residual U-Net 4.9207 5.4873 8.0632 6.1571
SegResNet 3.4936 4.0667 4.5787 4.0463
SwinUNETR 4.0749 4.7311 7.6993 5.5018
4.2.2. Textual explainability
The LLMs employed in our framework – Gemma, Llama, and Mistral
– exhibited varying performance levels in generating textual expla-
nations for the segmentation task. Each model demonstrated unique
strengths across different aspects of text generation. We evaluated the
performance of these LLMs using several metrics: lexical diversity,
readability, coherence, and coverage of information. We recognize the
critical importance of refining prompt engineering to ensure consis-
tent and reliable report generation across different LLMs. To address
this concern, we emphasize that an empirical selection process was
conducted, prioritizing a prompt design that optimizes coherence and
comprehensive coverage while minimizing inter-model performance
differences. Table 6 summarizes the evaluation results for the brain
tumor segmentation task. For each LLM, we report the average perfor-
mance across all generated reports in the test set, along with the stan-
dard deviation of each metric, to provide insight into the consistency
of the outputs.
The results in Table 6 highlight the diverse strengths among the
models. Gemma demonstrated superior lexical richness, achieving the
highest Type-Token Ratio of 0.351 and the lowest Maas’ Index of 0.023,
indicating a rich and diverse vocabulary. Mistral led in coherence with
a score of 0.357. Llama slightly outperformed the other models in read-
ability with a Flesch Reading Ease Score of 37.204, Embedding-based
Coverage (0.433) and Token-based Coverage (0.305). An example of
the output generated by Llama for the brain tumor segmentation is
shown in Fig. 6.
4.2.3. Error evaluation
Although this approach offers substantial benefits, it has shortcom-
ings and could be further refined. For instance, our top-performing
model achieves an average Dice Score of 0.90 for segmentation mask
generation in the brain tumor task. Nevertheless, even minor inaccura-
cies in these masks can lead to ambiguous visual explanations. These
imprecisions can also cascade into subsequent analyses, potentially
leading to misinterpretations of the tumor’s dimensions, shape, spread,
and potential effects on brain functionality. The model’s segmentation
errors, whether over-segmentation (excessive marking of tumor area)
or under-segmentation (omitting tumor parts) are distributed across
the entire tumor region. In our framework, we introduced a specific
mitigation strategy to reduce the impact of such errors on the report
generation process. By deliberately restricting the input to the five
brain areas most significantly affected by the disease, the pipeline limits
the propagation of false positives and false negatives to the textual
output. This targeted filtering is a purposeful design choice within
our framework and should not be considered an inherent property
of all segmentation-to-report pipelines. This constraint, while limiting,
safeguards against propagating segmentation errors.
To quantitatively assess the impact of segmentation inaccuracies
and further validate the applied mitigation strategy, we conducted a
comparative analysis across the entire test set. Specifically, we exam-
ined the brain regions extracted using the segmentation masks gener-
ated by the SegResNet model against their corresponding ground truth.
We then progressively reduced the number of regions considered for
medical report generation, starting from a maximum of 30 (the highest
value observed in the test set) down to a single region.
The results, presented in Fig. 7, show the percentage of test set
examples containing at least one region that would not have been af-
fected by the tumor if identified using the corresponding ground truth.
Notably, by setting a threshold of five regions for medical report gener-
ation, we achieve a 96.6% reduction in error propagation, decreasing
from 53.1% to 1.8% for cases with at least one incorrect region out
of five. Furthermore, for cases with two incorrect regions out of five,
the error is reduced by 98.3%, from 23.5% to 0.4%. Importantly, no
instances were found with more than two erroneous regions out of five,
demonstrating the high reliability of our error mitigation strategy and
the overall robustness of our methodology in minimizing segmentation
inaccuracies and ensuring consistency in medical report generation.
4.2.4. Statistical analysis
To determine whether observed differences in report quality are
statistically meaningful across the evaluated LLMs (Gemma, Llama,
and Mistral) we conducted pairwise statistical comparisons on the
six text quality metrics: Type-Token Ratio (TTR), Maas’ Index, Flesch
Reading Ease Score (FRES), Coherence Score (CohS), Embedding-based
Coverage Score (ECS), and Token-based Coverage Score (TCS). Given
that each LLM generated reports for the same test set instances, we
employed paired statistical tests, ensuring that comparisons were made
on a per-sample basis.
For each metric, the choice of statistical test was guided by the
distribution of paired differences. When the normality assumption held,
verified using the Shapiro–Wilk test [40], we applied a paired t-test;
otherwise, we used the Wilcoxon signed-rank test, a non-parametric
alternative suitable for skewed or ordinal data.
The results of the pairwise comparisons are summarized in Ta-
ble 7. Significant differences (with 𝑝 < 0.05) were found between
Llama and Mistral for all six metrics. Llama outperformed Mistral in
vocabulary diversity (TTR and Maas’), readability (FRES), coherence
(CohS), and both coverage metrics (ECS and TCS), with all p-values
indicating strong evidence of difference. Comparisons between Llama
and Gemma also revealed significant differences across all metrics.
These results indicate that Llama consistently produced outputs with
statistically distinct linguistic and semantic characteristics compared
to Gemma, suggesting meaningful performance differentiation between
the two models. In contrast, the comparison between Mistral and
Gemma yielded non-significant differences for Coherence Score (𝑝 =
Computer Methods and Programs in Biomedicine 270 (2025) 108922
8
A.G. Valerio et al.
Table 6
Performance metrics for brain tumor segmentation across different LLMs. Metrics include Type-Token Ratio (TTR), Maas’ Index (Maas’), Flesch
Reading Ease Score (FRES), Coherence Score (CohS), Embedding-based Coverage Score (ECS), and Token-based Coverage Score (TCS). For each
metric, the mean and standard deviation across the test set are reported to reflect the variability in the LLM outputs.
Metric Definition Gemma Llama Mistral
TTR The ratio of unique words (types) to the total number of
words (tokens) in a text.
𝟎.𝟑𝟓𝟏 ± 𝟎.𝟎𝟏𝟗 0.282 ± 0.031 0.276 ± 0.024
Maas’ TTR on a logarithmic scale. 𝟎.𝟎𝟐𝟑 ± 𝟎.𝟎𝟐𝟏 0.029 ± 0.002 0.027 ± 0.002
FRES The ease of reading a given text based on factors such as
word length and sentence complexity.
32.224 ± 4.749 𝟑𝟕.𝟐𝟎𝟒 ± 𝟓.𝟑𝟓𝟗 34.108 ± 6.059
CohS The coherence of the generated text, assessing the logical
structure and transitions between ideas within the output.
0.357 ± 0.039 0.319 ± 0.071 𝟎.𝟑𝟓𝟕 ± 𝟎.𝟎𝟑𝟔
ECS The coverage of information by comparing the embeddings
of the sentences in the prompt and the corresponding output.
0.426 ± 0.016 𝟎.𝟒𝟑𝟑 ± 𝟎.𝟎𝟐𝟎 0.424 ± 0.019
TCS The coverage of information by comparing the overlap
between the tokens in the prompt and the output.
0.257 ± 0.022 𝟎.𝟑𝟎𝟓 ± 𝟎.𝟎𝟑𝟑 0.278 ± 0.024
Fig. 6. Example of Llama output for the brain tumor segmentation task.
0.97) and Embedding-based Coverage Score (𝑝 = 0.07). This suggests
that, for these two metrics, Mistral and Gemma produced broadly
similar outputs. However, significant differences were still observed in
TTR, Maas’, FRES, and TCS, indicating that Mistral and Gemma vary in
terms of lexical richness and surface-level textual properties.
These statistical results quantitatively confirm that the three LLMs
exhibit distinct strengths and weaknesses across different aspects of
report quality. In particular, Llama demonstrates consistent superiority
in both syntactic and semantic dimensions, while Mistral and Gemma
show overlap in certain areas but diverge in others. The use of paired
statistical tests guarantees that the observed differences are derived
from robust, sample-wise comparisons. These findings indicate that the
LLMs exhibit systematic, rather than random, variations in performance
across metrics, offering a statistically grounded basis for interpreting
and comparing their behavior in clinical text generation tasks.
4.3. Beyond brain tumors
The results’ applicability can be extended far beyond the specific
domain of brain tumor segmentation. The framework can be adapted
and generalized to various medical imaging modalities and pathologies,
thereby encouraging the more widespread adoption of explainable AI
systems in healthcare. We applied the proposed methodology to a
second clinical scenario: segmenting multiple sclerosis (MS) lesions.
Computer Methods and Programs in Biomedicine 270 (2025) 108922
9
A.G. Valerio et al.
Table 7
Statistical comparison of text quality metrics between LLM-generated reports. Pairwise comparisons were conducted between
Gemma, Llama, and Mistral using either the paired t-test (T) or the Wilcoxon signed-rank test (W), depending on the
distribution of the differences. Metrics evaluated include Type-Token Ratio (TTR), Maas’ Index, Flesch Reading Ease Score
(FRES), Coherence Score (CohS), Embedding-based Coverage Score (ECS), and Token-based Coverage Score (TCS). For each
comparison, the statistical test used, 𝑝-value, and significance (at 𝑝 < 0.05) are reported.
Metric Llama-Mistral Llama-Gemma Mistral-Gemma
Test p-value Significant Test p-value Significant Test p-value Significant
TTR W 0.03 True W <0.001 True T <0.001 True
Maas’ T <0.001 True W <0.001 True W <0.001 True
FRES W <0.001 True W <0.001 True W <0.001 True
CohS W <0.001 True W <0.001 True T 0.97 False
ECS T <0.001 True T <0.001 True T 0.07 False
TCS T <0.001 True T <0.001 True T <0.001 True
Fig. 7. Percentages of error reduction achieved through the proposed mitigation strategy. The chart reports the reduction in region-level mismatches across test set cases, grouped
by severity: cases with at least one, at least two, and at least three mismatched regions between the predicted and ground truth annotations.
4.3.1. Data
For the MS lesion segmentation task, a separate dataset was used,
comprising three distinct subsets: MSSEG-1 [41], ISBI [42], and
PubMRI [43]. This dataset is part of the 2022 Shifts challenge [44] and
focuses on white matter MS lesion segmentation in 3D MRI of the brain.
Each sample in the MS dataset consists of a 3D brain scan captured
using both FLAIR and T1 contrasts. These scans have undergone several
preprocessing steps to enhance their quality and consistency, including
denoising to reduce noise and improve image clarity, skull stripping
to remove non-brain tissues and focus the analysis on brain matter,
and bias field correction to address intensity non-uniformities in the
images. The combination of FLAIR and T1 provides complementary
information, with FLAIR being particularly sensitive to white matter
lesions and T1 offering detailed anatomical structure.
4.3.2. Method adaptations
The methodology is the same as that applied for brain tumors,
described in Section 3, with a few adaptations to make it suitable for
MS lesions.
A similar JSON file was constructed for the MS lesion segmentation
task. The main difference from the brain tumor JSON file is the absence
of color coding. Unlike the brain tumor JSON, which includes color
information for different lesion types, the MS lesion JSON does not, as
MS lesions are typically represented as a single class.
For the MS lesion segmentation task, the performance was evaluated
using the metrics suggested by the Shifts challenge:
• Normalized Dice Similarity Coefficient (nDSC) [45]: It provides
an unbiased metric for evaluating model performance in identify-
ing lesions, regardless of the lesion load across different patients.
It normalizes the precision for each patient while keeping the
recall fixed, allowing for fair comparison across different patients.
The nDSC is defined as:
nDSC = 2 (2 + 𝜅𝑝 + 𝑛)−1 , 𝜅 = ℎ (𝑟−1 − 1) , (10)
where 𝑝 = FP
TP and 𝑛 = FN
TP . In this context, ℎ represents the ratio of
the positive to the negative classes in the predicted mask ̂ 𝑌 , and
0 < 𝑟 < 1 is a reference value that denotes the mean fraction of the
positive class in ̂ 𝑌 across a large number of subjects, indicating
the average occurrence rate of the positive class.
• F1 Lesion Score: It assesses the lesion detection quality and is
defined as:
F1 = TP
TP + 0.5(FP + FN) , (11)
where TP (true positives) is when the detected lesion matches the
actual lesion well (IoU > 0.5), FP (false positives) is when the
predicted lesion does not match any actual lesion closely (IoU
< 0.5), and FN (false negatives) is when an actual lesion is not
detected correctly (IoU < 0.5).
• Robust Area Under the Curve (R-AUC) [44,46]: This metric jointly
assesses the model’s robustness to distributional shifts and the
quality of its uncertainty estimates. It is calculated as the area
under error-retention curves, plotting the mean error against
the fraction of retained predictions. Lower R-AUC indicates bet-
ter performance, achieved by improving prediction accuracy or
better correlating uncertainty estimates with errors.
4.3.3. Results
The nDSC, F1 Lesion Score, and R-AUC results for the three segmen-
tation models (Residual U-Net, SegResNet, and SwinUNETR) on the MS
lesion segmentation task are presented in Table 8. Consistent with the
brain tumor segmentation results, the SegResNet model outperformed
the other architectures in the MS lesion segmentation task, achieving
the highest scores across all three metrics. It was selected for use in
the subsequent stages of the pipeline. Examples of the segmentation
provided by SegResNet for MS lesions are shown in Fig. 8.
We also evaluated the performance of LLMs using the same metrics
described above for the brain tumor segmentation task. The results
Computer Methods and Programs in Biomedicine 270 (2025) 108922
10
A.G. Valerio et al.
Fig. 8. Results provided by the SegResNet model for multiple sclerosis lesion segmentation. From left to right: FLAIR MRI input scan, ground truth mask, and predicted segmentation
mask.
Table 8
Performance metrics for multiple sclerosis lesion segmentation using the three segmen-
tation models: Residual U-Net, SegResNet, and SwinUNETR. Metrics include Normalized
Dice Similarity Coefficient (nDSC), F1 Lesion Score (F1), and Robust Area Under the
Curve (R-AUC).
Model nDSC F1 R-AUC
Residual U-Net 0.6505 0.2094 0.1764
SegResNet 0.7175 0.3511 0.8745
SwinUNETR 0.6755 0.3263 0.3529
Table 9
Performance metrics for multiple sclerosis lesion segmentation across different LLMs.
Metrics include Type-Token Ratio (TTR), Maas’ Index (Maas’), Flesch Reading Ease
Score (FRES), Coherence Score (CohS), Embedding-based Coverage Score (ECS), and
Token-based Coverage Score (TCS).
Model TTR Maas’ FRES CohS ECS TCS
Gemma 0.533 0.019 17.600 0.385 0.595 0.230
Llama 0.381 0.026 41.770 0.354 0.541 0.295
Mistral 0.453 0.020 41.560 0.334 0.559 0.268
varied for the MS lesion segmentation task, as shown in Table 9.
Gemma achieved the highest lexical richness with a TTR of 0.533
and the lowest Maas’ Index of 0.019, again demonstrating superior
vocabulary diversity. Regarding readability, Llama led with an FRES
of 41.77, closely followed by Mistral at 41.56. Gemma maintained its
strong performance in Coherence with a score of 0.385 and excelled in
Embedding-based Coverage (0.595). However, Llama achieved the best
Token-based Coverage (0.295). An example of the output generated by
Llama for the MS lesion segmentation task is shown in Fig. 9.
These results establish a solid foundation for further analysis, which
we will discuss in the following section regarding methodological
implications, limitations, and potential future directions.
5. Discussion and conclusion
This paper presented an innovative framework to enhance the ex-
plainability of AI systems in medical imaging, specifically for brain seg-
mentation tasks. By integrating state-of-the-art semantic segmentation
techniques, brain atlas mapping, and Large Language Models, our ap-
proach generates comprehensive, clinically meaningful textual reports
that align with physicians’ diagnostic reasoning and reporting prac-
tices. Unlike conventional explainability methods that rely on heatmaps
or concept attribution, our framework offers structured, anatomically
grounded, and linguistically validated explanations – marking a step
towards clinically deployable AI transparency.
A distinctive aspect of the framework is its anti-hallucination design,
which incorporates structured data extraction and a JSON-based inter-
mediary combined with constrained prompts to ground LLM outputs in
factual, quantitative information. This approach addresses a common
limitation in naïve LLM-based report generation, significantly reducing
the risk of hallucinated content. Furthermore, by bridging visual and
textual modalities, the framework provides a more holistic explanation,
improving the interpretability of segmentation outputs for medical
professionals.
To ensure robustness and flexibility, we conducted multi-model
validation using different segmentation models and multiple LLMs,
demonstrating consistent performance across architectures and con-
firming the model-agnostic applicability of our method. The SegResNet
model emerged as the top performer for semantic segmentation, achiev-
ing high Dice Scores and low Hausdorff Distances for brain tumor
segmentation, as well as competitive performance in MS lesion seg-
mentation. Using a comprehensive set of metrics, we evaluated three
state-of-the-art LLMs (Gemma, Llama, and Mistral) in the textual ex-
planation generation phase. The results revealed that each model had
unique strengths in different aspects of text generation, such as lexical
diversity, readability, coherence, and information coverage. Llama con-
sistently performed well across both tasks, excelling in lexical diversity
and coverage metrics.
While our framework was primarily implemented for segmenting
brain tumors in magnetic resonance imaging of the brain, its versatility
allows for broader applications. The underlying principles of this ap-
proach can be easily adapted and applied to the segmentation of various
other medical conditions across different imaging modalities, as demon-
strated by the MS lesion detection results. This work contributes to
the ongoing efforts to make AI systems in healthcare more transparent,
interpretable, and trustworthy. By providing medical professionals with
detailed, context-rich explanations of AI-driven analyses, our approach
has the potential to support clinical decision-making and increase
confidence in AI-assisted diagnoses.
While this approach offers significant benefits, it has limitations
that warrant further refinement. Our top-performing model achieves
an average Dice Score of 0.90 in brain tumor segmentation, yet minor
inaccuracies can lead to ambiguous visual explanations and misinter-
pretations of tumor characteristics. Segmentation errors, whether over
or under-segmentation, are distributed across the tumor region. To
mitigate their impact on report generation, our framework includes a
dedicated strategy that limits the input to the five brain areas most
affected by the disease, reducing error propagation by up to 98.3%.
Additionally, to further enhance the transparency and interpretability
of our decision-making process, we incorporate segmentation confi-
dence values into the JSON construction. These values are subsequently
Computer Methods and Programs in Biomedicine 270 (2025) 108922
11
A.G. Valerio et al.
Fig. 9. Example of Llama output for the MS lesion segmentation task.
reflected in the generated medical report, providing an additional layer
of factual basis and allowing for a more informed interpretation of
the results. The strategy of focusing on the top five most affected
brain regions has proven effective in mitigating error propagation;
however, it may miss smaller yet clinically important findings, and
its reliability depends on the accuracy of atlas registration, which can
be influenced by anatomical variability, surgical alterations, or patho-
logical deformations. Given these considerations, we acknowledge the
potential benefits of dynamic thresholding and uncertainty calibration
and consider them promising directions for future research.
Moreover, the quality of the generated reports remains sensitive to
prompt design and model-specific characteristics. Despite using consis-
tent input structures, different LLMs exhibit varied behavior in terms of
coherence, coverage, and phrasing. This highlights the need for further
optimization of prompt strategies and model selection, particularly for
clinical deployment where consistency and reliability are critical.
Although the framework has demonstrated consistent performance
on well-curated datasets such as BraTS and Shifts, real-world clinical
settings may introduce variability in scanner types, imaging protocols,
and patient populations. While not directly addressed in this study,
managing potential domain shifts remains an important considera-
tion for future work to guarantee broad applicability across diverse
healthcare environments.
Additional future work could focus on refining LLM prompts by
integrating patient metadata and medical history for more personalized
reports, embedding the system into clinical workflows for effective
patient monitoring, and adapting the methodology to other imag-
ing tasks like lesion detection and disease classification. While our
evaluation relies on quantitative text-based metrics (e.g., readabil-
ity, coherence, lexical coverage), it is essential to complement these
with clinical user studies to assess report usefulness and real-world
applicability. Additional imaging techniques and patient information
can improve report accuracy, and implementing interactive feedback
systems can help refine the process. Creating domain-specific language
models and conducting clinical validation studies with medical profes-
sionals will ensure the system’s performance, usability, and alignment
with real-world medical practices.
CRediT authorship contribution statement
Alberto G. Valerio: Writing – review & editing, Writing – original
draft, Software, Methodology. Katya Trufanova: Software, Method-
ology. Salvatore de Benedictis: Software. Gennaro Vessio: Writing
– review & editing, Validation, Supervision, Methodology. Giovanna
Castellano: Validation, Supervision, Methodology.
Funding information
One author, Giovanna Castellano, has received funding from the
FAIR – Future AI Research project, Spoke 6 – Symbiotic AI (Grant
number: CUP H97G22000210007), under the NRRP MUR program
funded by NextGenerationEU. None of the other authors received any
specific funding.
Declaration of competing interest
There are no competing interests associated with this manuscript.
None of the authors have any financial, personal, or professional in-
terests that could be perceived to influence the work reported in this
study.
Acknowledgments
While preparing this work, the authors utilized ChatGPT and Gram-
marly to enhance language clarity and readability. The authors, who
take full responsibility for the final version of the manuscript, carefully
reviewed and refined all content generated by these tools.
References
[1] E. Tjoa, C. Guan, A survey on explainable artificial intelligence (xai): Toward
medical xai, IEEE Trans. Neural Netw. Learn. Syst. 32 (11) (2020) 4793–4813.
[2] A.B. Arrieta, N. Díaz-Rodríguez, J.D. Ser, A. Bennetot, S. Tabik, A. Barbado,
S. García, S. Gil-López, D. Molina, R. Benjamins, et al., Explainable Artificial
Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward
responsible AI, Inf. Fusion 58 (2020) 82–115.
Computer Methods and Programs in Biomedicine 270 (2025) 108922
12
A.G. Valerio et al.
[3] R.R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, Grad-CAM:
Visual explanations from deep networks via gradient-based localization, Int. J.
Comput. Vis. 128 (2) (2019) 336–359.
[4] W. Jin, X. Li, M. Fatehi, G. Hamarneh, Guidelines and evaluation of clinical
explainable AI in medical image analysis, Med. Image Anal. 84 (2023) 102684.
[5] W. Sheng, Z. Zhao, X. Ouyang, W. Qian, D. Shen, ChatCAD: Interactive computer-
aided diagnosis on medical image using large language models, 2023, arXiv:
2302.07257.
[6] K. Kaczmarek-Majer, G. Casalino, G. Castellano, M. Dominiak, O. Hryniewicz, O.
Kamińska, G. Vessio, N. Díaz-Rodríguez, PLENARY: Explaining black-box models
in natural language through fuzzy linguistic summaries, Inform. Sci. 614 (2022)
374–399.
[7] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.R. Müller, W. Samek, On
pixel-wise explanations for non-linear classifier decisions by layer-wise relevance
propagation, in: PLoS ONE, Vol. 10, Public Library of Science, 2015, e0130140.
[8] R.A. Zeineldin, M.E. Karar, Z. Elshaer, C.R. Wirtz, O. Burgert, F. Mathis-Ullrich,
et al., Explainability of deepDeepEBTDNet neural networks for MRI analysis of
brain tumors, Int. J. Comput. Assist. Radiol. Surg. (2022) 1–11.
[9] M. Sadeghibakhi, H. Pourreza, H. Mahyar, Multiple sclerosis lesions segmentation
using Attention-Based CNNs in FLAIR images, IEEE J. Transl. Eng. Heal. Med.
10 (2022) 1–11.
[10] S. Basu, M. Gupta, P. Rana, P. Gupta, C. Arora, RadFormer: Transformers with
global-local attention for interpretable and accurate gallbladder cancer detection,
2022, arXiv preprint arXiv:2211.04793.
[11] D. Zhuangzhuang, et al., EgoCap and EgoFormer: First-person image captioning
with context fusion, Pattern Recognit. Lett. 181 (2024) 50–56.
[12] M. Renzulli, et al., Morphological, dynamic and functional characteristics of liver
pseudolesions and benign lesions, Radiol. Medica 127 (2022) 129–144.
[13] B. Arneth, Tumor microenvironment, Medicina 56 (2020).
[14] R. Kikinis, et al., 3D Slicer: A Platform for Subject-Specific Image Analysis,
Visualization, and Clinical Support, Springer New York, 2014, pp. 277–289, Ch.
Intraoperative Imaging and Image-Guided Therapy.
[15] U. Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor
segmentation and radiogenomic classification, 2021, arXiv:2107.02314.
[16] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al.,
Segmentation labels and radiomic features for the pre-operative scans of the
TCGA-LGG collection, Cancer Imaging Arch. (2017).
[17] S. Bakas, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, J. Freymann, K. Farahani,
C. Davatzikos, Advancing The Cancer Genome Atlas glioma MRI collections with
expert segmentation labels and radiomic features, Sci. Data 4 (2017).
[18] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, J. Freymann,
K. Farahani, C. Davatzikos, Segmentation labels and radiomic features for the
pre-operative scans of the TCGA-GBM collection, Cancer Imaging Arch. (2017).
[19] B.H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et
al., The multimodal brain tumor image segmentation benchmark (BRATS), IEEE
Trans. Med. Imaging 34 (10) (2015) 1993–2024.
[20] F. Garcea, A. Serra, F. Lamberti, L. Morra, Data augmentation for medical
imaging: A systematic literature review, Comput. Biol. Med. 152 (2023) 106391.
[21] J. Xu, M. Li, Z. Zhu, Automatic data augmentation for 3D medical image
segmentation, 2020, arXiv:2010.11695.
[22] E. Kerfoot, J. Clough, I. Oksuz, J. Lee, A. King, J. Schnabel, Left-ventricle
quantification using residual U-Net, in: Statistical Atlases and Computational
Models of the Heart. Atrial Segmentation and LV Quantification Challenges: 9th
International Workshop, STACOM 2018, Held in Conjunction with MICCAI 2018,
Granada, Spain, September 16, 2018, Revised Selected Papers, Springer Verlag,
2019, pp. 371–380.
[23] A. Myronenko, 3D MRI brain tumor segmentation using autoencoder
regularization, 2018, arXiv:1810.11654.
[24] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. Roth, D. Xu, Swin UNETR: Swin
transformers for semantic segmentation of brain tumors in MRI images, 2022,
arXiv:2201.01266.
[25] E. Yagis, A.G.S.D. Herrera, L. Citi, Generalization performance of deep learning
models in neurodegenerative disease classification, in: 2019 IEEE International
Conference on Bioinformatics and Biomedicine, BIBM, 2019, pp. 1692–1698.
[26] E. Yagis, S. Atnafu, A.G.S.D. Herrera, C. Marzi, R. Scheda, M. Giannelli, C. Tessa,
L. Citi, S. Diciotti, Effect of data leakage in brain MRI classification using 2D
convolutional neural networks, Sci. Rep. 11 (2021).
[27] K. Amunts, H. Mohlberg, S. Bludau, K. Zilles, Julich-Brain: A 3D probabilistic
atlas of the human brain’s cytoarchitecture, Sci. (N. Y. N. Y.) 369 (2020).
[28] GemmaTeam, GemMa: Open models based on Gemini research and technology,
2024, arXiv (Cornell University).
[29] AI@Meta, Llama 3 model card, 2024.
[30] A.Q. Jiang, et al., Mixtral of experts, 2024, arXiv:2401.04088.
[31] A. Paszke, et al., PyTorch: An imperative style, high-performance deep learning
library, 2019, arXiv:1912.01703.
[32] M.J. Cardoso, et al., MONAI: An open-source framework for deep learning in
healthcare, 2022, arXiv:2211.02701.
[33] M. Brett, NiBabel: Read and write access to common neuroimaging file formats,
2024.
[34] A. Abraham, F. Pedregosa, M. Eickenberg, P. Gervais, A. Mueller, J. Kossaifi,
A. Gramfort, B. Thirion, G. Varoquaux, Machine learning for neuroimaging with
scikit-learn, Front. Neuroinformatics 8 (2014).
[35] T. Dickscheid, X. Gui, A.N. Simsek, L. Koehnen, V. Marcenko, C. Schiffer, S.
Bludau, K. Amunts, Siibra-python, 2024.
[36] Groq, Groq: AI acceleration technologies, 2024, URL https://groq.com.
[37] M. Honnibal, I. Montani, S.V. Landeghem, A. Boyd, SpaCy: Industrial-strength
natural language processing in Python, 2020.
[38] G. Martínez, J.A. Hernández, J. Conde, P. Reviriego, E. Merino-Gómez, Beware of
words: Evaluating the lexical diversity of conversational LLMs using ChatGPT as
case study, ACM Trans. Intell. Syst. Technol. (2024) http://dx.doi.org/10.1145/
3696459.
[39] R. Flesch, How to Write Plain English, University of Canterbury, 2016.
[40] N. Razali, et al., Power comparisons of shapiro-wilk, kolmogorov-smirnov,
lilliefors and anderson-darling tests, J. Stat. Model. Anal. 2 (1) (2011) 21–33.
[41] O. Commowick, et al., Multiple sclerosis lesions segmentation from multiple
experts: The MICCAI 2016 challenge dataset, NeuroImage 244 (2021) 118589.
[42] A. Carass, et al., Longitudinal multiple sclerosis lesion segmentation data
resource, Data Brief 12 (2017) 346–350.
[43] Z. Lesjak, A. Galimzianova, A. Koren, M. Lukin, F. Pernus, B. Likar, Z. Spiclin,
A novel public MR image dataset of multiple sclerosis patients with lesion
segmentations based on multi-rater consensus, Neuroinformatics 16 (1) (2017)
51–63.
[44] A. Malinin, et al., Shifts 2.0: Extending the dataset of real distributional shifts,
2022, arXiv (Cornell University).
[45] V. Raina, N. Molchanova, M. Graziani, A. Malinin, H. Muller, M.B. Cuadra, M.
Gales, Tackling bias in the dice similarity coefficient: Introducing NDSC for white
matter lesion segmentation, in: 2023 IEEE 20th International Symposium on
Biomedical Imaging, ISBI, 2023, pp. 1–5.
[46] A. Malinin, Uncertainty Estimation in Deep Learning with Application to Spoken
Language Assessment (Ph.D. thesis), University of Cambridge, 2019.
Computer Methods and Programs in Biomedicine 270 (2025) 108922
13