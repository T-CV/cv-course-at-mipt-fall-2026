# Список тем на устном экзамене

**Список вопросов находится в разработке! Когда он будет готов, мы вам дополнительно сообщим в чате!**

1. **Основы обработки изображений.** Представление изображения, цветовые пространства, базовые операции (яркость, контраст, пороговая обработка).

2. **Дескрипторы ключевых точек и склейка изображений.** Локальные признаки: SIFT, BRIEF, ORB; сопоставление дескрипторов, panorama stitching.

3. **Трекинг объектов.** Фильтр Калмана: модель состояния, предсказание и коррекция; применение к сопровождению объектов на видео.

4. **Основы глубоких нейросетей.** Полносвязные сети, функции активации, обратное распространение, регуляризация (dropout, нормализация).

5. **Свёрточные нейросети (CNN).** Свёртка, пулинг, архитектуры (AlexNet, VGG, ResNet и residual-связи), функции потерь и обучение свёрточных сетей.

6. **Механизм внимания и архитектура Transformer.** Scaled dot-product attention, self- и cross-attention, multi-head attention, маски (padding, causal), FFN, layer normalization, encoder/decoder блоки.

7. **Позиционные эмбеддинги.** Обучаемые, синусоидальные, RoPE, M-RoPE (мультимодальный/2D); зачем нужна позиционная информация и как она кодируется.

8. **Способы обучения трансформеров и инференс.** MLM, teacher forcing / seq2seq, causal language modeling (CLM), label masking и sequence packing, KV-cache и сложность авторегрессионной генерации. Mixture of Experts (MoE).

9. **Vision Transformer (ViT) и Swin Transformer.** Разбиение изображения на патчи, CLS-токен, позиционные эмбеддинги; Swin: window-based attention, shifted windows, иерархическая структура, relative position bias.

10. **Метрики.** IoU, mAP (mAP@0.5, mAP@[.5:.95])

11. **One-stage детекторы.** YOLO (v1/v3), RetinaNet (focal loss), anchor-free подходы, детекция как set prediction (DETR, Hungarian matching).

12. **Multi-object tracking (MOT).** SORT, DeepSORT, ByteTrack, BoT-SORT; связь трекинга с детекцией.

13. **Задачи и метрики сегментации.** Semantic / instance / panoptic segmentation; метрики IoU, Dice, pixel accuracy.

14. **Функции потерь для сегментации.** Per-pixel (BCE, CE, focal loss, soft CE), metric-based (Dice/IoU loss), продвинутые (Tversky, Lovasz-Softmax).

15. **Архитектуры сегментации.** U-Net, FPN, PAN, U-Net++, DeepLab v3; Mask R-CNN, трансформеры в сегментации (Mask2Former) и SAM.

16. **Self-supervised learning.** Контрастивное обучение (SimCLR), семейство DINO (v1/v2/v3); linear probing.

17. **Text-Image alignment.** CLIP, SigLIP и SigLIP 2; zero-shot применение и калибровка.

18. **Metric Learning: задачи и метрики.** Задачи (verification, identification, ReID), метрики (ROC-AUC, TAR/FAR, recall@k, mAP, CMC).

19. **Face Losses.** Norm-Softmax, SphereFace, CosFace, ArcFace, AdaFace.

20. **Приближённый поиск ближайших соседей (ANN).** LSH, HNSW, IVF-PQ, K-D tree; библиотеки (FAISS, HNSWlib, Annoy, Qdrant).

21. **Метрики качества генерации.** FID, Inception Score, Precision/Recall, aesthetic score, оценка text-to-image (CLIP score, VLM-судьи, Image Arena).

22. **VAE и VQ-VAE.** Автоэнкодеры, постановка VAE, ELBO, reparameterization trick, CVAE; VQ-VAE: codebook, quantization, straight-through estimator.

23. **GAN.** GAN, DCGAN, WGAN; масштабирование (BigGAN, StyleGAN), CycleGAN; сравнение VAE и GAN, гибридные модели.

24. **Авторегрессионные модели.** Постановка (MLE, chain rule), PixelCNN (masked convolutions), VAR (next-scale prediction), Switti и современные autoregressive модели.

25. **Normalizing Flows.** Change of Variables, RealNVP (coupling layer), Continuous Normalizing Flows, Flow Matching и Rectified Flow.

26. **Диффузионные модели.** Forward/reverse process, DDPM и ELBO, параметризация (denoising), classifier-free guidance, DDIM и ускорение сэмплинга.

27. **Latent Diffusion и Stable Diffusion.** LDM (VAE + conditioning + latent diffusion), Stable Diffusion v1/v2, SDXL (каскадные модели, второй текстовый энкодер).

28. **Тонкая настройка и адаптеры.** Дообучение больших моделей, adapters и LoRA, дистилляция знаний (response/feature-based), обзор SOTA методов дообучения.

29. **Архитектуры и обучение VLM.** Encoder-decoder vs decoder-only; история: GLIP, BLIP-2 (Q-Former), LLaVA (линейная проекция), PaLI; современные семейства (InternVL, Gemma, Qwen-VL, GLM); пайплайн обучения VLM (pretrain → SFT → RL); open-source vs closed-source.

30. **Практика работы с VLM.** Reasoning (CoT, GRPO, DeepSeek-R1), chat templates, visual grounding, tool calling и MCP, воспроизводимость ответов.

31. **Инференс и сервинг LLM/VLM.** Continuous batching и PagedAttention (vLLM), FlashAttention, prefix caching, параметры сэмплирования (temperature, top-k, top-p), speculative decoding.
