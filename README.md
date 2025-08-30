# FAST_GAR: Fused-Attention Spatio-Temporal Skeleton based Group Activity Recogntiion

Group activity recognition (GAR) involves identify-
ing activities performed by multiple individuals and remains a
complex yet practically relevant task. Although skeleton-based
methods offer lightweight and efficient representations, existing
GAR approaches still fall short in jointly exploiting spatial
and temporal features within the skeleton modality. To address
this, we propose FAST-GAR, a Fused-Attention Spatio-Temporal
Transformer for Skeleton based GAR that captures spatio-
temporal dependencies while balancing model complexity and
performance. FAST-GAR adopts a two-stage transformer design,
where a dual-stream module first extracts spatial and temporal
joint features for each individual. These features are effectively
integrated using a gated attention-based fusion mechanism and
passed to the second stage, which operates with a group-level
perspective, modeling dependencies among individuals by captur-
ing spatial arrangements and interaction patterns. Furthermore,
a proposed relative importance encoding mechanisms enable
a more context-aware and dynamic way to improve feature
extraction. Our model achieves accuracies of 88.63% and 86.91%
on the Volleyball and Collective Activity dataset, respectively,
surpassing several state-of-the-art approaches and validating its
effectiveness for group activity recognition.

## Dependencies

- Python >= 3.13
- PyTorch >= 2.7.1 (with CUDA 11.8 support)
- TorchVision >= 0.22.1
- TorchAudio >= 2.7.1

## Train

```bash
python main_fused.py --config config/train_fused.yml
