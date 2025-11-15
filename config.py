from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    d_model: int = 512  # Hidden dimension 크기 (256 → 512로 증가)
    nhead: int = 8  # Multi-head attention heads
    num_encoder_layers: int = 4  # Transformer encoder layers
    num_decoder_layers: int = 4  # Transformer decoder layers
    dim_feedforward: int = 2048  # FFN hidden dimension
    dropout: float = 0.1  # Dropout rate
    use_transformer: bool = True  # Transformer 사용 여부 (False면 GRU)


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 3e-4  # 더 작은 learning rate (안정적 학습)
    warmup_steps: int = 1000  # Warmup steps
    valid_every: int = 200  # 검증 주기 증가 (50 → 200)
    max_gen_len: int = 50  # 더 긴 출력 허용 (32 → 50)
    show_valid_samples: int = 5
    num_epochs: int = 10  # 더 긴 학습 (4 → 10)
    save_best_path: Optional[str] = None
    label_smoothing: float = 0.1  # Label smoothing
    grad_clip: float = 1.0  # Gradient clipping
    label_smoothing: float = 0.1  # Label smoothing
    grad_clip: float = 1.0  # Gradient clipping


