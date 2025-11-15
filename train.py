from __future__ import annotations
from typing import List, Any, Tuple
from config import TrainConfig, ModelConfig, TokenizerConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
)
from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (
    TinySeq2Seq,
    CharTokenizer,      # 문자 단위 토크나이저
    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수
    INPUT_CHARS,        # 입력 문자 집합
    OUTPUT_CHARS,       # 출력 문자 집합
)


# ======================================================================================
# 1. 학습 루프
# ======================================================================================
def train_loop(
    model: nn.Module,           # 학습할 모델
    dataloader: DataLoader,     # DataLoader를 직접 전달받아 사용합니다.
    input_tokenizer: CharTokenizer,      # (tokenize_batch 유틸리티를 위해 유지)
    output_tokenizer: CharTokenizer,     # 출력 문자 토크나이저
    device: torch.device,       # cpu 또는 cuda
    val_dataloader: DataLoader | None = None,  # 별도 검증 DataLoader (None이면 훈련 배치로 검증)
    *,
    train_config: TrainConfig,  # 학습 설정
    model_config: ModelConfig,  # 모델 설정
    tokenizer_config: TokenizerConfig,  # 토크나이저 설정
):
    # 모델을 GPU/CPU로 보냄
    model.to(device)

    # 옵티마이저: AdamW는 Adam + weight decay가 들어간 버전
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    
    # 🔥 Learning Rate Scheduler (Warmup + Cosine Annealing)
    def get_lr(step, warmup_steps, max_steps, base_lr):
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step + 1) / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    # 총 스텝 수 계산
    total_steps = len(dataloader) * train_config.num_epochs
    
    # 🔥 Label Smoothing 적용
    # pad 토큰은 무시하도록(ignore_index) 설정
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_tokenizer.pad_id,
        label_smoothing=train_config.label_smoothing,  # Label smoothing 추가
    )

    step = 0
    model.train()  # 학습 모드로 전환 (Dropout 등 켜짐)

    # tqdm은 진행 상황을 예쁘게 보여주는 라이브러리입니다.
    pbar = tqdm(total=train_config.max_train_steps if train_config.max_train_steps is not None else None, desc="train", unit="step", ncols=120, dynamic_ncols=True, leave=True)

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")

    for epoch in range(train_config.num_epochs):
        # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps: break
        pbar.write(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")

        # 실제로 배치를 하나씩 뽑아서 학습하는 부분입니다.
        for batch in dataloader:
            # --------------------------------------------------------------
            # 1) 토크나이즈 & 텐서로 변환
            #    `tokenize_batch`는 BatchTensors(src, tgt_inp, tgt_out)를 반환합니다.
            #    변수 역할:
            #      - `src` (encoder input): 모델의 인코더 입력. 정수 텐서, shape (B, S).
            #      - `target_input` (tgt_inp): 디코더에 teacher-forcing으로 넣는 입력. shape (B, T).
            #          일반적으로 BOS를 앞에 붙이고 EOS는 제외한 시퀀스입니다.
            #      - `target_output` (tgt_out): 디코더가 예측해야 하는 정답(손실 대상). shape (B, T).
            #          일반적으로 target_input에서 BOS를 뺀 것에 EOS를 붙인 형태입니다.
            #    예시 (토큰 id가 다음과 같다고 가정):
            #      bos_id=1, eos_id=2, '1'->5, '6'->6
            #      원본 target_text: "16"
            #      target_input ids:  [1, 5, 6]    # [BOS, '1', '6']
            #      target_output ids: [5, 6, 2]    # ['1', '6', EOS']
            #    주의: 모든 텐서는 dtype=torch.long이고 `.to(device)`로 명시적 이동이 필요합니다.
            # --------------------------------------------------------------
            batch_tensors = tokenize_batch(batch, input_tokenizer, output_tokenizer)
            src = batch_tensors.src.to(device)
            target_input = batch_tensors.tgt_inp.to(device)
            target_output = batch_tensors.tgt_out.to(device)

            # Forward: 모델에 입력을 전달하고 출력을 얻습니다.
            # 출력은 (B, T, V) 형태로 반환됩니다.
            logits = model(src, target_input, input_tokenizer.pad_id)

            # --------------------------------------------------------------
            # 4) Loss 계산
            # --------------------------------------------------------------
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                target_output.view(-1),             # (B*T,)
            )

            # --------------------------------------------------------------
            # 5) Backward + optimizer step
            # --------------------------------------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optim.step()
            optim.zero_grad()
            
            # 🔥 Learning Rate 업데이트 (Warmup + Cosine Annealing)
            new_lr = get_lr(step, train_config.warmup_steps, total_steps, train_config.lr)
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr

            step += 1 

            # --------------------------------------------------------------
            # 6) Validation
            # --------------------------------------------------------------
            if step % train_config.valid_every == 0:
                model.eval()  # 평가 모드
                # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                with torch.no_grad():
                    preds_all: List[str] = []
                    targets_all: List[str] = []
                    inputs_all: List[str] = [] # 검증 데이터셋의 입력, 정답, 예측 결과를 저장할 리스트

                    for val_batch in val_dataloader: # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                        val_bt = tokenize_batch(val_batch, input_tokenizer, output_tokenizer)
                        val_src = val_bt.src.to(device)
                        gen_ids = model.generate(
                            src=val_src,
                            max_len=train_config.max_gen_len,
                            bos_id=output_tokenizer.bos_id,
                            eos_id=output_tokenizer.eos_id,
                            src_pad_id=input_tokenizer.pad_id,
                        )

                        for i in range(gen_ids.size(0)):
                            seq_chars: List[str] = []
                            for t in gen_ids[i].tolist():
                                idx = int(t)
                                if idx == output_tokenizer.eos_id:
                                    break
                                if idx in output_tokenizer.itos:
                                    ch = output_tokenizer.itos[idx]
                                    if ch.isdigit() or (ch == '-' and not seq_chars):
                                        seq_chars.append(ch)
                            pred_str = "".join(seq_chars)
                            if pred_str == "-":
                                pred_str = ""
                            preds_all.append(pred_str)
                        # 검증 데이터셋의 정답, 입력을 리스트에 추가합니다.
                        targets_all.extend(val_batch["target_text"])
                        inputs_all.extend(val_batch["input_text"])

                    # 검증 데이터셋의 예측, 정답을 사용하여 성능 지표를 계산합니다.
                    em_batch = compute_metrics(preds_all, targets_all)

                    # 진행바에도 성능을 표시합니다.
                    pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f}")
                    pbar.set_postfix(
                        EM=f"{em_batch['EM']:.3f}",
                        TES=f"{em_batch['TES']:.3f}",
                    )
                    
                    pbar.refresh()

                    # 최고 성능 갱신 시 전체 체크포인트 저장
                    if train_config.save_best_path is not None:
                        current_em = float(em_batch.get("EM", -1.0))
                        if current_em > best_em:
                            best_em = current_em
                            # 세 config를 dict로 변환하여 저장
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,  # 학습 설정 저장
                                "model_config": model_config.__dict__,  # 모델 설정 저장
                                "tokenizer_config": tokenizer_config.__dict__,  # 토크나이저 설정 저장
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            pbar.write(f"New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")

                    B = len(preds_all) # 검증 데이터셋의 크기
                    n_show = min(train_config.show_valid_samples, B)
                    pbar.write("Sample validation output:") # 예시로 몇 개만 보여줍니다.
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "OK" if pred == tgt else "ERR"
                        pbar.write(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
                model.train()  # 다시 학습 모드로

            # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                break # 학습을 종료합니다.

            pbar.update(1) # tqdm 진행 1 step

# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 1) 데이터 준비 (개선된 설정)
    # --------------------------------------------------------------------------
    
    # Train Dataset
    # 🔥 핵심 개선: 5자리까지 학습 (규정 최대치), depth 증가, 샘플 증가
    train_dataset = ArithmeticDataset(
        num_samples=1_000_000,  # 500K → 1M (더 많은 학습 데이터)
        max_depth=4,            # 3 → 4 (더 복잡한 수식)
        num_digits=(1, 5),      # (1,3) → (1,5) (규정 최대치, OOD 간극 감소!)
        seed=123,
        mode="train",
    )

    # Train DataLoader
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=64,  # 128 → 64 (Transformer는 메모리 많이 사용)
        num_workers=0,
        pin_memory=True,
    )

    # Validation Dataset
    # 검증은 더 어렵게 (6자리까지 포함)
    val_dataset = ArithmeticDataset(
        num_samples=256,        # 128 → 256 (더 많은 검증 샘플)
        max_depth=5,            # 4 → 5 (더 복잡)
        num_digits=(1, 5),      # 학습과 동일 (In-domain)
        seed=999,
        mode="val",
    )
    
    # Validation DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 2) 토크나이저 설정 준비
    # --------------------------------------------------------------------------
    # 토크나이저 설정 (별도로 관리)
    tokenizer_config = TokenizerConfig(
        input_chars=INPUT_CHARS,
        output_chars=OUTPUT_CHARS,
        add_special=True,
    )

    # --------------------------------------------------------------------------
    # 3) 토크나이저 준비
    # --------------------------------------------------------------------------
    # 입력 문자 토크나이저, 출력 문자 토크나이저를 준비합니다. 자세한 설정은 model.py를 참고하세요.
    input_tokenizer = CharTokenizer(
        tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )
    output_tokenizer = CharTokenizer(
        tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )

    # --------------------------------------------------------------------------
    # 4) 모델 설정 준비 (개선된 설정)
    # --------------------------------------------------------------------------
    # 🔥 Transformer 사용
    model_config = ModelConfig(
        d_model=512,             # 256 → 512 (더 큰 모델)
        nhead=8,                 # Multi-head attention
        num_encoder_layers=4,    # Encoder layers
        num_decoder_layers=4,    # Decoder layers
        dim_feedforward=2048,    # FFN dimension
        dropout=0.1,             # Dropout
        use_transformer=True,    # Transformer 사용!
    )

    # --------------------------------------------------------------------------
    # 5) 학습 설정 준비 (개선된 설정)
    # --------------------------------------------------------------------------
    train_config = TrainConfig(
        max_train_steps=None,
        lr=3e-4,                 # 2e-3 → 3e-4 (더 작은 learning rate)
        warmup_steps=1000,       # Warmup 추가
        valid_every=200,
        max_gen_len=50,          # 24 → 50 (더 긴 출력)
        show_valid_samples=5,
        num_epochs=10,           # 50 → 10 (Transformer는 빠르게 수렴)
        save_best_path="best_model.pt",
        label_smoothing=0.1,     # Label smoothing 추가
        grad_clip=1.0,
    )

    # --------------------------------------------------------------------------
    # 6) 모델 준비 (Transformer 또는 GRU)
    # --------------------------------------------------------------------------
    if model_config.use_transformer:
        # 🔥 TransformerSeq2Seq 사용
        from model import TransformerSeq2Seq
        model = TransformerSeq2Seq(
            in_vocab=input_tokenizer.vocab_size,
            out_vocab=output_tokenizer.vocab_size,
            **model_config.__dict__,
        )
    else:
        # 기존 GRU 모델
        model = TinySeq2Seq(
            in_vocab=input_tokenizer.vocab_size,
            out_vocab=output_tokenizer.vocab_size,
            **model_config.__dict__,
        )

    # --------------------------------------------------------------------------
    # 6) 학습 시작
    # --------------------------------------------------------------------------

    train_loop(
        model=model,
        dataloader=train_dataloader,              # 미리 만든 DataLoader를 직접 전달합니다.
        input_tokenizer=input_tokenizer,        # (tokenize_batch 유틸리티를 위해 전달)
        output_tokenizer=output_tokenizer,
        device=device,
        val_dataloader=val_dataloader,
        train_config=train_config,  # 학습 설정 전달
        model_config=model_config,  # 모델 설정 전달
        tokenizer_config=tokenizer_config,  # 토크나이저 설정 전달
    )

    # --------------------------------------------------------------------------
    # 6) 학습 끝나면 모델 저장
    # --------------------------------------------------------------------------
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


# python train.py로 실행했을 때만 main()을 돌게 합니다.
if __name__ == "__main__":
    main()
