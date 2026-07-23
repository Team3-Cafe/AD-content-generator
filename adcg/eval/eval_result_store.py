"""# 이미지별 평가 점수 통합 저장

네 평가 스크립트가 중간 파일 없이 동일한 `eval_results.json`을 순차적으로
갱신하도록 지원한다.

## 저장 형식

```json
{
  "image_id": "product.png",
  "clip_score": 0.8,
  "aesthetic_score": 5.1,
  "dino_similarity": 0.9,
  "hps_v2_score": 0.3
}
```

새 이미지는 네 점수를 `null`로 초기화하고 현재 평가가 전달한 점수만
채운다. 같은 `image_id`가 다시 들어오면 해당 지표의 이전 값을 덮어쓴다.
"""
import json
from pathlib import Path


SCORE_FIELDS = (
    "clip_score",
    "aesthetic_score",
    "dino_similarity",
    "hps_v2_score",
)


def update_eval_results(output_path, results, score_field):
    """이미지별 점수 하나를 기존 최종 JSON에 병합한다.

    Args:
        output_path: 최종 `eval_results.json` 경로.
        results: `image_id`와 현재 점수 필드를 가진 딕셔너리 목록.
        score_field: 갱신할 점수 필드 이름.
    """
    output_path = Path(output_path)
    # ## 1. 앞선 평가가 기록한 점수가 있으면 먼저 불러온다.
    existing = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as file:
            existing = json.load(file)

    # 반복 검색을 피하기 위해 기존 배열을 image_id 기반 딕셔너리로 바꾼다.
    by_image_id = {row["image_id"]: row for row in existing}

    # ## 2. 현재 지표만 이미지별 행에 병합한다.
    for result in results:
        image_id = result["image_id"]
        # 처음 등장한 이미지는 모든 점수를 null로 초기화한다.
        row = by_image_id.setdefault(
            image_id,
            {
                "image_id": image_id,
                **{field: None for field in SCORE_FIELDS},
            },
        )
        # 이미 계산된 다른 지표는 건드리지 않고 현재 필드만 덮어쓴다.
        row[score_field] = result[score_field]

    # ## 3. 출력 순서를 고정하고 UTF-8 JSON으로 저장한다.
    combined = sorted(
        by_image_id.values(),
        key=lambda row: row["image_id"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(combined, file, ensure_ascii=False, indent=2)