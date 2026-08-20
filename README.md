# 패턴/그리드 각도 추정 (find_pattern_angle)

🔗 **웹에서 바로 보기**: [프로젝트 소개](https://kylelee6.github.io/find_pattern_angle/) ·
[웹 도구로 직접 실행](https://kylelee6.github.io/find_pattern_angle/tool.html)

## 개요

반도체 웨이퍼 다이(Die) 그리드처럼 반복되는 줄무늬 패턴이 있는 영상 **한 장**에서, 그 패턴이 이미지 축
대비 몇 도 기울어져 있는지를 라돈(Radon) 변환 방식으로 추정합니다. 두 영상을 비교해 회전중심을 구하는
[find_rotation_center](https://github.com/kylelee6/find_rotation_center)와 달리, 이미지 한 장만으로
그리드/패턴 자체의 절대 방향(0~180° 주기)을 구합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 단일 이미지 각도 추정

```bash
python estimate_pattern_angle.py images/sample1.jpg
```

- 콘솔에 coarse 각도와 최종(서브픽셀) 각도를 출력합니다.
- `images/sample1_annotated.png`를 저장합니다 (원본 위에 각도 화살표 + 숫자).
- `--csv result.csv` 옵션으로 결과를 CSV로 저장할 수 있습니다.

### 2. 폴더 일괄 처리

```bash
python estimate_pattern_angle.py images
```

폴더 안의 모든 이미지를 처리하고 `images/angles.csv`를 생성합니다.

### 주요 옵션

- `--step` : coarse 탐색 각도 스텝 (기본 0.1°)
- `--keep-ratio` : 중앙 원형 마스크 비율 (웨이퍼처럼 둥근 에지가 있는 경우 그 영향을 제거)
- `--max-size` : 다운샘플 최대 변 (속도용, 기본 768px)
- `--arrow {line,normal}` : 화살표 방향 — `line`(그리드 선 방향, 기본) 또는 `normal`(라돈 법선 방향)

## 검증: 각도를 알고 있는 테스트 이미지로 정확도 확인하기

```bash
python gen_rotate_image.py images/sample1.jpg rot_out --angles 0:170:10
python estimate_pattern_angle.py rot_out
```

`gen_rotate_image.py`가 만든 `rot_out/angles.csv`(생성 시 사용한 정답 각도)와
`estimate_pattern_angle.py`가 만든 `rot_out/angles.csv`(추정 각도)를 비교하면 정확도를 확인할 수 있습니다.

## 예제 이미지로 확인해보기

저장소에 포함된 두 이미지(`images/sample1.jpg`, `images/sample1_30.jpg`)는 같은 웨이퍼 다이를 촬영한 뒤
하나를 약 30° 회전시킨 것입니다.

| 이미지 | 추정 각도 |
|---|---|
| `images/sample1.jpg` | 179.67° (≡ -0.33°, 180° 주기) |
| `images/sample1_30.jpg` | 29.62° |

두 값의 차이는 29.95°로, 실제 회전(30°)과 0.05° 이내로 일치합니다.

## 웹에서 바로 실행해보기

파이썬 설치 없이 브라우저에서 이미지 한 장을 올려 바로 각도를 계산해볼 수 있습니다:
**[tool.html](https://kylelee6.github.io/find_pattern_angle/tool.html)** — 모든 계산은
OpenCV.js(WebAssembly)로 브라우저 안에서 실행되며, 이미지는 서버로 전송되지 않습니다.

웹 버전은 `estimate_pattern_angle.py`의 핵심 알고리즘(회전 → 열 합 → 고주파 에너지로 최적각 탐색,
coarse-to-fine)을 자바스크립트로 재현한 시연용입니다. 정밀 분석/배치 처리에는 파이썬 스크립트를 사용하세요.

## 알고리즘 요약

1. 그레이스케일 변환 + Hanning 윈도우(경계 아티팩트 억제)를 적용하고, 필요하면 중앙 원형 마스크로
   웨이퍼의 둥근 에지 영향을 제거합니다.
2. 0~180°를 성기게 훑으며, 각 각도로 영상을 회전한 뒤 열(column) 합으로 1D 프로파일을 만들고,
   그 프로파일의 1차 차분 에너지(고주파 성분)를 지표로 씁니다 — 그리드 선과 정확히 정렬될 때
   열 경계가 급격히 바뀌어 에너지가 최대가 됩니다.
3. coarse 최댓값 근방만 더 촘촘한 스텝 + 3차보간(cubic 보간)으로 재탐색합니다.
4. 3점 포물선 보간으로 서브픽셀(서브디그리) 각도를 산출합니다.

## 파일 구성

- `estimate_pattern_angle.py` — 메인 각도 추정 스크립트
- `gen_rotate_image.py` — 알려진 각도로 회전된 테스트 이미지 생성 (검증용)
- `images/` — 샘플 이미지 (`sample1.jpg`, `sample1_30.jpg`)
- `tool.html`, `index.html` — 브라우저 데모 (GitHub Pages)

## 주의

- **180° 주기**: 그리드 패턴은 90°/180° 대칭성이 있어 절대 방향이 아니라 (mod 180°) 각도로 나옵니다.
- 반복 패턴이 뚜렷할수록 정확도가 높습니다. 텍스트 오버레이 등 비주기적 구조가 섞여 있어도 그리드가
  지배적이면 대체로 안정적으로 동작합니다 (`images/sample1.jpg` 참고).
