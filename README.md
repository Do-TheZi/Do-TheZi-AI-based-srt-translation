***

# 자막 생성기 & 번역기 (Whisper + NLLB-200 기반)

## 개요

이 프로그램은 OpenAI의 Whisper(음성 인식)와 Meta의 NLLB-200(기계번역)을 결합한 **GUI 자막 생성 및 번역 자동화 도구**입니다.
- 동영상의 음성을 Whisper로 자동 자막화(SRT)
- SRT 파일을 NLLB-200으로 다양한 언어로 자동 번역
- 다양한 GPU/CPU 환경에서 구동 선택 가능
- 주요 언어 지원: 한국어(kor), 영어(eng), 일본어(jpn), 중국어(zh), 스페인어(es)
- 직관적인 Tkinter 기반 GUI 지원

## 주요 기능

- 다중 영상 파일 일괄 자막 생성
- Whisper 음성 인식 모델(여러 등급) 및 NLLB-200 번역 모델 선택
- GPU/CPU 선택 지원 (환경정보 자동 인식)
- SRT 자막 원본과 번역 파일 동시 관리
- 진행상황, 로그 표시 및 직관적인 상태 알림

## 사용법

1. 프로그램 실행 후, GUI에서 변환할 영상 파일(.mp4, .avi 등) 선택
2. 사용 모델(Whisper, NLLB-200), 장치(GPU/CPU), 소스/목적 언어 선택
3. [시작] 버튼 클릭 시 자동으로 자막 생성 및 번역까지 일괄 완료

## 라이센스

- **NLLB-200**: CC-BY-NC 4.0 (비상업적/저작자표시)
- **Whisper**: MIT License
- **본 소스코드 전체**: CC-BY-NC 4.0
    - 비상업적 용도만 허용

```
본 소프트웨어는 연구 및 개인적 비상업적 용도에 한하여 사용될 수 있습니다. 상업적 이용이나 2차 배포는 CC-BY-NC 4.0 라이선스를 반드시 준수해야 합니다.
```

## 저작권 고지 및 참고

- Meta AI: NLLB-200(https://ai.meta.com/research/no-language-left-behind/)
- OpenAI: Whisper(https://github.com/openai/whisper)
