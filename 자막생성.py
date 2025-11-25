import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import gc
import whisper
from transformers import pipeline
import torch

class SubtitleTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 모델 기반 자막생성")
        self.root.geometry("400x850")

        self.whisper_model = None
        self.nllb_model = None

        self.gpuinfo_whisper = {
            "tiny": "CPU, GTX1050 2GB~",
            "base": "CPU, GTX1050 2GB~",
            "small": "GTX1650 4GB~",
            "medium": "RTX3060 6GB~",
            "large": "RTX3060 12GB~",
            "turbo": "RTX3060 12GB~",
            "large-v2": "RTX3060 12GB~",
            "large-v3": "RTX3060 12GB~",
        }
        self.gpuinfo_nllb = {
            "NLLB-200-600M": "CPU, GTX1050 2GB~",
            "NLLB-200-1.3B": "GTX1650 4GB~",
            "NLLB-200-3.3B": "RTX3060 12GB~",
        }

        self.language_codes = {
            "영어": "eng_Latn",
            "한국어": "kor_Hang",
            "일본어": "jpn_Jpan",
            "중국어(간체)": "zho_Hans",
            "스페인어": "spa_Latn"
        }

        self.init_ui()

    def init_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 파일 관리
        file_frame = ttk.LabelFrame(main_frame, text="파일 관리", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="비디오 선택", command=self.select_files).pack(side=tk.LEFT)
        self.file_list = tk.Listbox(file_frame, height=8)
        self.file_list.pack(fill=tk.X, pady=5)

        # 모델 설정
        model_frame = ttk.LabelFrame(main_frame, text="모델 설정", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Whisper 모델:").grid(row=0, column=0, sticky=tk.W)
        self.whisper_model_var = ttk.Combobox(
            model_frame,
            values=['tiny', 'base', 'small', 'medium', 'large', 'turbo', 'large-v2', 'large-v3']
        )
        self.whisper_model_var.set('medium')
        self.whisper_model_var.grid(row=0, column=1)

        ttk.Label(model_frame, text="NLLB200 모델:").grid(row=1, column=0, sticky=tk.W)
        self.nllb_model_var = ttk.Combobox(
            model_frame,
            values=['NLLB-200-600M', 'NLLB-200-1.3B', 'NLLB-200-3.3B']
        )
        self.nllb_model_var.set('NLLB-200-1.3B')
        self.nllb_model_var.grid(row=1, column=1)

        ttk.Label(model_frame, text="연산 방식:").grid(row=2, column=0, sticky=tk.W)
        self.device_var = ttk.Combobox(
            model_frame,
            values=['cpu', 'gpu']
        )
        # GPU 사용 가능 시 기본값 'gpu' 설정
        self.device_var.set('gpu' if torch.cuda.is_available() else 'cpu')
        self.device_var.grid(row=2, column=1)

        # 추천 GPU 정보
        self.gpu_label_whisper = ttk.Label(model_frame, text="Whisper 추천: 모델을 선택하세요.")
        self.gpu_label_whisper.grid(row=3, column=0, columnspan=2, pady=2)
        self.gpu_label_nllb = ttk.Label(model_frame, text="NLLB200 추천: 모델을 선택하세요.")
        self.gpu_label_nllb.grid(row=4, column=0, columnspan=2, pady=2)

        self.whisper_model_var.bind("<<ComboboxSelected>>", self.update_gpuinfo)
        self.nllb_model_var.bind("<<ComboboxSelected>>", self.update_gpuinfo)

        # 언어 설정
        lang_frame = ttk.LabelFrame(main_frame, text="언어 설정", padding=10)
        lang_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lang_frame, text="입력 언어:").grid(row=0, column=0)
        self.src_lang = ttk.Combobox(lang_frame, values=list(self.language_codes.keys()))
        self.src_lang.set("영어")
        self.src_lang.grid(row=0, column=1)
        ttk.Label(lang_frame, text="출력 언어:").grid(row=1, column=0)
        self.tgt_lang = ttk.Combobox(lang_frame, values=list(self.language_codes.keys()))
        self.tgt_lang.set("한국어")
        self.tgt_lang.grid(row=1, column=1)

        # 진행 표시줄
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        # 로그 영역
        self.log_area = scrolledtext.ScrolledText(main_frame, height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)

        # 시작 버튼
        ttk.Button(main_frame, text="처리 시작", command=self.start_processing).pack(pady=10)

        self.update_gpuinfo() # 초기 GPU 정보 업데이트

    def update_gpuinfo(self, event=None):
        whisper_name = self.whisper_model_var.get()
        nllb_name = self.nllb_model_var.get()
        info_w = self.gpuinfo_whisper.get(whisper_name, "모델을 선택하세요.")
        info_n = self.gpuinfo_nllb.get(nllb_name, "모델을 선택하세요.")
        self.gpu_label_whisper.config(text=f"Whisper 추천: {info_w}")
        self.gpu_label_nllb.config(text=f"NLLB200 추천: {info_n}")

    def select_files(self):
        files = filedialog.askopenfilenames(
            title='비디오 파일 선택',
            filetypes=(('비디오 파일', '*.mp4 *.avi *.mkv *.mov'), ('모든 파일', '*.*'))
        )
        self.file_list.delete(0, tk.END)
        for file in files:
            self.file_list.insert(tk.END, file)

    def log(self, message):
        self.log_area.insert(tk.END, f"{message}\n")
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def generate_subtitles(self, file_path):
        try:
            base_name = os.path.splitext(file_path)[0]
            srt_path = f"{base_name}.srt"
            self.log(f"자막 생성 중: {os.path.basename(file_path)}")
            
            # Whisper의 transcribe 함수를 사용하여 자막 생성
            result = self.whisper_model.transcribe(file_path)
            
            # SRT 파일로 저장
            writer = whisper.utils.get_writer("srt", os.path.dirname(file_path))
            writer(result, os.path.basename(srt_path))
            return srt_path
        except Exception as e:
            self.log(f"자막 생성 오류: {str(e)}")
            return None

    def translate_subtitle(self, srt_path, src_code, tgt_code):
        try:
            self.log(f"번역 시작: {os.path.basename(srt_path)}")
            
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            translated_content = ""
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                
                # 자막 블록이 유효한지 확인 (번호, 시간, 텍스트)
                if len(lines) < 3:
                    translated_content += block + '\n\n'
                    continue
                    
                header = '\n'.join(lines[:2]) # 번호와 시간 정보
                text = '\n'.join(lines[2:]) # 실제 자막 텍스트
                
                if text.strip():
                    # NLLB 모델을 사용하여 텍스트 번역
                    translated = self.nllb_model(
                        text,
                        src_lang=src_code,
                        tgt_lang=tgt_code,
                        max_length=512
                    )[0]['translation_text']
                else:
                    translated = "" # 텍스트가 없으면 빈 문자열 유지
                    
                translated_content += f"{header}\n{translated}\n\n"
                
            translated_path = f"{os.path.splitext(srt_path)[0]}_translated.srt"
            
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
                
            return True
            
        except Exception as e:
            self.log(f"번역 오류: {str(e)}")
            return False

    def start_processing(self):
        """처리 파이프라인을 시작하는 스레드를 실행합니다."""
        files = [self.file_list.get(i) for i in range(self.file_list.size())]
        if not files:
            messagebox.showwarning("경고", "파일을 선택해주세요")
            return
        # 메인 UI를 멈추지 않도록 백그라운드 스레드에서 실행
        threading.Thread(target=self.process_pipeline, args=(files,), daemon=True).start()

    def cleanup(self, device_mode):
        if self.whisper_model:
            del self.whisper_model
            self.whisper_model = None
        if self.nllb_model:
            del self.nllb_model
            self.nllb_model = None
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # GPU 모드인 경우 CUDA 메모리 캐시 비우기
        if device_mode == "gpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_pipeline(self, files):
        self.progress.start()
        device_mode = self.device_var.get()
        actual_device = device_mode
        
        try:
            # 1. 실제 연산 장치 설정 및 확인
            if device_mode == "gpu" and not torch.cuda.is_available():
                actual_device = "cpu"
                self.log("경고: GPU 모드로 설정되었으나 CUDA를 사용할 수 없습니다. CPU로 대체합니다.")
            
            # --- 2. WHISPER 모델 로드 (자막 생성) ---
            model_name = self.whisper_model_var.get()
            self.log(f"Whisper 모델 로드 중... ({model_name}, device: {actual_device})")
            
            # **모델 로드 안정화 로직 적용 (UntypedStorage 오류 방지)**
            if actual_device == "gpu":
                 # 'cuda' 대신 명시적인 device=None을 사용하거나, 
                 # 안정적인 장치 이름 'cuda'를 사용합니다.
                 self.whisper_model = whisper.load_model(model_name, device="cuda")
            else:
                 self.whisper_model = whisper.load_model(model_name, device="cpu")

            # 자막 생성
            subtitle_paths = []
            for file in files:
                srt_path = self.generate_subtitles(file)
                if srt_path:
                    subtitle_paths.append(srt_path)
                    
            # --- 3. WHISPER 모델 언로드 (메모리 확보) ---
            self.log("Whisper 모델 언로드 중...")
            self.cleanup(device_mode) # cleanup 함수를 이용해 메모리 정리

            # --- 4. NLLB 모델 로드 (번역) ---
            src_lang_name = self.src_lang.get()
            tgt_lang_name = self.tgt_lang.get()

            if src_lang_name == tgt_lang_name:
                # 입력 언어와 출력 언어가 같으면 번역 건너뛰기
                self.log("입력 언어와 출력 언어가 같아 번역 단계를 건너뜁니다.")
            else:
                nllb_name = self.nllb_model_var.get()
                self.log(f"NLLB200 모델 로드 중... ({nllb_name}, device: {device_mode})")
                
                if nllb_name == "NLLB-200-600M":
                    model_path = "facebook/nllb-200-distilled-600M"
                elif nllb_name == "NLLB-200-1.3B":
                    model_path = "facebook/nllb-200-distilled-1.3B"
                else:
                    model_path = "facebook/nllb-200-3.3B"

                # **NLLB 모델 로드 시 장치 설정 개선**
                if device_mode == "gpu" and torch.cuda.is_available():
                    # device=0은 첫 번째 GPU를 의미 (CUDA 인덱스)
                    device_param = 0 
                else:
                    # device=-1은 CPU를 의미
                    device_param = -1
                
                self.nllb_model = pipeline(
                    'translation',
                    model=model_path,
                    device=device_param
                )

                # 번역 실행
                src_code = self.language_codes[src_lang_name]
                tgt_code = self.language_codes[tgt_lang_name]
                
                for srt_path in subtitle_paths:
                    if self.translate_subtitle(srt_path, src_code, tgt_code):
                        self.log(f"번역 완료: {os.path.basename(srt_path).replace('.srt', '_translated.srt')}")
                        
            # --- 5. NLLB 모델 언로드 ---
            self.log("NLLB200 모델 언로드 중...")
            self.cleanup(device_mode)
            
            self.log("✅ 모든 처리가 완료되었습니다!")
            
        except Exception as e:
            self.log(f"🚨 전체 처리 오류: {str(e)}")
            self.cleanup(device_mode) # 오류 발생 시에도 정리
            
        finally:
            self.progress.stop()

if __name__ == "__main__":
    # CUDA 사용 가능 여부를 먼저 확인하고 PyTorch 라이브러리 사용 가능 여부를 점검합니다.
    if not (torch.cuda.is_available() or torch.has_mps):
        print("경고: CUDA 또는 MPS를 사용할 수 없습니다. 모든 처리는 CPU 모드로 진행됩니다.")
    
    root = tk.Tk()
    app = SubtitleTranslatorApp(root)
    root.mainloop()

