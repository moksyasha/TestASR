import nemo.collections.asr as nemo_asr
import time
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def speech_to_text(file_path):
    waveform, sample_rate = load_audio(file_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]


def main():
    #https://huggingface.co/nvidia/stt_ru_conformer_transducer_large
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_ru_conformer_transducer_large")
    # get transcr by using nemo // 0.04 sec
    transcription = asr_model.transcribe(["trash.wav"])
    start_time = time.time()
    transcription = asr_model.transcribe(["trash.wav"])
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time} sec")
    print(transcription[0])

    # get transcr by using wav2vec // 0.19 sec
    transcription = speech_to_text("trash.wav")
    start_time = time.time()
    transcription = speech_to_text("trash.wav")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time: {execution_time} sec")
    print("Распознанный текст:", transcription)

if __name__ == "__main__":
    main()